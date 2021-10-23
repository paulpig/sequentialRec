from ..base import BaseModel
from .embeddings import *
from .bodies import ExactSasBody

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class Caser(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.output_info = args.output_info
        self.token_embedding = TokenEmbedding(args)
        # self.positional_embedding = PositionalEmbedding(args)
        # self.body = nn.GRU(input_size=args.hidden_units, hidden_size=args.hidden_units, batch_first=True)
        self.args = args
        self.num_horizon = args.hidden_units
        self.num_vertical = 8
        self.hidden_units = args.hidden_units
        self.hidden_size = args.hidden_units
        self.emb_size = args.hidden_units
        self.attention_size = args.hidden_units
        # self.conv_h = nn.ModuleList(
        #     [nn.Conv2d(in_channels=1, out_channels=self.num_horizon, kernel_size=(2*i+1, self.hidden_units), padding=(i,0)) for i in range(0, args.num_heads)])
        # self.conv_v = nn.Conv2d(in_channels=1, out_channels=self.num_vertical, kernel_size=(args.max_len, 1))

        # self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.encoder_g = nn.GRU(input_size=self.emb_size, hidden_size=self.hidden_size, batch_first=True)
        self.encoder_l = nn.GRU(input_size=self.emb_size, hidden_size=self.hidden_size, batch_first=True)
        self.A1 = nn.Linear(self.hidden_size, self.attention_size, bias=False)
        self.A2 = nn.Linear(self.hidden_size, self.attention_size, bias=False)
        self.attention_out = nn.Linear(self.attention_size, 1, bias=False)
        self.out = nn.Linear(2 * self.hidden_size, self.emb_size, bias=False)


        # self.body = ExactSasBody(args)
        self.ln = nn.LayerNorm(args.hidden_units)
        self.dropout = nn.Dropout(p=args.dropout)
        self.fc = nn.Linear(args.hidden_units*(args.num_heads + self.num_vertical), args.hidden_units)
        self.init_weights()

        self.sigmoid = nn.Sigmoid()
        self.vocab_size = args.num_items + 1

    @classmethod
    def code(cls):
        return 'narm'

    def forward(self, d):
        logits, info = self.get_logits(d)
        ret = {'logits':logits, 'info':info}
        if self.training:
            labels = d['labels']
            negative_labels = d['negative_labels']
            loss, loss_cnt = self.get_loss(logits, labels, negative_labels)
            ret['loss'] = loss
            ret['loss_cnt'] = loss_cnt
        else:
            # get scores (B x C) for validation
            x = d['tokens']
            lengths = (x > 0).sum(-1) -1 #(bs) 
            lengths = lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.args.hidden_units) #(bs, 1, dim)
            # pdb.set_trace()
            last_logits = torch.gather(logits, 1, lengths)  # B x 1 x H
            
            candidate_embeddings = self.token_embedding.emb(d['candidates'])  # B x C x H
            scores = (last_logits * candidate_embeddings).sum(-1)  # B x C
            ret['scores'] = scores
        return ret

    def get_logits(self, d):
        x = d['tokens'] #(bs, sl)
        attn_mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        attn_mask.tril_()  # causal attention for sasrec
        # e = self.token_embedding(d) + self.positional_embedding(d)
        e = self.token_embedding(d)
        e = self.dropout(e)
        info = None
        if self.output_info:
            info = {}
        # b = self.body(e, attn_mask, info)  # B x T x H
        # b, hidden = self.body(e, None)  # B x T x H
        e_size = e.size()

        # his_vectors = e.unsqueeze(1)
        his_vectors = e
        # Convolution Layers
        lengths = (x > 0).sum(-1) #(bs)
        sort_his_lengths, sort_idx = torch.topk(lengths, k=len(lengths))
        sort_his_length = sort_his_lengths.cpu().detach().numpy().tolist()
        
        sort_his_vectors = his_vectors.index_select(dim=0, index=sort_idx)
        history_packed = torch.nn.utils.rnn.pack_padded_sequence(sort_his_vectors, sort_his_length, batch_first=True)
        output_g, hidden_g = self.encoder_g(history_packed, None)
        output_l, hidden_l = self.encoder_l(history_packed, None)
        output_l, _ = torch.nn.utils.rnn.pad_packed_sequence(output_l, batch_first=True, total_length=self.args.max_len)
        output_g, _ = torch.nn.utils.rnn.pad_packed_sequence(output_g, batch_first=True, total_length=self.args.max_len)
        unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]
        output_l = output_l.index_select(dim=0, index=unsort_idx)  # [batch_size, history_max, emb_size]
        output_g = output_g.index_select(dim=0, index=unsort_idx)  # [batch_size, history_max, emb_size]
        # hidden_g = hidden_g[-1].index_select(dim=0, index=unsort_idx)  # [batch_size, emb_size]

        # pdb.set_trace()
        output_l_sqe = output_l.unsqueeze(1).expand(-1, self.args.max_len, -1, -1) #(bs, sl, sl, dim)
        output_g_sqe = output_g.unsqueeze(2) #(bs, sl, 1, dim)


        # output_l_sqe 
        # Attention Layer
        # pdb.set_trace()
        attention_g = self.A1(output_g_sqe)
        attention_l = self.A2(output_l_sqe)
        attention_value = self.attention_out((attention_g + attention_l).sigmoid()) #(bs, sl, sl, dim)

        x_triu_mask_1 = x.unsqueeze(-1).expand(-1, -1, self.args.max_len)
        x_triu_mask_2 = x.unsqueeze(1).expand(-1, self.args.max_len, -1)
        mask_1 = (x_triu_mask_1 > 0) #(bs, sl, sl)
        mask_2 = (x_triu_mask_2 > 0) #(bs, sl, sl)
        mask = (mask_1 * mask_2) #(bs, sl, sl, 1)
        # mask = torch.tril(mask, 0)
        mask.tril_()
        # pdb.set_trace() #mask[0].squeeze(-1)
        attention_value = attention_value.masked_fill(mask.unsqueeze(-1) == 0, 0)
        c_l = (attention_value * output_l_sqe).sum(-2) #(bs, sl, dim)

        # Prediction Layer
        b = self.out(torch.cat((output_g, c_l), dim=-1)) #(bs, sl, dim)

        b = self.ln(b)  # original code does this at the end of body
        return b, info
        # h = self.bert_head(b)  # B x T x V
        # h = self.bert_head(b, self.bert_embedding.token)  # B x T x V
        # return h, info

    def get_loss(self, logits, labels, negative_labels):
        _logits = logits.view(-1, logits.size(-1))  # BT x H
        _labels = labels.view(-1)  # BT
        _negative_labels = negative_labels.view(-1)  # BT

        valid = _labels > 0
        loss_cnt = valid.sum()  # = M
        valid_index = valid.nonzero().squeeze()  # M

        valid_logits = _logits[valid_index]  # M x H
        valid_labels = _labels[valid_index]  # M
        valid_negative_labels = _negative_labels[valid_index]  # M

        valid_labels_emb = self.token_embedding.emb(valid_labels)  # M x H
        valid_negative_labels_emb = self.token_embedding.emb(valid_negative_labels)  # M x H

        valid_labels_prob = self.sigmoid((valid_logits * valid_labels_emb).sum(-1))  # M
        valid_negative_labels_prob = self.sigmoid((valid_logits * valid_negative_labels_emb).sum(-1))  # M

        loss = -torch.log(valid_labels_prob + 1e-24) - torch.log((1-valid_negative_labels_prob) + 1e-24)
        loss = loss.mean()
        loss, loss_cnt = loss.unsqueeze(0), loss_cnt.unsqueeze(0)
        return loss, loss_cnt
