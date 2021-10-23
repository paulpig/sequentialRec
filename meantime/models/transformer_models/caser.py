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

        
        self.conv_h = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.num_horizon, kernel_size=(2*i+1, self.hidden_units), padding=(i,0)) for i in range(0, args.num_heads)])
        self.conv_v = nn.Conv2d(in_channels=1, out_channels=self.num_vertical, kernel_size=(args.max_len, 1))

        # self.body = ExactSasBody(args)
        self.ln = nn.LayerNorm(args.hidden_units)
        self.dropout = nn.Dropout(p=args.dropout)
        self.fc = nn.Linear(args.hidden_units*(args.num_heads + self.num_vertical), args.hidden_units)
        self.init_weights()

        self.sigmoid = nn.Sigmoid()
        self.vocab_size = args.num_items + 1

    @classmethod
    def code(cls):
        return 'caser'

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
            last_logits = logits[:, -1, :].unsqueeze(1)  # B x 1 x H
            candidate_embeddings = self.token_embedding.emb(d['candidates'])  # B x C x H
            scores = (last_logits * candidate_embeddings).sum(-1)  # B x C
            ret['scores'] = scores
        return ret

    def get_logits(self, d):
        x = d['tokens']
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

        his_vectors = e.unsqueeze(1)
        # Convolution Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.num_vertical > 0:
            out_v = self.conv_v(his_vectors) #(bs, 8, 1, dim)
            out_v = out_v.view(e_size[0], self.num_vertical * self.hidden_units)  # prepare for fully connect
            out_v = out_v.unsqueeze(1).expand(-1, self.args.max_len, -1) #(bs, sl, num_v*hidden)
        # horizontal conv layer
        out_hs = list()
        if self.num_horizon > 0:
            for conv in self.conv_h:
                conv_out = conv(his_vectors).squeeze(3).relu() #(bs, 16, sl)
                # pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                # pdb.set_trace()
                out_hs.append(conv_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect #(bs, 16*num_h, sl)
        out_h = out_h.permute(0, 2, 1) #(bs, sl, 16*dim)

        # pdb.set_trace()
        b = self.fc(torch.cat([out_v, out_h], -1)).relu() #(bs, sl, dim)

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
