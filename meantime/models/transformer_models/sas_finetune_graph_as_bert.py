from meantime.dataloaders import graph
from ..base import BaseModel
from .embeddings import *
# from .bodies import ExactSasBody
from .bodies import SasBody
from .heads import *

import torch
import torch.nn as nn
import pdb

class SASModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.output_info = args.output_info
        self.token_embedding = TokenEmbedding(args)
        self.positional_embedding = PositionalEmbedding(args)
        self.body = SasBody(args) #hidden_size = hidden_size * 2
        self.ln = nn.LayerNorm(args.hidden_units)
        self.dropout = nn.Dropout(p=args.dropout)
        self.ce = nn.CrossEntropyLoss()

        # self.gru = nn.GRU(self.args.hidden_units * 2, self.args.hidden_units, self.args.gru_layer_number, dropout=self.args.dropout, batch_first=True)
        # self.gru = nn.GRU(self.args.hidden_units, self.args.hidden_units, self.args.gru_layer_number, dropout=self.args.dropout, batch_first=True)
        
        # self.head是为加载bert模型参数, 在此处不使用;
        if args.headtype == 'dot':
            self.head = BertDotProductPredictionHead(args, self.token_embedding.emb)
        elif args.headtype == 'linear':
            self.head = BertLinearPredictionHead(args)
        else:
            raise ValueError
        # self.project_layer = nn.Sequential(
        #         nn.Linear(self.args.hidden_units, self.args.hidden_units),
        #         GELU(),
        #         nn.LayerNorm(self.args.hidden_units),
        #         nn.Linear(self.args.hidden_units, self.args.hidden_units)
        #     )
        self.init_weights()

        self.sigmoid = nn.Sigmoid()
        self.vocab_size = args.num_items + 1

    @classmethod
    def code(cls):
        return 'sas_finetune_graph_as_bert'


    def setUserItemRepFromGraph(self, user_rep_graph, item_rep_graph):
        """
        The representations from the LightGCN model; 
        user_rep_graph: (|users|, dim)
        item_rep_graph: (|items|, dim)
        """
        self.user_rep_graph = user_rep_graph
        self.item_rep_graph = item_rep_graph
        return
    

    def forward(self, d):
        """
        add the graph representation; 
        """
        logits, info = self.get_logits(d)

        # pdb.set_trace()
        ret = {'logits':logits, 'info':info}
        if self.training:
            labels = d['labels']
            negative_labels = d['negative_labels']
            # loss, loss_cnt = self.get_loss(logits, labels, negative_labels)
            loss, loss_cnt = self.get_loss_as_bert(logits, labels)
            ret['loss'] = loss
            ret['loss_cnt'] = loss_cnt
        else:
            # get scores (B x C) for validation
            # last_logits = logits[:, -1, :].unsqueeze(1)  # B x 1 x H
            last_logits = logits[:, -1, :]  # B x 1 x H
            # candidate_embeddings = self.token_embedding.emb(d['candidates'])  # B x C x H
            # scores = (last_logits * candidate_embeddings).sum(-1)  # B x C
            candidates = d['candidates']
            scores = self.head(last_logits, candidates)
            ret['scores'] = scores
        return ret

    def get_logits(self, d):
        """
        add the graph representation; 
        """
        x = d['tokens'] #(bs, sl)

        
        x_unsqueeenze = x.reshape(-1)
        # pdb.set_trace()

        attn_mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        attn_mask.tril_()  # causal attention for sasrec
        # e = self.token_embedding(d) + self.positional_embedding(d)

        #采用图模型输出的表征初始化序列推荐模型item lookup table, 初始化的效果增强;
        # graph_e = self.token_embedding(d)
        graph_e = self.item_rep_graph[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1) #(bs, sl, dim) #use the hidden representation to init the embedding lookup table;
        e = graph_e + self.positional_embedding(d)

        #concat the representation from the graph and the init item embedding;
        # e = torch.cat((graph_e, e), dim=-1) #(bs, sl, 2*dim)

        e = self.dropout(e)
        info = None
        if self.output_info:
            info = {}
        b = self.body(e, attn_mask, info)  # B x T x H
        b = self.ln(b)  # original code does this at the end of body

        #add graph representation
        # graph_e = self.item_rep_graph[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1) #(bs, sl, dim)

        # merge_e = torch.cat((b, graph_e), dim=-1)
        # merge_e = b

        #add GRU layer
        # h0 = torch.zeros(self.args.gru_layer_number, e.size(0), self.args.hidden_units).to(self.args.device)
        # output, hidden = self.gru(merge_e, h0)

        # b = output
        return b, info
        # h = self.bert_head(b)  # B x T x V
        # h = self.bert_head(b, self.bert_embedding.token)  # B x T x V
        # return h, info

    # def get_loss(self, logits, labels, negative_labels):
    #     _logits = logits.reshape(-1, logits.size(-1))  # BT x H
    #     _labels = labels.reshape(-1)  # BT
    #     _negative_labels = negative_labels.reshape(-1)  # BT

    #     valid = _labels > 0
    #     loss_cnt = valid.sum()  # = M
    #     valid_index = valid.nonzero().squeeze()  # M

    #     valid_logits = _logits[valid_index]  # M x H
    #     valid_labels = _labels[valid_index]  # M
    #     valid_negative_labels = _negative_labels[valid_index]  # M

    #     valid_labels_emb = self.token_embedding.emb(valid_labels)  # M x H
    #     valid_negative_labels_emb = self.token_embedding.emb(valid_negative_labels)  # M x H

    #     valid_labels_prob = self.sigmoid((valid_logits * valid_labels_emb).sum(-1))  # M
    #     valid_negative_labels_prob = self.sigmoid((valid_logits * valid_negative_labels_emb).sum(-1))  # M

    #     loss = -torch.log(valid_labels_prob + 1e-24) - torch.log((1-valid_negative_labels_prob) + 1e-24)
    #     loss = loss.mean()
    #     loss, loss_cnt = loss.unsqueeze(0), loss_cnt.unsqueeze(0)
    #     return loss, loss_cnt

    def get_loss_as_bert(self, logits, labels):
        _logits = logits.reshape(-1, logits.size(-1))  # BT x H
        _labels = labels.reshape(-1)  # BT

        valid = _labels > 0
        loss_cnt = valid.sum()  # = M
        valid_index = valid.nonzero().squeeze()  # M

        valid_logits = _logits[valid_index]  # M x H
        valid_labels = _labels[valid_index]  # M

        # valid_scores = self.get_scores(d, valid_logits)  # M x V, V是词表的数量;
        valid_scores = self.head(valid_logits)

        loss = self.ce(valid_scores, valid_labels)
        
        loss, loss_cnt = loss.unsqueeze(0), loss_cnt.unsqueeze(0)

        return loss, loss_cnt

    # def get_loss(self, d, logits, labels):
    #     """
    #     return:
    #         valid_scores: (M, V)
    #     """
    #     _logits = logits.view(-1, logits.size(-1))  # BT x H
    #     _labels = labels.view(-1)  # BT

    #     valid = _labels > 0
    #     loss_cnt = valid.sum()  # = M
    #     valid_index = valid.nonzero().squeeze()  # M: mask的数量;

    #     valid_logits = _logits[valid_index]  # M x H
    #     valid_scores = self.get_scores(d, valid_logits)  # M x V, V是词表的数量;
    #     valid_labels = _labels[valid_index]  # M

    #     loss = self.ce(valid_scores, valid_labels)
    #     loss, loss_cnt = loss.unsqueeze(0), loss_cnt.unsqueeze(0)
    #     return loss, loss_cnt, valid_scores