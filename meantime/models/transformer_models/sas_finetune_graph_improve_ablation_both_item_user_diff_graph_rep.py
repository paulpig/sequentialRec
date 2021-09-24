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
        #创建gate所需函数;
        self.createMergeParameter()
        self.init_weights()

        self.sigmoid = nn.Sigmoid()
        self.vocab_size = args.num_items + 1

    @classmethod
    def code(cls):
        return 'sas_finetune_graph_improve_ablation_both_item_user_diff_graph_rep'



    def createMergeParameter(self):
        self.W1_para = nn.Linear(self.args.hidden_units, 1).cuda()
        self.W2_para = nn.Linear(self.args.hidden_units, 1).cuda()

        self.W_graph_para_1 = nn.Linear(self.args.hidden_units, self.args.hidden_units).cuda()
        self.W_graph_para_2 = nn.Linear(self.args.hidden_units, self.args.hidden_units).cuda()

        self.W_graph_para_1_1 = nn.Linear(self.args.hidden_units, self.args.hidden_units).cuda()
        self.W_graph_para_2_1 = nn.Linear(self.args.hidden_units, self.args.hidden_units).cuda()
        return

    
    def setUserItemRepFromGraph(self, user_rep_buy, item_rep_buy, user_rep_view, item_rep_view):
        """
        The representations from the LightGCN model; 
        user_rep_graph: (|users|, dim)
        item_rep_graph: (|items|, dim)
        """
        self.user_rep_graph_buy = user_rep_buy
        self.item_rep_graph_buy = item_rep_buy

        self.user_rep_graph_view = user_rep_view
        self.item_rep_graph_view = item_rep_view
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
            loss, loss_cnt = self.get_loss(logits, labels, negative_labels)
            # loss, loss_cnt = self.get_loss_as_bert(logits, labels, negative_labels, d)
            ret['loss'] = loss
            ret['loss_cnt'] = loss_cnt
        else:
            # get scores (B x C) for validation
            last_logits = logits[:, -1, :].unsqueeze(1)  # B x 1 x H
            # candidate_embeddings = self.token_embedding.emb(d['candidates'])  # B x C x H
            x = d['candidates'] #(bs, C)
            x_unsqueeenze = x.reshape(-1) #(bs*C)
            candidate_embeddings_graph_1 = self.item_rep_graph_buy[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1)
            candidate_embeddings_graph_2 = self.user_rep_graph_buy[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1)
            candidate_embeddings_gate = torch.sigmoid(self.W_graph_para_1(candidate_embeddings_graph_1) + self.W_graph_para_2(candidate_embeddings_graph_2))
            # gate = torch.sigmoid(self.W1_para(candidate_embeddings_bert) + self.W2_para(candidate_embeddings_graph))
            candidate_embeddings_buy = candidate_embeddings_gate * candidate_embeddings_graph_1 + (1. - candidate_embeddings_gate) * candidate_embeddings_graph_2
            
            candidate_embeddings_graph_1 = self.item_rep_graph_view[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1)
            candidate_embeddings_graph_2 = self.user_rep_graph_view[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1)
            candidate_embeddings_gate = torch.sigmoid(self.W_graph_para_1_1(candidate_embeddings_graph_1) + self.W_graph_para_2_1(candidate_embeddings_graph_2))
            # gate = torch.sigmoid(self.W1_para(candidate_embeddings_bert) + self.W2_para(candidate_embeddings_graph))
            candidate_embeddings_review = candidate_embeddings_gate * candidate_embeddings_graph_1 + (1. - candidate_embeddings_gate) * candidate_embeddings_graph_2

            # candidate_embeddings = candidate_embeddings_buy + candidate_embeddings_review
            candidate_embeddings = candidate_embeddings_buy

            scores = (last_logits * candidate_embeddings).sum(-1)  # B x C
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
        # graph_e = self.item_rep_graph[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1) #(bs, sl, dim) #use the hidden representation to init the embedding lookup table;
        candidate_embeddings_graph_1 = self.item_rep_graph_buy[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1)
        candidate_embeddings_graph_2 = self.item_rep_graph_buy[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1)
        candidate_embeddings_gate = torch.sigmoid(self.W_graph_para_1(candidate_embeddings_graph_1) + self.W_graph_para_2(candidate_embeddings_graph_2))
        # gate = torch.sigmoid(self.W1_para(candidate_embeddings_bert) + self.W2_para(candidate_embeddings_graph))
        graph_e_buy = candidate_embeddings_gate * candidate_embeddings_graph_1 + (1. - candidate_embeddings_gate) * candidate_embeddings_graph_2


        candidate_embeddings_graph_1 = self.item_rep_graph_view[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1)
        candidate_embeddings_graph_2 = self.user_rep_graph_view[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1)
        candidate_embeddings_gate = torch.sigmoid(self.W_graph_para_1_1(candidate_embeddings_graph_1) + self.W_graph_para_2_1(candidate_embeddings_graph_2))
        # gate = torch.sigmoid(self.W1_para(candidate_embeddings_bert) + self.W2_para(candidate_embeddings_graph))
        graph_e_review = candidate_embeddings_gate * candidate_embeddings_graph_1 + (1. - candidate_embeddings_gate) * candidate_embeddings_graph_2

        # graph_e = graph_e_buy + graph_e_review
        graph_e = graph_e_buy

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

    def get_loss(self, logits, labels, negative_labels):
        _logits = logits.reshape(-1, logits.size(-1))  # BT x H
        _labels = labels.reshape(-1)  # BT
        _negative_labels = negative_labels.reshape(-1)  # BT

        valid = _labels > 0
        loss_cnt = valid.sum()  # = M
        valid_index = valid.nonzero().squeeze()  # M

        valid_logits = _logits[valid_index]  # M x H
        valid_labels = _labels[valid_index]  # M
        valid_negative_labels = _negative_labels[valid_index]  # M

        # valid_labels_emb = self.token_embedding.emb(valid_labels)  # M x H
        # valid_negative_labels_emb = self.token_embedding.emb(valid_negative_labels)  # M x H

        valid_labels_emb_graph_1 = self.item_rep_graph_buy[valid_labels, :]  # M x H
        valid_labels_emb_graph_2 = self.item_rep_graph_buy[valid_labels, :]  # M x H
        valid_labels_gate = torch.sigmoid(self.W_graph_para_1(valid_labels_emb_graph_1) + self.W_graph_para_2(valid_labels_emb_graph_2))
        valid_labels_emb_buy = valid_labels_gate * valid_labels_emb_graph_1 + (1. - valid_labels_gate) * valid_labels_emb_graph_2


        valid_labels_emb_graph_1 = self.item_rep_graph_view[valid_labels, :]  # M x H
        valid_labels_emb_graph_2 = self.item_rep_graph_view[valid_labels, :]  # M x H
        valid_labels_gate = torch.sigmoid(self.W_graph_para_1_1(valid_labels_emb_graph_1) + self.W_graph_para_2_1(valid_labels_emb_graph_2))
        valid_labels_emb_review = valid_labels_gate * valid_labels_emb_graph_1 + (1. - valid_labels_gate) * valid_labels_emb_graph_2

        # valid_labels_emb = valid_labels_emb_buy + valid_labels_emb_review
        valid_labels_emb = valid_labels_emb_buy
        
        valid_negative_labels_emb_graph_1 = self.item_rep_graph_buy[valid_negative_labels, :]  # M x H
        valid_negative_labels_emb_graph_2 = self.item_rep_graph_buy[valid_negative_labels, :]  # M x H
        valid_negative_labels_gate = torch.sigmoid(self.W_graph_para_1(valid_negative_labels_emb_graph_1) + self.W_graph_para_2(valid_negative_labels_emb_graph_2))
        valid_negative_labels_emb_buy = valid_negative_labels_gate * valid_negative_labels_emb_graph_1 + (1. - valid_negative_labels_gate) * valid_negative_labels_emb_graph_2
        # valid_negative_labels_emb = self.item_rep_graph[valid_negative_labels, :]  # M x H
        
        valid_negative_labels_emb_graph_1 = self.item_rep_graph_view[valid_negative_labels, :]  # M x H
        valid_negative_labels_emb_graph_2 = self.item_rep_graph_view[valid_negative_labels, :]  # M x H
        valid_negative_labels_gate = torch.sigmoid(self.W_graph_para_1_1(valid_negative_labels_emb_graph_1) + self.W_graph_para_2_1(valid_negative_labels_emb_graph_2))
        valid_negative_labels_emb_review = valid_negative_labels_gate * valid_negative_labels_emb_graph_1 + (1. - valid_negative_labels_gate) * valid_negative_labels_emb_graph_2

        # valid_negative_labels_emb = valid_negative_labels_emb_buy + valid_negative_labels_emb_review
        valid_negative_labels_emb = valid_negative_labels_emb_buy

        valid_labels_prob = self.sigmoid((valid_logits * valid_labels_emb).sum(-1))  # M
        valid_negative_labels_prob = self.sigmoid((valid_logits * valid_negative_labels_emb).sum(-1))  # M

        loss = -torch.log(valid_labels_prob + 1e-24) - torch.log((1-valid_negative_labels_prob) + 1e-24)
        loss = loss.mean()
        loss, loss_cnt = loss.unsqueeze(0), loss_cnt.unsqueeze(0)
        return loss, loss_cnt