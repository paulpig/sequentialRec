from meantime.dataloaders import graph
from ..base import BaseModel
from .embeddings import *
# from .bodies import ExactSasBody
from .bodies import SasBody
from .heads import *


import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
     
        
    def loglikeli(self, x_samples, y_samples):
#         print(x_samples.size())
#         print(y_samples.size())
        mu, logvar = self.get_mu_logvar(x_samples)

        return (-(mu - y_samples)**2 /2./logvar.exp()).sum(dim=1).mean(dim=0)
    

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        #random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

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

        
        self.mi_Discs = CLUBSample(args.hidden_units, args.hidden_units, args.hidden_units)

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
        return 'graph_sasrec_improve_lightgcn_kgat_add_regular'



    def loss_dependence_hisc(self, zdata_trn, ncaps, nhidden):
        loss_dep = torch.zeros(1).cuda()
        hH = (-1/nhidden)*torch.ones(nhidden, nhidden).cuda() + torch.eye(nhidden).cuda()
        kfactor = torch.zeros(ncaps, nhidden, nhidden).cuda()

        for mm in range(ncaps):
            data_temp = zdata_trn[:, mm * nhidden:(mm + 1) * nhidden]
            kfactor[mm, :, :] = torch.mm(data_temp.t(), data_temp)

        for mm in range(ncaps):
            for mn in range(mm + 1, ncaps):
                mat1 = torch.mm(hH, kfactor[mm, :, :])
                mat2 = torch.mm(hH, kfactor[mn, :, :])
                mat3 = torch.mm(mat1, mat2)
                teststat = torch.trace(mat3) / zdata_trn.size(0)
                # pdb.set_trace()
                loss_dep = loss_dep + teststat
        return loss_dep

    def DistanceCorrelation(self):
        """
            tensor_1: (bs*sl, dim)
            tensor_2: (bs*sl, dim)
        """
        batch_size = self.item_rep_graph_buy.size(0)
        feature_dim = self.item_rep_graph_buy.size(1)
        candidate_embeddings_graph_1 = self.item_rep_graph_buy.reshape(batch_size, feature_dim)
        candidate_embeddings_graph_2 = self.user_rep_graph_buy.reshape(batch_size, feature_dim)
        # pdb.set_trace()
        candidate_embeddings_gate = torch.sigmoid(self.W_graph_para_1(candidate_embeddings_graph_1) + self.W_graph_para_2(candidate_embeddings_graph_2))
        # gate = torch.sigmoid(self.W1_para(candidate_embeddings_bert) + self.W2_para(candidate_embeddings_graph))
        candidate_embeddings_buy = candidate_embeddings_gate * candidate_embeddings_graph_1 + (1. - candidate_embeddings_gate) * candidate_embeddings_graph_2
        candidate_embeddings_graph_attribute = self.user_rep_graph_attribute.reshape(batch_size, feature_dim)

        # loss = self.mi_Discs.learning_loss(candidate_embeddings_buy, candidate_embeddings_graph_attribute)

        loss = self.loss_dependence_hisc(torch.cat([candidate_embeddings_buy, candidate_embeddings_graph_attribute], dim=-1), 2, self.args.hidden_units)
        return loss
        # correlation;
        # tensor_1 = candidate_embeddings_buy
        # tensor_2 = candidate_embeddings_graph_attribute
        # # tensor_1, tensor_2: [channel]
        # # ref: https://en.wikipedia.org/wiki/Distance_correlation
        # channel = tensor_1.shape[0]
        # zeros = torch.zeros(channel, channel).to(tensor_1.device)
        # zero = torch.zeros(1).to(tensor_1.device)
        # # tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
        # """cul distance matrix"""
        # a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
        #         torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
        # tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
        # a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
        #         torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
        # """cul distance correlation"""
        # A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
        # B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
        # dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
        # dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
        # dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
        # return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

    def createMergeParameter(self):
        # hidden_W = self.args.hidden_units // 2
        if self.args.merge_type == "concat":
            hidden_W = self.args.hidden_units //2
        else:
            hidden_W = self.args.hidden_units
        
        self.W1_para_1 = nn.Linear(hidden_W, 1).cuda()
        self.W1_para_2 = nn.Linear(hidden_W, 1).cuda()
        self.W1_para_3 = nn.Linear(hidden_W, 1).cuda()

        self.W_graph_para_1 = nn.Linear(hidden_W, hidden_W).cuda()
        self.W_graph_para_2 = nn.Linear(hidden_W, hidden_W).cuda()

        self.W_graph_para_1_1 = nn.Linear(hidden_W, hidden_W).cuda()
        self.W_graph_para_2_1 = nn.Linear(hidden_W, hidden_W).cuda()
        self.W_graph_para_3_1 = nn.Linear(hidden_W, hidden_W).cuda()
        # W_graph_para_3_1
        return

    
    def setUserItemRepFromGraph(self, user_rep_buy, item_rep_buy, user_rep_view, item_rep_view):
        """
        The representations from the LightGCN model; 
        user_rep_graph: (|users|, dim)
        item_rep_graph: (|items|, dim)
        """
        self.user_rep_graph_buy = user_rep_buy
        self.item_rep_graph_buy = item_rep_buy

        self.user_rep_graph_attribute = user_rep_view
        self.item_rep_graph_attribute = item_rep_view

        # self.user_rep_graph_buy = F.normalize(self.user_rep_graph_buy, p=2, dim=1)
        # self.item_rep_graph_buy = F.normalize(self.item_rep_graph_buy, p=2, dim=1)
        # self.user_rep_graph_attribute = F.normalize(self.user_rep_graph_attribute, p=2, dim=1)
        # self.item_rep_graph_attribute = F.normalize(self.item_rep_graph_attribute, p=2, dim=1)
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
            candidate_embeddings_graph_attribute = self.user_rep_graph_attribute[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1)

            if self.args.merge_type == "gate":
                # candidate_embeddings_graph_1 = self.item_rep_graph_view[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1)
                candidate_embeddings_gate = torch.sigmoid(self.W_graph_para_1_1(candidate_embeddings_buy) + self.W_graph_para_2_1(candidate_embeddings_graph_attribute))
                # gate = torch.sigmoid(self.W1_para(candidate_embeddings_bert) + self.W2_para(candidate_embeddings_graph))
                # candidate_embeddings_review = candidate_embeddings_gate * candidate_embeddings_graph_1 + (1. - candidate_embeddings_gate) * candidate_embeddings_graph_2
                # candidate_embeddings = candidate_embeddings_buy + candidate_embeddings_review
                candidate_embeddings = candidate_embeddings_buy * candidate_embeddings_gate + (1. - candidate_embeddings_gate) * candidate_embeddings_graph_attribute
            elif self.args.merge_type == "rm_kgat":
                candidate_embeddings = candidate_embeddings_buy
            elif self.args.merge_type == "rm_lightgcn":
                candidate_embeddings = candidate_embeddings_graph_attribute
            elif self.args.merge_type == "concat":
                candidate_embeddings = torch.cat((candidate_embeddings_buy, candidate_embeddings_graph_attribute), dim=-1)
            elif self.args.merge_type == "attention":
                seq_1 = self.W1_para_1(torch.tanh(self.W_graph_para_1_1(candidate_embeddings_graph_1)))#(bs, sl, 1)
                seq_2 = self.W1_para_2(torch.tanh(self.W_graph_para_2_1(candidate_embeddings_graph_2))) #(bs, sl, 1)
                seq_3 = self.W1_para_3(torch.tanh(self.W_graph_para_3_1(candidate_embeddings_graph_attribute))) #(bs, sl, 1)
                seq_merge = torch.cat([seq_1, seq_2, seq_3], dim=-1)
                seq_merge_weight = nn.functional.softmax(seq_merge, dim=-1) #(bs, sl, 3)
                seq_merge_tensor = torch.cat([torch.unsqueeze(candidate_embeddings_graph_1, dim=-2), torch.unsqueeze(candidate_embeddings_graph_2, dim=-2), torch.unsqueeze(candidate_embeddings_graph_attribute, dim=-2)], dim=-2) #(bs, sl, 3, dim)
                candidate_embeddings = (seq_merge_tensor * torch.unsqueeze(seq_merge_weight, dim=-1)).sum(-2) #(bs, sl, dim)
            elif self.args.merge_type == "add":
                candidate_embeddings = candidate_embeddings_graph_1 + candidate_embeddings_graph_2 + candidate_embeddings_graph_attribute
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
        candidate_embeddings_graph_2 = self.user_rep_graph_buy[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1)
        candidate_embeddings_gate = torch.sigmoid(self.W_graph_para_1(candidate_embeddings_graph_1) + self.W_graph_para_2(candidate_embeddings_graph_2))
        # gate = torch.sigmoid(self.W1_para(candidate_embeddings_bert) + self.W2_para(candidate_embeddings_graph))
        graph_e_buy = candidate_embeddings_gate * candidate_embeddings_graph_1 + (1. - candidate_embeddings_gate) * candidate_embeddings_graph_2
        candidate_embeddings_graph_attribute = self.user_rep_graph_attribute[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1)
        
        if self.args.merge_type == "gate":
            # candidate_embeddings_graph_1 = self.item_rep_graph_view[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1)
            candidate_embeddings_gate = torch.sigmoid(self.W_graph_para_1_1(graph_e_buy) + self.W_graph_para_2_1(candidate_embeddings_graph_attribute))
            graph_e = graph_e_buy * candidate_embeddings_gate + (1. - candidate_embeddings_gate) * candidate_embeddings_graph_attribute
        elif self.args.merge_type == "rm_kgat":
            graph_e = graph_e_buy
        elif self.args.merge_type == "rm_lightgcn":
            graph_e = candidate_embeddings_graph_attribute
        elif self.args.merge_type == "concat":
            graph_e = torch.cat((graph_e_buy, candidate_embeddings_graph_attribute), dim=-1)
        elif self.args.merge_type == "attention":
            seq_1 = self.W1_para_1(torch.tanh(self.W_graph_para_1_1(candidate_embeddings_graph_1)))#(bs, sl, 1)
            seq_2 = self.W1_para_2(torch.tanh(self.W_graph_para_2_1(candidate_embeddings_graph_2))) #(bs, sl, 1)
            seq_3 = self.W1_para_3(torch.tanh(self.W_graph_para_3_1(candidate_embeddings_graph_attribute))) #(bs, sl, 1)
            seq_merge = torch.cat([seq_1, seq_2, seq_3], dim=-1)
            seq_merge_weight = nn.functional.softmax(seq_merge, dim=-1) #(bs, sl, 3)
            seq_merge_tensor = torch.cat([torch.unsqueeze(candidate_embeddings_graph_1, dim=-2), torch.unsqueeze(candidate_embeddings_graph_2, dim=-2), torch.unsqueeze(candidate_embeddings_graph_attribute, dim=-2)], dim=-2) #(bs, sl, 3, dim)
            # pdb.set_trace()
            graph_e = (seq_merge_tensor * torch.unsqueeze(seq_merge_weight, dim=-1)).sum(-2) #(bs, sl, dim)
        elif self.args.merge_type == "add":
            graph_e = candidate_embeddings_graph_1 + candidate_embeddings_graph_2 + candidate_embeddings_graph_attribute
        # candidate_embeddings_graph_2 = self.user_rep_graph_view[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1)
        # candidate_embeddings_gate = torch.sigmoid(self.W_graph_para_1_1(graph_e_buy) + self.W_graph_para_2_1(candidate_embeddings_graph_attribute))
        # gate = torch.sigmoid(self.W1_para(candidate_embeddings_bert) + self.W2_para(candidate_embeddings_graph))
        # graph_e = candidate_embeddings_gate * graph_e_buy + (1. - candidate_embeddings_gate) * candidate_embeddings_graph_attribute
        
        # graph_e = torch.cat((graph_e_buy, candidate_embeddings_graph_attribute), dim=-1) #(bs, sl, 2*dim)
        # graph_e = graph_e_buy + graph_e_review
        # graph_e = graph_e_buy

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
        valid_labels_emb_graph_2 = self.user_rep_graph_buy[valid_labels, :]  # M x H
        valid_labels_gate = torch.sigmoid(self.W_graph_para_1(valid_labels_emb_graph_1) + self.W_graph_para_2(valid_labels_emb_graph_2))
        valid_labels_emb_buy = valid_labels_gate * valid_labels_emb_graph_1 + (1. - valid_labels_gate) * valid_labels_emb_graph_2
        valid_labels_emb_graph_attribute = self.user_rep_graph_attribute[valid_labels, :]  # M x H

        if self.args.merge_type == "gate":
            candidate_embeddings_gate = torch.sigmoid(self.W_graph_para_1_1(valid_labels_emb_buy) + self.W_graph_para_2_1(valid_labels_emb_graph_attribute))
            valid_labels_emb = valid_labels_emb_buy * candidate_embeddings_gate + (1. - candidate_embeddings_gate) * valid_labels_emb_graph_attribute
        elif self.args.merge_type == "rm_kgat":
            valid_labels_emb = valid_labels_emb_buy
        elif self.args.merge_type == "rm_lightgcn":
            valid_labels_emb = valid_labels_emb_graph_attribute
        elif self.args.merge_type == "concat":
            valid_labels_emb = torch.cat((valid_labels_emb_buy, valid_labels_emb_graph_attribute), dim=-1)
        elif self.args.merge_type == "attention":
            seq_1 = self.W1_para_1(torch.tanh(self.W_graph_para_1_1(valid_labels_emb_graph_1)))#(bs, sl, 1)
            seq_2 = self.W1_para_2(torch.tanh(self.W_graph_para_2_1(valid_labels_emb_graph_2))) #(bs, sl, 1)
            seq_3 = self.W1_para_3(torch.tanh(self.W_graph_para_3_1(valid_labels_emb_graph_attribute))) #(bs, sl, 1)
            seq_merge = torch.cat([seq_1, seq_2, seq_3], dim=-1)
            seq_merge_weight = nn.functional.softmax(seq_merge, dim=-1) #(bs, sl, 3)
            seq_merge_tensor = torch.cat([torch.unsqueeze(valid_labels_emb_graph_1, dim=-2), torch.unsqueeze(valid_labels_emb_graph_2, dim=-2), torch.unsqueeze(valid_labels_emb_graph_attribute, dim=-2)], dim=-2) #(bs, sl, 3, dim)
            valid_labels_emb = (seq_merge_tensor * torch.unsqueeze(seq_merge_weight, dim=-1)).sum(-2) #(bs, sl, dim)
        elif self.args.merge_type == "add":
            valid_labels_emb = valid_labels_emb_graph_1 + valid_labels_emb_graph_2 + valid_labels_emb_graph_attribute
        # valid_labels_emb_graph_2 = self.item_rep_graph_view[valid_labels, :]  # M x H
        # valid_labels_gate = torch.sigmoid(self.W_graph_para_1_1(valid_labels_emb_buy) + self.W_graph_para_2_1(valid_labels_emb_graph_attribute))
        # valid_labels_emb = valid_labels_gate * valid_labels_emb_buy + (1. - valid_labels_gate) * valid_labels_emb_graph_attribute
        # valid_labels_emb = torch.cat((valid_labels_emb_buy, valid_labels_emb_graph_attribute), dim=-1)

        # valid_labels_emb = valid_labels_emb_buy + valid_labels_emb_review
        # valid_labels_emb = valid_labels_emb_buy
        
        valid_negative_labels_emb_graph_1 = self.item_rep_graph_buy[valid_negative_labels, :]  # M x H
        valid_negative_labels_emb_graph_2 = self.user_rep_graph_buy[valid_negative_labels, :]  # M x H
        valid_negative_labels_gate = torch.sigmoid(self.W_graph_para_1(valid_negative_labels_emb_graph_1) + self.W_graph_para_2(valid_negative_labels_emb_graph_2))
        valid_negative_labels_emb_buy = valid_negative_labels_gate * valid_negative_labels_emb_graph_1 + (1. - valid_negative_labels_gate) * valid_negative_labels_emb_graph_2
        # valid_negative_labels_emb = self.item_rep_graph[valid_negative_labels, :]  # M x H
        
        valid_negative_labels_emb_graph_attribute = self.user_rep_graph_attribute[valid_negative_labels, :]  # M x H
        
        if self.args.merge_type == "gate":
            candidate_embeddings_gate = torch.sigmoid(self.W_graph_para_1_1(valid_negative_labels_emb_buy) + self.W_graph_para_2_1(valid_negative_labels_emb_graph_attribute))
            valid_negative_labels_emb = valid_negative_labels_emb_buy * candidate_embeddings_gate + (1. - candidate_embeddings_gate) * valid_negative_labels_emb_graph_attribute
        elif self.args.merge_type == "rm_kgat":
            valid_negative_labels_emb = valid_negative_labels_emb_buy
        elif self.args.merge_type == "rm_lightgcn":
            valid_negative_labels_emb = valid_negative_labels_emb_graph_attribute
        elif self.args.merge_type == "concat":
            valid_negative_labels_emb = torch.cat((valid_negative_labels_emb_buy, valid_negative_labels_emb_graph_attribute), dim=-1)
        elif self.args.merge_type == "attention":
            seq_1 = self.W1_para_1(torch.tanh(self.W_graph_para_1_1(valid_negative_labels_emb_graph_1)))#(bs, sl, 1)
            seq_2 = self.W1_para_2(torch.tanh(self.W_graph_para_2_1(valid_negative_labels_emb_graph_2))) #(bs, sl, 1)
            seq_3 = self.W1_para_3(torch.tanh(self.W_graph_para_3_1(valid_negative_labels_emb_graph_attribute))) #(bs, sl, 1)
            seq_merge = torch.cat([seq_1, seq_2, seq_3], dim=-1)
            seq_merge_weight = nn.functional.softmax(seq_merge, dim=-1) #(bs, sl, 3)
            seq_merge_tensor = torch.cat([torch.unsqueeze(valid_negative_labels_emb_graph_1, dim=-2), torch.unsqueeze(valid_negative_labels_emb_graph_2, dim=-2), torch.unsqueeze(valid_negative_labels_emb_graph_attribute, dim=-2)], dim=-2) #(bs, sl, 3, dim)
            valid_negative_labels_emb = (seq_merge_tensor * torch.unsqueeze(seq_merge_weight, dim=-1)).sum(-2) #(bs, sl, dim)
        elif self.args.merge_type == "add":
            valid_negative_labels_emb = valid_negative_labels_emb_graph_1 + valid_negative_labels_emb_graph_2 + valid_negative_labels_emb_graph_attribute
        # valid_negative_labels_emb_graph_2 = self.item_rep_graph_view[valid_negative_labels, :]  # M x H
        # valid_negative_labels_gate = torch.sigmoid(self.W_graph_para_1_1(valid_negative_labels_emb_buy) + self.W_graph_para_2_1(valid_negative_labels_emb_graph_attribute))
        # valid_negative_labels_emb = valid_negative_labels_gate * valid_negative_labels_emb_buy + (1. - valid_negative_labels_gate) * valid_negative_labels_emb_graph_attribute
        # valid_negative_labels_emb = torch.cat((valid_negative_labels_emb_buy, valid_negative_labels_emb_graph_attribute), dim=-1)

        # valid_negative_labels_emb = valid_negative_labels_emb_buy + valid_negative_labels_emb_review
        # valid_negative_labels_emb = valid_negative_labels_emb_buy

        valid_labels_prob = self.sigmoid((valid_logits * valid_labels_emb).sum(-1))  # M
        valid_negative_labels_prob = self.sigmoid((valid_logits * valid_negative_labels_emb).sum(-1))  # M

        loss = -torch.log(valid_labels_prob + 1e-24) - torch.log((1-valid_negative_labels_prob) + 1e-24)
        loss = loss.mean()
        loss, loss_cnt = loss.unsqueeze(0), loss_cnt.unsqueeze(0)

        #add distance correlation
        distance_correl_loss = self.DistanceCorrelation()
        # pdb.set_trace()
        loss += 0.01 * distance_correl_loss
        return loss, loss_cnt