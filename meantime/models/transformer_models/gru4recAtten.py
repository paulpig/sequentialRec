from ..base import BaseModel
from .embeddings import *
from .bodies import ExactSasBody

import torch
import torch.nn as nn


class GRU4Rec(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.output_info = args.output_info
        self.token_embedding = TokenEmbedding(args)
        # self.positional_embedding = PositionalEmbedding(args)
        self.body = nn.GRU(input_size=args.hidden_units, hidden_size=args.hidden_units, batch_first=True)
        # self.body = ExactSasBody(args)
        self.ln = nn.LayerNorm(args.hidden_units)
        self.dropout = nn.Dropout(p=args.dropout)
        self.createMergeParameter()
        self.init_weights()

        self.sigmoid = nn.Sigmoid()
        self.vocab_size = args.num_items + 1

    @classmethod
    def code(cls):
        return 'gru4recGraph'


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
    
    def setUserItemRepFromGraph(self, user_rep_buy, item_rep_buy, user_rep_view, item_rep_view, user_ori, item_ori, k_user_ori, k_item_ori):
        """
        The representations from the LightGCN model; 
        user_rep_graph: (|users|, dim)
        item_rep_graph: (|items|, dim)
        """
        self.user_rep_graph_buy = user_rep_buy #out
        self.item_rep_graph_buy = item_rep_buy #in

        self.user_rep_graph_attribute = user_rep_view
        self.item_rep_graph_attribute = item_rep_view

        self.user_ori = user_ori
        self.item_ori = item_ori

        self.k_user_ori = k_user_ori
        self.k_item_ori = k_item_ori
        return
    
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
            # candidate_embeddings = self.token_embedding.emb(d['candidates'])  # B x C x H
            x = d['candidates'] #(bs, C)
            # x_cates = d['candidate_cates'].reshape(-1)
            # x_brands = d['candidate_brands'].reshape(-1)
            # x_prices = d['candidate_prices'].reshape(-1)

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
                # candidate_embeddings = candidate_embeddings_buy
                # seq_1 = self.W1_para_1(torch.tanh(self.W_graph_para_1_1(ori_embeddings_graph_1)))#(bs, sl, 1)
                seq_2 = self.W1_para_2(torch.tanh(self.W_graph_para_2_1(candidate_embeddings_graph_1))) #(bs, sl, 1)
                seq_3 = self.W1_para_3(torch.tanh(self.W_graph_para_3_1(candidate_embeddings_graph_2)))
                seq_merge = torch.cat([seq_2, seq_3], dim=-1)
                seq_merge_weight = nn.functional.softmax(seq_merge, dim=-1) #(bs, sl, 3)
                seq_merge_tensor = torch.cat([torch.unsqueeze(candidate_embeddings_graph_1, dim=-2), torch.unsqueeze(candidate_embeddings_graph_2, dim=-2)], dim=-2) #(bs, sl, 3, dim)
                # pdb.set_trace()
                candidate_embeddings = (seq_merge_tensor * torch.unsqueeze(seq_merge_weight, dim=-1)).sum(-2) #(bs, sl, dim)
            elif self.args.merge_type == "rm_lightgcn":
                # ori_embeddings_graph_2 = self.k_user_ori[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1)
                # # graph_e = candidate_embeddings_graph_attribute
                # seq_1 = self.W1_para_1(torch.tanh(self.W_graph_para_1_1(candidate_embeddings_graph_attribute)))#(bs, sl, 1)
                # seq_2 = self.W1_para_2(torch.tanh(self.W_graph_para_2_1(ori_embeddings_graph_2))) #
                # seq_merge = torch.cat([seq_1, seq_2], dim=-1)
                # seq_merge_weight = nn.functional.softmax(seq_merge, dim=-1) #(bs, sl, 3)
                # seq_merge_tensor = torch.cat([torch.unsqueeze(candidate_embeddings_graph_attribute, dim=-2), torch.unsqueeze(ori_embeddings_graph_2, dim=-2)], dim=-2) #(bs, sl, 3, dim)
                # # pdb.set_trace()
                # candidate_embeddings = (seq_merge_tensor * torch.unsqueeze(seq_merge_weight, dim=-1)).sum(-2)
                candidate_embeddings = candidate_embeddings_graph_attribute
            elif self.args.merge_type == "concat":
                candidate_embeddings = torch.cat((candidate_embeddings_buy, candidate_embeddings_graph_attribute), dim=-1)
            elif self.args.merge_type == "attention" or self.args.merge_type == "attention_attribute" or self.args.merge_type == "behavior_rel_items":
                seq_1 = self.W1_para_1(torch.tanh(self.W_graph_para_1_1(candidate_embeddings_graph_1)))#(bs, sl, 1)
                seq_2 = self.W1_para_2(torch.tanh(self.W_graph_para_2_1(candidate_embeddings_graph_2))) #(bs, sl, 1)
                seq_3 = self.W1_para_3(torch.tanh(self.W_graph_para_3_1(candidate_embeddings_graph_attribute))) #(bs, sl, 1)
                seq_merge = torch.cat([seq_1, seq_2, seq_3], dim=-1)
                seq_merge_weight = nn.functional.softmax(seq_merge, dim=-1) #(bs, sl, 3)
                seq_merge_tensor = torch.cat([torch.unsqueeze(candidate_embeddings_graph_1, dim=-2), torch.unsqueeze(candidate_embeddings_graph_2, dim=-2), torch.unsqueeze(candidate_embeddings_graph_attribute, dim=-2)], dim=-2) #(bs, sl, 3, dim)
                candidate_embeddings = (seq_merge_tensor * torch.unsqueeze(seq_merge_weight, dim=-1)).sum(-2) #(bs, sl, dim)
            elif self.args.merge_type == "add":
                candidate_embeddings = candidate_embeddings_graph_1 + candidate_embeddings_graph_2 + candidate_embeddings_graph_attribute
            elif self.args.merge_type == "simple_add":
                candidate_embeddings = candidate_embeddings_graph_1 + candidate_embeddings_graph_2 + candidate_embeddings_graph_attribute
            
            elif self.args.merge_type == "behavior_rel_items_bad":
                ori_embeddings_graph_1 = self.user_ori[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1)
                ori_embeddings_graph_2 = self.k_user_ori[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1)

                seq_1 = self.W1_para_1(torch.tanh(self.W_graph_para_1_1(candidate_embeddings_graph_1)))#(bs, sl, 1)
                seq_2 = self.W1_para_2(torch.tanh(self.W_graph_para_2_1(candidate_embeddings_graph_2))) #(bs, sl, 1)
                seq_3 = self.W1_para_3(torch.tanh(self.W_graph_para_3_1(candidate_embeddings_graph_attribute))) #(bs, sl, 1)
                ori_1 = self.W1_para_4(torch.tanh(self.W_graph_para_4_1(ori_embeddings_graph_1))) #(bs, sl, 1)
                ori_2 = self.W1_para_5(torch.tanh(self.W_graph_para_5_1(ori_embeddings_graph_2))) #(bs, sl, 1)
                # seq_merge = torch.cat([seq_1, seq_2, seq_3, ori_1, ori_2], dim=-1)
                seq_merge = torch.cat([seq_1, seq_2, seq_3], dim=-1)

                seq_merge_weight = nn.functional.softmax(seq_merge, dim=-1) #(bs, sl, 4)
                # seq_merge_tensor = torch.cat([torch.unsqueeze(candidate_embeddings_graph_1, dim=-2), torch.unsqueeze(candidate_embeddings_graph_2, dim=-2), torch.unsqueeze(candidate_embeddings_graph_attribute, dim=-2), \
                #     torch.unsqueeze(ori_embeddings_graph_1, dim=-2)], dim=-2) #(bs, sl, 5, dim)
                seq_merge_tensor = torch.cat([torch.unsqueeze(candidate_embeddings_graph_1, dim=-2), torch.unsqueeze(candidate_embeddings_graph_2, dim=-2), torch.unsqueeze(candidate_embeddings_graph_attribute, dim=-2)], dim=-2) #(bs, sl, 5, dim)
                # pdb.set_trace()
                graph_e_behavior = (seq_merge_tensor * torch.unsqueeze(seq_merge_weight, dim=-1)).sum(-2) #(bs, sl, dim)


                seq_merge_attri_weight = torch.cat([ori_1, ori_2], dim=-1)
                seq_merge_attri_weight = nn.functional.softmax(seq_merge_attri_weight, dim=-1) #(bs, sl, 4)
                seq_merge_tensor_attri = torch.cat([torch.unsqueeze(ori_embeddings_graph_1, dim=-2), torch.unsqueeze(ori_embeddings_graph_2, dim=-2)], dim=-2) #(bs, sl, 5, dim)
                graph_e_attri = (seq_merge_tensor_attri * torch.unsqueeze(seq_merge_attri_weight, dim=-1)).sum(-2) #(bs, sl, dim) 

                candidate_embeddings_gate = torch.sigmoid(self.W_graph_para_gate_1(graph_e_behavior) + self.W_graph_para_gate_2(graph_e_attri))
                candidate_embeddings = graph_e_behavior * candidate_embeddings_gate + (1. - candidate_embeddings_gate) * graph_e_attri

                #2th attention layer.
                # merge_1 = self.W1_para_merge_1(torch.tanh(self.W_graph_para_merge_1_1(graph_e_behavior))) #(bs, sl, 1)
                # merge_2 = self.W1_para_merge_2(torch.tanh(self.W_graph_para_merge_1_2(graph_e_attri))) #(bs, sl, 1)

                # merge_1_2_weight = torch.cat([merge_1, merge_2], dim=-1)
                # merge_1_2_weight = nn.functional.softmax(merge_1_2_weight, dim=-1) #(bs, sl, 4)

                # seq_v2_attri = torch.cat([torch.unsqueeze(merge_1, dim=-2), torch.unsqueeze(merge_2, dim=-2)], dim=-2) #(bs, sl, 5, dim)
                # graph_e = (seq_v2_attri * torch.unsqueeze(merge_1_2_weight, dim=-1)).sum(-2) #(bs, sl, dim)

                
            elif self.args.merge_type == "test":
                # add side information
                # cate_emb = self.item_rep_graph_attribute[x_cates, :].reshape(x.size(0), x.size(1), -1)
                # brand_emb = self.item_rep_graph_attribute[x_brands, :].reshape(x.size(0), x.size(1), -1)
                # price_emb = self.item_rep_graph_attribute[x_prices, :].reshape(x.size(0), x.size(1), -1)

                # cate_emb = self.item_rep_graph_attribute(x_cates).reshape(x.size(0), x.size(1), -1)
                # brand_emb = self.item_rep_graph_attribute(x_brands).reshape(x.size(0), x.size(1), -1)
                # price_emb = self.item_rep_graph_attribute(x_prices).reshape(x.size(0), x.size(1), -1)

                #concat the representation from the graph and the init item embedding;
                # candidate_embeddings = torch.cat((candidate_embeddings, cate_emb, brand_emb, price_emb), dim=-1) #(bs, sl, 4*dim)
                candidate_embeddings = None
            
            scores = (last_logits * candidate_embeddings).sum(-1)  # B x C
            ret['scores'] = scores
        return ret

    def get_logits(self, d):
        x = d['tokens']
        attn_mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        attn_mask.tril_()  # causal attention for sasrec
        # e = self.token_embedding(d) + self.positional_embedding(d)
        x_unsqueeenze = x.reshape(-1)

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
        # e = self.token_embedding(d)
        e = graph_e
        e = self.dropout(e)
        info = None
        if self.output_info:
            info = {}
        # b = self.body(e, attn_mask, info)  # B x T x H
        b, hidden = self.body(e, None)  # B x T x H
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
        
        # valid_labels_emb = self.token_embedding.emb(valid_labels)  # M x H
        # valid_negative_labels_emb = self.token_embedding.emb(valid_negative_labels)  # M x H

        valid_labels_prob = self.sigmoid((valid_logits * valid_labels_emb).sum(-1))  # M
        valid_negative_labels_prob = self.sigmoid((valid_logits * valid_negative_labels_emb).sum(-1))  # M

        loss = -torch.log(valid_labels_prob + 1e-24) - torch.log((1-valid_negative_labels_prob) + 1e-24)
        loss = loss.mean()
        loss, loss_cnt = loss.unsqueeze(0), loss_cnt.unsqueeze(0)
        return loss, loss_cnt
