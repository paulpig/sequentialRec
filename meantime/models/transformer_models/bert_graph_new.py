from .bert_base import BertBaseModel
from .embeddings import *
from .bodies import BertBody
from .heads import *

import torch.nn as nn
import pdb

class BertModel(BertBaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.output_info = args.output_info
        self.token_embedding = TokenEmbedding(args)
        self.positional_embedding = PositionalEmbedding(args)
        self.body = BertBody(args)
        if args.headtype == 'dot':
            self.head = BertDotProductPredictionHead(args, self.token_embedding.emb)
        elif args.headtype == 'linear':
            self.head = BertLinearPredictionHead(args)
        else:
            raise ValueError
        self.ln = nn.LayerNorm(args.hidden_units)
        self.dropout = nn.Dropout(p=args.dropout)
        self.createMergeParameter()
        self.init_weights()

    @classmethod
    def code(cls):
        return 'bert_graph_merge_new'


    def createMergeParameter(self):
        # hidden_W = self.args.hidden_units // 2
        if self.args.merge_type == "concat":
            hidden_W = self.args.hidden_units //2
        else:
            hidden_W = self.args.hidden_units
        
        self.W1_para_1 = nn.Linear(self.args.latent_dim_rec, 1).cuda()
        self.W1_para_2 = nn.Linear(self.args.latent_dim_rec, 1).cuda()
        self.W1_para_3 = nn.Linear(self.args.latent_dim_rec, 1).cuda()
        self.W1_para_4 = nn.Linear(self.args.latent_dim_rec, 1).cuda()
        self.W1_para_5 = nn.Linear(self.args.latent_dim_rec, 1).cuda()

        self.W1_para_1_a = nn.Linear(self.args.latent_dim_rec, 1).cuda()
        self.W1_para_2_a = nn.Linear(self.args.latent_dim_rec, 1).cuda()
        self.W1_para_3_a = nn.Linear(self.args.latent_dim_rec, 1).cuda()

        self.W_graph_para_1 = nn.Linear(self.args.latent_dim_rec, self.args.latent_dim_rec).cuda()
        self.W_graph_para_2 = nn.Linear(self.args.latent_dim_rec, self.args.latent_dim_rec).cuda()

        self.W_graph_para_1_1 = nn.Linear(self.args.latent_dim_rec, self.args.latent_dim_rec).cuda()
        self.W_graph_para_2_1 = nn.Linear(self.args.latent_dim_rec, self.args.latent_dim_rec).cuda()
        self.W_graph_para_3_1 = nn.Linear(self.args.latent_dim_rec, self.args.latent_dim_rec).cuda()
        self.W_graph_para_4_1 = nn.Linear(self.args.latent_dim_rec, self.args.latent_dim_rec).cuda()
        self.W_graph_para_5_1 = nn.Linear(self.args.latent_dim_rec, self.args.latent_dim_rec).cuda()

        self.W_graph_para_1_2 = nn.Linear(self.args.latent_dim_rec, self.args.latent_dim_rec).cuda()
        self.W_graph_para_2_2 = nn.Linear(self.args.latent_dim_rec, self.args.latent_dim_rec).cuda()
        self.W_graph_para_3_2 = nn.Linear(self.args.latent_dim_rec, self.args.latent_dim_rec).cuda()
        # W_graph_para_3_1


        self.W_graph_para_gate_1 = nn.Linear(self.args.latent_dim_rec, self.args.latent_dim_rec).cuda()
        self.W_graph_para_gate_2 = nn.Linear(self.args.latent_dim_rec, self.args.latent_dim_rec).cuda()

        self.W_graph_para_merge_1_1 = nn.Linear(self.args.latent_dim_rec, self.args.latent_dim_rec).cuda()
        self.W_graph_para_merge_1_2 = nn.Linear(self.args.latent_dim_rec, self.args.latent_dim_rec).cuda()

        self.W1_para_merge_1 = nn.Linear(self.args.latent_dim_rec, 1).cuda()
        self.W1_para_merge_2 = nn.Linear(self.args.latent_dim_rec, 1).cuda()
        
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
    # def setUserItemRepFromGraph(self, user_rep_graph, item_rep_graph):
    #     """
    #     The representations from the LightGCN model; 
    #     user_rep_graph: (|users|, dim)
    #     item_rep_graph: (|items|, dim)
    #     """
    #     self.user_rep_graph = user_rep_graph
    #     self.item_rep_graph = item_rep_graph

    #     # #增加一行是mask token对应的表征
    #     # mask_emb1 = torch.zeros(1, user_rep_graph.size(-1)).cuda()
    #     # mask_emb2 = torch.zeros(1, user_rep_graph.size(-1)).cuda()
    #     # self.user_rep_graph = torch.cat()
    #     return

    def get_logits(self, d):
        x = d['tokens']
        attn_mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        # e = self.token_embedding(d) + self.positional_embedding(d)
        # pdb.set_trace() #由于多特殊的mask token, 导致超出词表数量.
        #采用图模型输出的表征初始化序列推荐模型item lookup table, 初始化的效果增强;
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
            ori_embeddings_graph_1 = self.user_ori[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1)
            # ori_embeddings_graph_2 = self.k_user_ori[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1)
            #attention merge
            seq_1 = self.W1_para_1(torch.tanh(self.W_graph_para_1_1(ori_embeddings_graph_1)))#(bs, sl, 1)
            seq_2 = self.W1_para_2(torch.tanh(self.W_graph_para_2_1(candidate_embeddings_graph_1))) #(bs, sl, 1)
            seq_3 = self.W1_para_3(torch.tanh(self.W_graph_para_3_1(candidate_embeddings_graph_2)))
            seq_merge = torch.cat([seq_1, seq_2, seq_3], dim=-1)
            seq_merge_weight = nn.functional.softmax(seq_merge, dim=-1) #(bs, sl, 3)
            seq_merge_tensor = torch.cat([torch.unsqueeze(ori_embeddings_graph_1, dim=-2), torch.unsqueeze(candidate_embeddings_graph_1, dim=-2), torch.unsqueeze(candidate_embeddings_graph_2, dim=-2)], dim=-2) #(bs, sl, 3, dim)
            # pdb.set_trace()
            graph_e = (seq_merge_tensor * torch.unsqueeze(seq_merge_weight, dim=-1)).sum(-2) #(bs, sl, dim)
            # graph_e = graph_e_buy
        elif self.args.merge_type == "rm_lightgcn":
            # ori_embeddings_graph_2 = self.k_user_ori[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1)
            # # graph_e = candidate_embeddings_graph_attribute
            # seq_1 = self.W1_para_1(torch.tanh(self.W_graph_para_1_1(candidate_embeddings_graph_attribute)))#(bs, sl, 1)
            # seq_2 = self.W1_para_2(torch.tanh(self.W_graph_para_2_1(ori_embeddings_graph_2))) #
            # seq_merge = torch.cat([seq_1, seq_2], dim=-1)
            # seq_merge_weight = nn.functional.softmax(seq_merge, dim=-1) #(bs, sl, 3)
            # seq_merge_tensor = torch.cat([torch.unsqueeze(candidate_embeddings_graph_attribute, dim=-2), torch.unsqueeze(ori_embeddings_graph_2, dim=-2)], dim=-2) #(bs, sl, 3, dim)
            # # pdb.set_trace()
            # graph_e = (seq_merge_tensor * torch.unsqueeze(seq_merge_weight, dim=-1)).sum(-2) #(bs, sl, dim)
            graph_e = candidate_embeddings_graph_attribute
        elif self.args.merge_type == "concat":
            graph_e = torch.cat((graph_e_buy, candidate_embeddings_graph_attribute), dim=-1)
        elif self.args.merge_type == "attention":
            # pdb.set_trace()
            seq_1 = self.W1_para_1(torch.tanh(self.W_graph_para_1_1(candidate_embeddings_graph_1)))#(bs, sl, 1)
            seq_2 = self.W1_para_2(torch.tanh(self.W_graph_para_2_1(candidate_embeddings_graph_2))) #(bs, sl, 1)
            seq_3 = self.W1_para_3(torch.tanh(self.W_graph_para_3_1(candidate_embeddings_graph_attribute))) #(bs, sl, 1)
            seq_merge = torch.cat([seq_1, seq_2, seq_3], dim=-1)
            seq_merge_weight = nn.functional.softmax(seq_merge, dim=-1) #(bs, sl, 3)
            seq_merge_tensor = torch.cat([torch.unsqueeze(candidate_embeddings_graph_1, dim=-2), torch.unsqueeze(candidate_embeddings_graph_2, dim=-2), torch.unsqueeze(candidate_embeddings_graph_attribute, dim=-2)], dim=-2) #(bs, sl, 3, dim)
            # pdb.set_trace()
            graph_e = (seq_merge_tensor * torch.unsqueeze(seq_merge_weight, dim=-1)).sum(-2) #(bs, sl, dim)
        elif self.args.merge_type == "attention_attribute":
            # pdb.set_trace()
            cates = d['seq_cates'] #(bs, sl)
            brands = d['seq_brands'] #(bs, sl)
            prices = d['seq_prices'] #(bs, sl)
            cate_emb = self.item_rep_graph_attribute[cates, :].reshape(x.size(0), x.size(1), -1)
            brand_emb = self.item_rep_graph_attribute[brands, :].reshape(x.size(0), x.size(1), -1)
            price_emb = self.item_rep_graph_attribute[prices, :].reshape(x.size(0), x.size(1), -1)
            seq_1 = self.W1_para_1(torch.tanh(self.W_graph_para_1_1(candidate_embeddings_graph_1)))#(bs, sl, 1)
            seq_2 = self.W1_para_2(torch.tanh(self.W_graph_para_2_1(candidate_embeddings_graph_2))) #(bs, sl, 1)
            seq_3 = self.W1_para_3(torch.tanh(self.W_graph_para_3_1(candidate_embeddings_graph_attribute))) #(bs, sl, 1)
            cate_1 = self.W1_para_1_a(torch.tanh(self.W_graph_para_1_2(cate_emb))) #(bs, sl, 1)
            brand_1 = self.W1_para_2_a(torch.tanh(self.W_graph_para_2_2(brand_emb))) #(bs, sl, 1)
            price_1 = self.W1_para_3_a(torch.tanh(self.W_graph_para_3_2(price_emb))) #(bs, sl, 1)
            seq_merge = torch.cat([seq_1, seq_2, seq_3, cate_1, brand_1, price_1], dim=-1)
            seq_merge_weight = nn.functional.softmax(seq_merge, dim=-1) #(bs, sl, 6)
            seq_merge_tensor = torch.cat([self.W_graph_para_1_1(torch.unsqueeze(candidate_embeddings_graph_1, dim=-2)), self.W_graph_para_2_1(torch.unsqueeze(candidate_embeddings_graph_2, dim=-2)), self.W_graph_para_3_1(torch.unsqueeze(candidate_embeddings_graph_attribute, dim=-2)), \
                self.W_graph_para_1_2(torch.unsqueeze(cate_emb, dim=-2)), self.W_graph_para_2_2(torch.unsqueeze(brand_emb, dim=-2)), self.W_graph_para_3_2(torch.unsqueeze(price_emb, dim=-2))], dim=-2) #(bs, sl, 3, dim)
            # pdb.set_trace()
            graph_e = (seq_merge_tensor * torch.unsqueeze(seq_merge_weight, dim=-1)).sum(-2) #(bs, sl, dim)
        elif self.args.merge_type == "simple_add":
            ori_embeddings_graph_1 = self.user_ori[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1)
            ori_embeddings_graph_2 = self.k_user_ori[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1)
            graph_e = candidate_embeddings_graph_1 + candidate_embeddings_graph_2 + candidate_embeddings_graph_attribute + ori_embeddings_graph_1 + ori_embeddings_graph_2
        # elif self.args.merge_type == "simple_add":
        #     candidate_embeddings = candidate_embeddings_graph_1 + candidate_embeddings_graph_2 + candidate_embeddings_graph_attribute
        elif self.args.merge_type == "behavior_rel_items":
            # behavior_rel_items = d['behavior_rel_items'] #(bs, sl, 3)
            # # behavior_rel_items_test = behavior_rel_items.reshape(-1)
            # # behavior_rel_items_test_recover = behavior_rel_items_test.reshape(behavior_rel_items.size(0), behavior_rel_items.size(1), behavior_rel_items.size(2))
            # # pdb.set_trace()
            # behavior_rel_items_mask = behavior_rel_items > 0  #(bs, sl, 3)
            # behavior_rel_items_squeeze = behavior_rel_items.reshape(-1) #(bs*sl*3), self.user_rep_graph_buy[behavior_rel_items[-1][-1]]

            # # behavior_rel_in = self.item_rep_graph_buy[behavior_rel_items_squeeze, :].reshape(behavior_rel_items.size(0), behavior_rel_items.size(1), behavior_rel_items.size(2), -1) #(bs, sl, 3, dim)
            # behavior_rel_in = self.user_rep_graph_buy[behavior_rel_items_squeeze, :].reshape(behavior_rel_items.size(0), behavior_rel_items.size(1), behavior_rel_items.size(2), -1) #(bs, sl, 3, dim)
            
            # attention_score = torch.einsum('bld,blid->bli', candidate_embeddings_graph_1, behavior_rel_in) #(bs, sl, 3)
            # # attention_score = torch.einsum('bld,blid->bli', candidate_embeddings_graph_2, behavior_rel_in) #(bs, sl, 3)
            # attention_score = attention_score.masked_fill(behavior_rel_items_mask == 0, -1e9)

            # attention_score = F.softmax(attention_score, dim=-1)  # (bs, sl, 3)
            
            # behavior_rel_in = (behavior_rel_in * attention_score.unsqueeze(-1)).sum(-2) #(bs, sl, dim)


            # behavior_rel_in = behavior_rel_in.sum(-2)
            # pdb.set_trace()
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


            #2th attention layer.
            # merge_1 = self.W1_para_merge_1(torch.tanh(self.W_graph_para_merge_1_1(graph_e_behavior))) #(bs, sl, 1)
            # merge_2 = self.W1_para_merge_2(torch.tanh(self.W_graph_para_merge_1_2(graph_e_attri))) #(bs, sl, 1)

            # merge_1_2_weight = torch.cat([merge_1, merge_2], dim=-1)
            # merge_1_2_weight = nn.functional.softmax(merge_1_2_weight, dim=-1) #(bs, sl, 4)

            # seq_v2_attri = torch.cat([torch.unsqueeze(merge_1, dim=-2), torch.unsqueeze(merge_2, dim=-2)], dim=-2) #(bs, sl, 5, dim)
            # graph_e = (seq_v2_attri * torch.unsqueeze(merge_1_2_weight, dim=-1)).sum(-2) #(bs, sl, dim)

            candidate_embeddings_gate = torch.sigmoid(self.W_graph_para_gate_1(graph_e_behavior) + self.W_graph_para_gate_2(graph_e_attri))
            graph_e = graph_e_behavior * candidate_embeddings_gate + (1. - candidate_embeddings_gate) * graph_e_attri
        # graph_e = self.token_embedding(d)
        # graph_e = self.item_rep_graph[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1) #(bs, sl, dim) #use the hidden representation to init the embedding lookup table;
        e = graph_e + self.positional_embedding(d)

        e = self.ln(e)
        e = self.dropout(e)  # B x T x H
        # pdb.set_trace()
        info = {} if self.output_info else None
        
        b = self.body(e, attn_mask, info)  # B x T x H
        return b, info

    def get_scores(self, d, logits):
        # logits : B x H or M x H
        seq_1 = self.W1_para_1(torch.tanh(self.W_graph_para_1_1(self.item_rep_graph_buy)))#(bs, sl, 1)
        seq_2 = self.W1_para_2(torch.tanh(self.W_graph_para_2_1(self.user_rep_graph_buy))) #(bs, sl, 1)
        seq_3 = self.W1_para_3(torch.tanh(self.W_graph_para_3_1(self.user_rep_graph_attribute))) #(bs, sl, 1)
        seq_merge = torch.cat([seq_1, seq_2, seq_3], dim=-1)
        seq_merge_weight = nn.functional.softmax(seq_merge, dim=-1) #(bs, sl, 3)
        seq_merge_tensor = torch.cat([torch.unsqueeze(self.item_rep_graph_buy, dim=-2), torch.unsqueeze(self.user_rep_graph_buy, dim=-2), torch.unsqueeze(self.user_rep_graph_attribute, dim=-2)], dim=-2) #(bs, sl, 3, dim)
        # pdb.set_trace()
        graph_e = (seq_merge_tensor * torch.unsqueeze(seq_merge_weight, dim=-1)).sum(-2) #(bs, sl, dim)
        self.head.token_embeddings.from_pretrained(graph_e) #初始化embedding层

        if self.training:  # logits : M x H, returns M x V
            h = self.head(logits)  # M x V
        else:  # logits : B x H,  returns B x C
            candidates = d['candidates']  # B x C
            h = self.head(logits, candidates)  # B x C
        return h