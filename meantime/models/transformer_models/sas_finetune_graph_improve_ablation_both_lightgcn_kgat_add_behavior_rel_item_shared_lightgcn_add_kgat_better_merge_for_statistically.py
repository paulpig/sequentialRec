from meantime.dataloaders import graph
from ..base import BaseModel
from .embeddings import *
# from .bodies import ExactSasBody
from .bodies import SasBody
from .heads import *
from torch.nn import functional as F


import torch
import torch.nn as nn
import pdb

class SASModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.output_info = args.output_info
        self.token_embedding = TokenEmbedding(args)
        self.positional_embedding = PositionalEmbedding(args)
        if args.encoder_type == "sasrec":
            self.body = SasBody(args) #hidden_size = hidden_size * 2
        elif args.encoder_type == "gru4rec":
            self.body = nn.GRU(input_size=args.hidden_units, hidden_size=args.hidden_units, batch_first=True)

        self.ln = nn.LayerNorm(args.hidden_units)
        self.dropout = nn.Dropout(p=args.dropout)

        # pdb.set_trace()
        # self.item_rep_graph_attribute = nn.Embedding(args.num_attributes + 3, args.latent_dim_rec, padding_idx=0)
        # self.item_rep_graph_attribute = nn.Embedding(args.num_attributes + 3, args.feature_dim, padding_idx=0)
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

        self.concat_layer = nn.Linear(args.hidden_units + 3*args.feature_dim, args.hidden_units)

        #创建gate所需函数;
        self.createMergeParameter()
        self.init_weights()



        # self.user_rep_graph_buy_tmp = None
        # self.item_rep_graph_buy_tmp = None
        # self.user_rep_graph_attribute_tmp = None
        # self.item_rep_graph_attribute_tmp = None

        self.sigmoid = nn.Sigmoid()
        self.vocab_size = args.num_items + 1

    @classmethod
    def code(cls):
        return 'graph_sasrec_improve_lightgcn_kgat_behavoir_rel_items_shared_lightgcn_add_kgat_better_fusion_for_statistically'


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

    def saveTensorAsWeight(self):
        # self.user_rep_graph_buy  = torch.nn.Parameter(self.user_rep_graph_buy)
        # self.user_rep_graph_buy  = torch.nn.Parameter(self.user_rep_graph_buy)
        # self.user_rep_graph_buy  = torch.nn.Parameter(self.user_rep_graph_buy)
        # self.user_rep_graph_buy  = torch.nn.Parameter(self.user_rep_graph_buy)
        self.user_rep_graph_buy_tmp  = torch.nn.Parameter(self.user_rep_graph_buy)
        self.item_rep_graph_buy_tmp  = torch.nn.Parameter(self.item_rep_graph_buy)
        self.user_rep_graph_attribute_tmp  = torch.nn.Parameter(self.user_rep_graph_attribute)
        self.item_rep_graph_attribute_tmp  = torch.nn.Parameter(self.item_rep_graph_attribute)
        
        # self.user_ori  = torch.nn.Parameter(self.user_ori)
        # self.item_ori  = torch.nn.Parameter(self.item_ori)
        # self.k_user_ori  = torch.nn.Parameter(self.k_user_ori)
        # self.k_item_ori  = torch.nn.Parameter(self.k_item_ori)
    
    def reloadWeightForTest(self):
        self.user_rep_graph_buy = self.user_rep_graph_buy_tmp
        self.item_rep_graph_buy = self.item_rep_graph_buy_tmp
        self.user_rep_graph_attribute = self.user_rep_graph_attribute_tmp
        self.item_rep_graph_attribute = self.item_rep_graph_attribute_tmp



    def setUserItemRepFromGraph(self, user_rep_buy, item_rep_buy, user_rep_view, item_rep_view, user_ori, item_ori, k_user_ori, k_item_ori):
        """
        The representations from the LightGCN model; 
        user_rep_graph: (|users|, dim)
        item_rep_graph: (|items|, dim)
        """

        # self.user_rep_graph_buy  = torch.nn.Parameter(user_rep_buy)
        # self.item_rep_graph_buy  = torch.nn.Parameter(item_rep_buy)
        # self.user_rep_graph_attribute  = torch.nn.Parameter(user_rep_view)
        # self.item_rep_graph_attribute  = torch.nn.Parameter(item_rep_view)
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
        """
        add the graph representation; 
        """
        logits, info = self.get_logits(d)

        # pdb.set_trace()
        ret = {'logits':logits, 'info':info}
        if self.training:
            labels = d['labels']
            negative_labels = d['negative_labels']
            # cates = d['seq_cates'] #(bs, sl)
            # brands = d['seq_brands'] #(bs, sl)
            # prices = d['seq_prices'] #(bs, sl)
            loss, loss_cnt = self.get_loss(logits, labels, negative_labels, d)
            # loss, loss_cnt = self.get_loss_as_bert(logits, labels, negative_labels, d)
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

        # candidate_embeddings_graph_2 = self.user_rep_graph_view[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1)
        # candidate_embeddings_gate = torch.sigmoid(self.W_graph_para_1_1(graph_e_buy) + self.W_graph_para_2_1(candidate_embeddings_graph_attribute))
        # gate = torch.sigmoid(self.W1_para(candidate_embeddings_bert) + self.W2_para(candidate_embeddings_graph))
        # graph_e = candidate_embeddings_gate * graph_e_buy + (1. - candidate_embeddings_gate) * candidate_embeddings_graph_attribute
        
        # graph_e = torch.cat((graph_e_buy, candidate_embeddings_graph_attribute), dim=-1) #(bs, sl, 2*dim)
        # graph_e = graph_e_buy + graph_e_review
        # graph_e = graph_e_buy

        e = graph_e
        # e = graph_e

        #add feature
        
        # pdb.set_trace()
        # cate_emb = self.item_rep_graph_attribute(cates).reshape(x.size(0), x.size(1), -1)
        # brand_emb = self.item_rep_graph_attribute(brands).reshape(x.size(0), x.size(1), -1)
        # price_emb = self.item_rep_graph_attribute(prices).reshape(x.size(0), x.size(1), -1)
        
        #concat the representation from the graph and the init item embedding;
        # e = torch.cat((e, cate_emb, brand_emb, price_emb), dim=-1) #(bs, sl, 4*dim)
        # e = self.concat_layer(e)

        if self.args.encoder_type == "sasrec":

            e = e+ self.positional_embedding(d)

            b = self.ln(e)
            e = self.dropout(e)
            info = None
            if self.output_info:
                info = {}
            b = self.body(e, attn_mask, info)  # B x T x H
            b = self.ln(b)  # original code does this at the end of body
        elif self.args.encoder_type == "gru4rec":
            e = self.dropout(e)
            info = None
            if self.output_info:
                info = {}
            # b = self.body(e, attn_mask, info)  # B x T x H
            b, hidden = self.body(e, None)  # B x T x H
            b = self.ln(b)  # original code 
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

    def get_loss(self, logits, labels, negative_labels, d):
        _logits = logits.reshape(-1, logits.size(-1))  # BT x H
        _labels = labels.reshape(-1)  # BT
        _negative_labels = negative_labels.reshape(-1)  # BT


        # label_cates = d["label_cates"].reshape(-1)
        # label_brands = d["label_brands"].reshape(-1)
        # label_prices = d["label_prices"].reshape(-1)


        # neg_cates = d["neg_cates"].reshape(-1)
        # neg_brands = d["neg_cates"].reshape(-1)
        # neg_prices = d["neg_cates"].reshape(-1)

        valid = _labels > 0
        loss_cnt = valid.sum()  # = M
        valid_index = valid.nonzero().squeeze()  # M

        valid_logits = _logits[valid_index]  # M x H
        valid_labels = _labels[valid_index]  # M
        valid_negative_labels = _negative_labels[valid_index]  # M

        # label_cates = label_cates[valid_index]
        # label_brands = label_brands[valid_index]
        # label_prices = label_prices[valid_index]

        # neg_cates = neg_cates[valid_index]
        # neg_brands = neg_brands[valid_index]
        # neg_prices = neg_prices[valid_index]


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
            # valid_labels_emb = valid_labels_emb_buy
            seq_2 = self.W1_para_2(torch.tanh(self.W_graph_para_2_1(valid_labels_emb_graph_1))) #(bs, sl, 1)
            seq_3 = self.W1_para_3(torch.tanh(self.W_graph_para_3_1(valid_labels_emb_graph_2)))
            seq_merge = torch.cat([seq_2, seq_3], dim=-1)
            seq_merge_weight = nn.functional.softmax(seq_merge, dim=-1) #(bs, sl, 3)
            seq_merge_tensor = torch.cat([torch.unsqueeze(valid_labels_emb_graph_1, dim=-2), torch.unsqueeze(valid_labels_emb_graph_2, dim=-2)], dim=-2) #(bs, sl, 3, dim)
            # pdb.set_trace()
            valid_labels_emb = (seq_merge_tensor * torch.unsqueeze(seq_merge_weight, dim=-1)).sum(-2) #(bs, sl, dim)

        elif self.args.merge_type == "rm_lightgcn":
            # ori_embeddings_graph_2 = self.k_user_ori[valid_labels, :]
            # # graph_e = candidate_embeddings_graph_attribute
            # seq_1 = self.W1_para_1(torch.tanh(self.W_graph_para_1_1(valid_labels_emb_graph_attribute)))#(bs, sl, 1)
            # seq_2 = self.W1_para_2(torch.tanh(self.W_graph_para_2_1(ori_embeddings_graph_2))) #
            # seq_merge = torch.cat([seq_1, seq_2], dim=-1)
            # seq_merge_weight = nn.functional.softmax(seq_merge, dim=-1) #(bs, sl, 3)
            # seq_merge_tensor = torch.cat([torch.unsqueeze(valid_labels_emb_graph_attribute, dim=-2), torch.unsqueeze(ori_embeddings_graph_2, dim=-2)], dim=-2) #(bs, sl, 3, dim)
            # # pdb.set_trace()
            # valid_labels_emb = (seq_merge_tensor * torch.unsqueeze(seq_merge_weight, dim=-1)).sum(-2) #(bs, sl, dim)
            valid_labels_emb = valid_labels_emb_graph_attribute
        elif self.args.merge_type == "concat":
            valid_labels_emb = torch.cat((valid_labels_emb_buy, valid_labels_emb_graph_attribute), dim=-1)
        elif self.args.merge_type == "attention" or self.args.merge_type == "attention_attribute" or self.args.merge_type == "behavior_rel_items":
            seq_1 = self.W1_para_1(torch.tanh(self.W_graph_para_1_1(valid_labels_emb_graph_1)))#(bs, sl, 1)
            seq_2 = self.W1_para_2(torch.tanh(self.W_graph_para_2_1(valid_labels_emb_graph_2))) #(bs, sl, 1)
            seq_3 = self.W1_para_3(torch.tanh(self.W_graph_para_3_1(valid_labels_emb_graph_attribute))) #(bs, sl, 1)
            seq_merge = torch.cat([seq_1, seq_2, seq_3], dim=-1)
            seq_merge_weight = nn.functional.softmax(seq_merge, dim=-1) #(bs, sl, 3)
            seq_merge_tensor = torch.cat([torch.unsqueeze(valid_labels_emb_graph_1, dim=-2), torch.unsqueeze(valid_labels_emb_graph_2, dim=-2), torch.unsqueeze(valid_labels_emb_graph_attribute, dim=-2)], dim=-2) #(bs, sl, 3, dim)
            valid_labels_emb = (seq_merge_tensor * torch.unsqueeze(seq_merge_weight, dim=-1)).sum(-2) #(bs, sl, dim)
        elif self.args.merge_type == "simple_add":
            valid_labels_emb = valid_labels_emb_graph_1 + valid_labels_emb_graph_2 + valid_labels_emb_graph_attribute

        elif self.args.merge_type =="behavior_rel_items_bad":
            ori_embeddings_graph_1 = self.user_ori[valid_labels, :]
            ori_embeddings_graph_2 = self.k_user_ori[valid_labels, :]

            seq_1 = self.W1_para_1(torch.tanh(self.W_graph_para_1_1(valid_labels_emb_graph_1)))#(bs, sl, 1)
            seq_2 = self.W1_para_2(torch.tanh(self.W_graph_para_2_1(valid_labels_emb_graph_2))) #(bs, sl, 1)
            seq_3 = self.W1_para_3(torch.tanh(self.W_graph_para_3_1(valid_labels_emb_graph_attribute))) #(bs, sl, 1)
            ori_1 = self.W1_para_4(torch.tanh(self.W_graph_para_4_1(ori_embeddings_graph_1))) #(bs, sl, 1)
            ori_2 = self.W1_para_5(torch.tanh(self.W_graph_para_5_1(ori_embeddings_graph_2))) #(bs, sl, 1)
            # seq_merge = torch.cat([seq_1, seq_2, seq_3, ori_1, ori_2], dim=-1)
            seq_merge = torch.cat([seq_1, seq_2, seq_3], dim=-1)

            seq_merge_weight = nn.functional.softmax(seq_merge, dim=-1) #(bs, sl, 4)
            # seq_merge_tensor = torch.cat([torch.unsqueeze(candidate_embeddings_graph_1, dim=-2), torch.unsqueeze(candidate_embeddings_graph_2, dim=-2), torch.unsqueeze(candidate_embeddings_graph_attribute, dim=-2), \
            #     torch.unsqueeze(ori_embeddings_graph_1, dim=-2)], dim=-2) #(bs, sl, 5, dim)
            seq_merge_tensor = torch.cat([torch.unsqueeze(valid_labels_emb_graph_1, dim=-2), torch.unsqueeze(valid_labels_emb_graph_2, dim=-2), torch.unsqueeze(valid_labels_emb_graph_attribute, dim=-2)], dim=-2) #(bs, sl, 5, dim)
            # pdb.set_trace()
            graph_e_behavior = (seq_merge_tensor * torch.unsqueeze(seq_merge_weight, dim=-1)).sum(-2) #(bs, sl, dim)


            seq_merge_attri_weight = torch.cat([ori_1, ori_2], dim=-1)
            seq_merge_attri_weight = nn.functional.softmax(seq_merge_attri_weight, dim=-1) #(bs, sl, 4)
            seq_merge_tensor_attri = torch.cat([torch.unsqueeze(ori_embeddings_graph_1, dim=-2), torch.unsqueeze(ori_embeddings_graph_2, dim=-2)], dim=-2) #(bs, sl, 5, dim)
            graph_e_attri = (seq_merge_tensor_attri * torch.unsqueeze(seq_merge_attri_weight, dim=-1)).sum(-2) #(bs, sl, dim) 

            candidate_embeddings_gate = torch.sigmoid(self.W_graph_para_gate_1(graph_e_behavior) + self.W_graph_para_gate_2(graph_e_attri))
            valid_labels_emb = graph_e_behavior * candidate_embeddings_gate + (1. - candidate_embeddings_gate) * graph_e_attri
        # valid_labels_emb_graph_2 = self.item_rep_graph_view[valid_labels, :]  # M x H
        # valid_labels_gate = torch.sigmoid(self.W_graph_para_1_1(valid_labels_emb_buy) + self.W_graph_para_2_1(valid_labels_emb_graph_attribute))
        # valid_labels_emb = valid_labels_gate * valid_labels_emb_buy + (1. - valid_labels_gate) * valid_labels_emb_graph_attribute
        # valid_labels_emb = torch.cat((valid_labels_emb_buy, valid_labels_emb_graph_attribute), dim=-1)

        # valid_labels_emb = valid_labels_emb_buy + valid_labels_emb_review
        # valid_labels_emb = valid_labels_emb_buy
        # cate_emb = self.item_rep_graph_attribute[label_cates, :]
        # brand_emb = self.item_rep_graph_attribute[label_brands, :]
        # price_emb = self.item_rep_graph_attribute[label_prices, :]
        
        # cate_emb = self.item_rep_graph_attribute(label_cates)
        # brand_emb = self.item_rep_graph_attribute(label_brands)
        # price_emb = self.item_rep_graph_attribute(label_prices)

        # valid_labels_emb = torch.cat((valid_labels_emb, cate_emb, brand_emb, price_emb), dim=-1) #(bs, sl, 4*dim)

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
            # valid_negative_labels_emb = valid_negative_labels_emb_buy
            seq_2 = self.W1_para_2(torch.tanh(self.W_graph_para_2_1(valid_negative_labels_emb_graph_1))) #(bs, sl, 1)
            seq_3 = self.W1_para_3(torch.tanh(self.W_graph_para_3_1(valid_negative_labels_emb_graph_2)))
            seq_merge = torch.cat([seq_2, seq_3], dim=-1)
            seq_merge_weight = nn.functional.softmax(seq_merge, dim=-1) #(bs, sl, 3)
            seq_merge_tensor = torch.cat([torch.unsqueeze(valid_negative_labels_emb_graph_1, dim=-2), torch.unsqueeze(valid_negative_labels_emb_graph_2, dim=-2)], dim=-2) #(bs, sl, 3, dim)
            # pdb.set_trace()
            valid_negative_labels_emb = (seq_merge_tensor * torch.unsqueeze(seq_merge_weight, dim=-1)).sum(-2) #(bs, sl, dim)
        elif self.args.merge_type == "rm_lightgcn":
            # ori_embeddings_graph_2 = self.k_user_ori[valid_negative_labels, :]
            # # graph_e = candidate_embeddings_graph_attribute
            # seq_1 = self.W1_para_1(torch.tanh(self.W_graph_para_1_1(valid_negative_labels_emb_graph_attribute)))#(bs, sl, 1)
            # seq_2 = self.W1_para_2(torch.tanh(self.W_graph_para_2_1(ori_embeddings_graph_2))) #
            # seq_merge = torch.cat([seq_1, seq_2], dim=-1)
            # seq_merge_weight = nn.functional.softmax(seq_merge, dim=-1) #(bs, sl, 3)
            # seq_merge_tensor = torch.cat([torch.unsqueeze(valid_negative_labels_emb_graph_attribute, dim=-2), torch.unsqueeze(ori_embeddings_graph_2, dim=-2)], dim=-2) #(bs, sl, 3, dim)
            # # pdb.set_trace()
            # valid_negative_labels_emb = (seq_merge_tensor * torch.unsqueeze(seq_merge_weight, dim=-1)).sum(-2) #(bs, sl, dim)
            valid_negative_labels_emb = valid_negative_labels_emb_graph_attribute
        elif self.args.merge_type == "concat":
            valid_negative_labels_emb = torch.cat((valid_negative_labels_emb_buy, valid_negative_labels_emb_graph_attribute), dim=-1)
        elif self.args.merge_type == "attention" or self.args.merge_type == "attention_attribute" or self.args.merge_type == "behavior_rel_items":
            seq_1 = self.W1_para_1(torch.tanh(self.W_graph_para_1_1(valid_negative_labels_emb_graph_1)))#(bs, sl, 1)
            seq_2 = self.W1_para_2(torch.tanh(self.W_graph_para_2_1(valid_negative_labels_emb_graph_2))) #(bs, sl, 1)
            seq_3 = self.W1_para_3(torch.tanh(self.W_graph_para_3_1(valid_negative_labels_emb_graph_attribute))) #(bs, sl, 1)
            seq_merge = torch.cat([seq_1, seq_2, seq_3], dim=-1)
            seq_merge_weight = nn.functional.softmax(seq_merge, dim=-1) #(bs, sl, 3)
            seq_merge_tensor = torch.cat([torch.unsqueeze(valid_negative_labels_emb_graph_1, dim=-2), torch.unsqueeze(valid_negative_labels_emb_graph_2, dim=-2), torch.unsqueeze(valid_negative_labels_emb_graph_attribute, dim=-2)], dim=-2) #(bs, sl, 3, dim)
            valid_negative_labels_emb = (seq_merge_tensor * torch.unsqueeze(seq_merge_weight, dim=-1)).sum(-2) #(bs, sl, dim)
        elif self.args.merge_type == "simple_add":
            valid_negative_labels_emb = valid_negative_labels_emb_graph_1 + valid_negative_labels_emb_graph_2 + valid_negative_labels_emb_graph_attribute
        
        elif self.args.merge_type =="behavior_rel_items_bad":
            ori_embeddings_graph_1 = self.user_ori[valid_negative_labels, :]
            ori_embeddings_graph_2 = self.k_user_ori[valid_negative_labels, :]

            seq_1 = self.W1_para_1(torch.tanh(self.W_graph_para_1_1(valid_negative_labels_emb_graph_1)))#(bs, sl, 1)
            seq_2 = self.W1_para_2(torch.tanh(self.W_graph_para_2_1(valid_negative_labels_emb_graph_2))) #(bs, sl, 1)
            seq_3 = self.W1_para_3(torch.tanh(self.W_graph_para_3_1(valid_negative_labels_emb_buy))) #(bs, sl, 1)
            ori_1 = self.W1_para_4(torch.tanh(self.W_graph_para_4_1(ori_embeddings_graph_1))) #(bs, sl, 1)
            ori_2 = self.W1_para_5(torch.tanh(self.W_graph_para_5_1(ori_embeddings_graph_2))) #(bs, sl, 1)
            # seq_merge = torch.cat([seq_1, seq_2, seq_3, ori_1, ori_2], dim=-1)
            seq_merge = torch.cat([seq_1, seq_2, seq_3], dim=-1)

            seq_merge_weight = nn.functional.softmax(seq_merge, dim=-1) #(bs, sl, 4)
            # seq_merge_tensor = torch.cat([torch.unsqueeze(candidate_embeddings_graph_1, dim=-2), torch.unsqueeze(candidate_embeddings_graph_2, dim=-2), torch.unsqueeze(candidate_embeddings_graph_attribute, dim=-2), \
            #     torch.unsqueeze(ori_embeddings_graph_1, dim=-2)], dim=-2) #(bs, sl, 5, dim)
            seq_merge_tensor = torch.cat([torch.unsqueeze(valid_negative_labels_emb_graph_1, dim=-2), torch.unsqueeze(valid_negative_labels_emb_graph_2, dim=-2), torch.unsqueeze(valid_negative_labels_emb_buy, dim=-2)], dim=-2) #(bs, sl, 5, dim)
            # pdb.set_trace()
            graph_e_behavior = (seq_merge_tensor * torch.unsqueeze(seq_merge_weight, dim=-1)).sum(-2) #(bs, sl, dim)


            seq_merge_attri_weight = torch.cat([ori_1, ori_2], dim=-1)
            seq_merge_attri_weight = nn.functional.softmax(seq_merge_attri_weight, dim=-1) #(bs, sl, 4)
            seq_merge_tensor_attri = torch.cat([torch.unsqueeze(ori_embeddings_graph_1, dim=-2), torch.unsqueeze(ori_embeddings_graph_2, dim=-2)], dim=-2) #(bs, sl, 5, dim)
            graph_e_attri = (seq_merge_tensor_attri * torch.unsqueeze(seq_merge_attri_weight, dim=-1)).sum(-2) #(bs, sl, dim) 

            candidate_embeddings_gate = torch.sigmoid(self.W_graph_para_gate_1(graph_e_behavior) + self.W_graph_para_gate_2(graph_e_attri))
            valid_negative_labels_emb = graph_e_behavior * candidate_embeddings_gate + (1. - candidate_embeddings_gate) * graph_e_attri
        # valid_negative_labels_emb_graph_2 = self.item_rep_graph_view[valid_negative_labels, :]  # M x H
        # valid_negative_labels_gate = torch.sigmoid(self.W_graph_para_1_1(valid_negative_labels_emb_buy) + self.W_graph_para_2_1(valid_negative_labels_emb_graph_attribute))
        # valid_negative_labels_emb = valid_negative_labels_gate * valid_negative_labels_emb_buy + (1. - valid_negative_labels_gate) * valid_negative_labels_emb_graph_attribute
        # valid_negative_labels_emb = torch.cat((valid_negative_labels_emb_buy, valid_negative_labels_emb_graph_attribute), dim=-1)

        # valid_negative_labels_emb = valid_negative_labels_emb_buy + valid_negative_labels_emb_review
        # valid_negative_labels_emb = valid_negative_labels_emb_buy
        # cate_emb = self.item_rep_graph_attribute[neg_cates, :]
        # brand_emb = self.item_rep_graph_attribute[neg_brands, :]
        # price_emb = self.item_rep_graph_attribute[neg_prices, :]

        # cate_emb = self.item_rep_graph_attribute(neg_cates)
        # brand_emb = self.item_rep_graph_attribute(neg_brands)
        # price_emb = self.item_rep_graph_attribute(neg_prices)

        # valid_negative_labels_emb = torch.cat((valid_negative_labels_emb, cate_emb, brand_emb, price_emb), dim=-1) #(bs, sl, 4*dim)

        # pdb.set_trace()

        valid_labels_prob = self.sigmoid((valid_logits * valid_labels_emb).sum(-1))  # M
        valid_negative_labels_prob = self.sigmoid((valid_logits * valid_negative_labels_emb).sum(-1))  # M

        loss = -torch.log(valid_labels_prob + 1e-24) - torch.log((1-valid_negative_labels_prob) + 1e-24)
        loss = loss.mean()
        loss, loss_cnt = loss.unsqueeze(0), loss_cnt.unsqueeze(0)
        return loss, loss_cnt