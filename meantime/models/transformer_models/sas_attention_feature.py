from os import PRIO_PROCESS
from .bert_base import BaseModel
from .embeddings import *
# from .bodies import MeantimeBody
from .bodies import SasFeatureBody
from .bodies import SasBody
from .heads import *

import torch
import torch.nn as nn
import pdb


class SASFeatureModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        hidden = args.hidden_units
        self.output_info = args.output_info
        # absolute_kernel_types = args.absolute_kernel_types
        # relative_kernel_types = args.relative_kernel_types
        ##### Footers
        # Token Embeddings
        self.token_embedding = TokenEmbedding(args)
        self.positional_embedding = PositionalEmbedding(args)
        self.ln = nn.LayerNorm(args.hidden_units)

        self.item_rep_graph_attribute = nn.Embedding(args.num_attributes + 3, args.feature_dim, padding_idx=0)
        self.item_rep_graph_attribute_rel = nn.Embedding(args.num_attributes + 3, args.feature_dim, padding_idx=0)
        self.concat_layer = nn.Linear(args.hidden_units + 3*args.feature_dim, args.hidden_units)

        # Absolute Embeddings
        # self.absolute_kernel_embeddings_list = nn.ModuleList()
        # if absolute_kernel_types is not None and len(absolute_kernel_types) > 0:
        #     for kernel_type in absolute_kernel_types.split('-'):
        #         if kernel_type == 'p':  # position
        #             emb = PositionalEmbedding(args)
        #         elif kernel_type == 'd':  # day
        #             emb = DayEmbedding(args)
        #         elif kernel_type == 'c':  # constant
        #             emb = ConstantEmbedding(args)
        #         else:
        #             raise ValueError
        #         self.absolute_kernel_embeddings_list.append(emb)
        # # Relative Embeddings
        # self.relative_kernel_embeddings_list = nn.ModuleList()
        # if relative_kernel_types is not None and len(relative_kernel_types) > 0:
        #     for kernel_type in relative_kernel_types.split('-'):
        #         if kernel_type == 's':  # time difference
        #             emb = SinusoidTimeDiffEmbedding(args)
        #         elif kernel_type == 'e':
        #             emb = ExponentialTimeDiffEmbedding(args)
        #         elif kernel_type == 'l':
        #             emb = Log1pTimeDiffEmbedding(args)
        #         else:
        #             raise ValueError
        #         self.relative_kernel_embeddings_list.append(emb)
        # Lengths
        # self.La = len(self.absolute_kernel_embeddings_list)
        # self.Lr = len(self.relative_kernel_embeddings_list)
        self.La = 1
        self.Lr = 3
        # self.L = self.La + self.Lr
        self.L = self.Lr
        # Sanity check
        # pdb.set_trace()
        assert hidden % self.L == 0, 'multi-head has to be possible'
        # assert self.La == self.Lr
        # assert len(self.absolute_kernel_embeddings_list) > 0 or len(self.relative_kernel_embeddings_list) > 0
        ##### BODY
        # self.body = SasFeatureBody(args, self.La, self.Lr)
        self.body = SasBody(args) #hidden_size = hidden_size * 2
        ##### Heads
        # self.bert_head = BertDotProductPredictionHead(args)
        if args.headtype == 'dot':
            self.head = BertDotProductPredictionHead(args, self.token_embedding.emb)
        elif args.headtype == 'linear':
            self.head = BertLinearPredictionHead(args)
        else:
            raise ValueError
        ##### dropout
        self.dropout = nn.Dropout(p=args.dropout)
        ##### Weight Initialization
        self.init_weights()
        ##### MISC
        # self.ce = nn.CrossEntropyLoss()
        # self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.vocab_size = args.num_items + 1

    @classmethod
    def code(cls):
        return 'sasrec_attention_feature'

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
            x_cates = d['candidate_cates'].reshape(-1)
            x_brands = d['candidate_brands'].reshape(-1)
            x_prices = d['candidate_prices'].reshape(-1)

            x_unsqueeenze = x.reshape(-1) #(bs*C)

            candidate_embeddings = self.token_embedding.emb(x_unsqueeenze).reshape(x.size(0), x.size(1), -1)
            
            scores = (last_logits * candidate_embeddings).sum(-1)  # B x C
            ret['scores'] = scores
        return ret



    def get_logits(self, d):
        x = d['tokens']

        cates = d['seq_cates']
        brands = d['seq_brands']
        prices = d['seq_prices']

        attn_mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)  # B x 1 x T x T
        # token_embeddings = self.dropout(self.token_embedding(d)) # B x T x H
        token_embeddings = self.token_embedding(d)


        # token_embeddings = token_embeddings.unsqueeze(0).expand(self.L, -1, -1, -1)  # L x B x T x H
        # token_embeddings = token_embeddings.chunk(self.L, dim=0)  # L of [1 x B x T x H]
        # token_embeddings = [x.squeeze(0) for x in token_embeddings]  # L of [B x T x H]

        cate_emb = self.item_rep_graph_attribute(cates).reshape(x.size(0), x.size(1), -1) #(bs, sl, dim)
        brand_emb = self.item_rep_graph_attribute(brands).reshape(x.size(0), x.size(1), -1)
        price_emb = self.item_rep_graph_attribute(prices).reshape(x.size(0), x.size(1), -1)

        feature_abs = torch.cat((token_embeddings, cate_emb, brand_emb, price_emb), dim=-1) #(bs, sl, 4*dim)
        feature_abs = self.concat_layer(feature_abs)
        token_embeddings = feature_abs + self.positional_embedding(d)

        e = token_embeddings
        # token_embeddings = self.ln(token_embeddings)

        # absolute_kernel_embeddings = [self.ln(cate_emb), self.ln(brand_emb), self.ln(price_emb)]
        absolute_kernel_embeddings = [token_embeddings]
        # absolute_kernel_embeddings = [self.dropout(emb(d)) for emb in self.absolute_kernel_embeddings_list]  # La of [B x T x H]
        
        #conduct relative feature
        def genRelFeature(feature):
            """
            feature: (bs, sl)
            """
            feature_unsq_1 = feature.unsqueeze(-1)
            feature_unsq_2 = feature.unsqueeze(1)
            feature_valid_mask = (feature_unsq_1 == feature_unsq_2) & ((feature_unsq_1!=0) & (feature_unsq_2!=0)) #(bs, sl, sl)
            # pdb.set_trace()
            feature_valid = feature_unsq_2 * feature_valid_mask
            return feature_valid, feature_valid_mask
        
        cates_feature_valid, cates_feature_mask = genRelFeature(cates)
        brands_feature_valid, brands_feature_mask = genRelFeature(brands)
        prices_feature_valid, prices_feature_mask = genRelFeature(prices)
        # pdb.set_trace()

        # cate_emb_rel = self.item_rep_graph_attribute_rel(cates_feature_valid).reshape(x.size(0), x.size(1), x.size(1), -1) #(bs, sl, dim)
        # brand_emb_rel = self.item_rep_graph_attribute_rel(brands_feature_valid).reshape(x.size(0), x.size(1), x.size(1), -1)
        # price_emb_rel = self.item_rep_graph_attribute_rel(prices_feature_valid).reshape(x.size(0), x.size(1), x.size(1), -1)

        cate_emb_rel = self.item_rep_graph_attribute_rel(cates_feature_valid).reshape(x.size(0), x.size(1), x.size(1), -1) #(bs, sl, dim)
        brand_emb_rel = self.item_rep_graph_attribute_rel(brands_feature_valid).reshape(x.size(0), x.size(1), x.size(1), -1)
        price_emb_rel = self.item_rep_graph_attribute_rel(prices_feature_valid).reshape(x.size(0), x.size(1), x.size(1), -1)

        # relative_kernel_embeddings = [self.ln(cate_emb_rel), self.ln(brand_emb_rel), self.ln(price_emb_rel)]
        relative_kernel_embeddings = [cate_emb_rel, brand_emb_rel, price_emb_rel]
        relative_kernel_mask = [cates_feature_mask, brands_feature_mask, prices_feature_mask]
        relative_kernel_mask = torch.stack(relative_kernel_mask, dim=1)
        # relative_kernel_embeddings = [self.dropout(emb(d)) for emb in self.relative_kernel_embeddings_list]  # Lr of [B x T x T x H]

        info = {} if self.output_info else None

        # last_hidden = L of [B x T x H]
        # last_hidden = self.body(token_embeddings, attn_mask,
        #                         absolute_kernel_embeddings,
        #                         relative_kernel_embeddings,
        #                         relative_kernel_mask,
        #                         info=info)
        b = self.ln(e) # 不添加layer normal, 效果相差特别多;
        e = self.dropout(b)
        info = None
        if self.output_info:
            info = {}
        last_hidden = self.body(e, attn_mask, info)  # B x T x H
        last_hidden = self.ln(last_hidden)  # original code does this at the end of body
        # last_hidden = torch.cat(last_hidden, dim=-1)  # B x T x LH

        return last_hidden, info

    def get_scores(self, d, logits):
        # logits : B x H or M x H
        if self.training:  # logits : M x H, returns M x V
            h = self.head(logits)  # M x V
        else:  # logits : B x H,  returns B x C
            candidates = d['candidates']  # B x C
            h = self.head(logits, candidates)  # B x C
        return h

    def get_loss(self, logits, labels, negative_labels, d):
        _logits = logits.reshape(-1, logits.size(-1))  # BT x H
        _labels = labels.reshape(-1)  # BT
        _negative_labels = negative_labels.reshape(-1)  # BT


        valid = _labels > 0
        loss_cnt = valid.sum()  # = M
        valid_index = valid.nonzero().squeeze()  # M

        valid_logits = _logits[valid_index]  # M x H
        valid_labels = _labels[valid_index]  # M
        valid_negative_labels = _negative_labels[valid_index]  # M

        valid_labels_emb = self.token_embedding.emb(valid_labels)
 

        valid_negative_labels_emb = self.token_embedding.emb(valid_negative_labels)

        valid_labels_prob = self.sigmoid((valid_logits * valid_labels_emb).sum(-1))  # M
        valid_negative_labels_prob = self.sigmoid((valid_logits * valid_negative_labels_emb).sum(-1))  # M

        loss = -torch.log(valid_labels_prob + 1e-24) - torch.log((1-valid_negative_labels_prob) + 1e-24)
        loss = loss.mean()
        loss, loss_cnt = loss.unsqueeze(0), loss_cnt.unsqueeze(0)
        return loss, loss_cnt