from .simple_graph_base import SimpleGraphBaseModel
from .embeddings import *
from .bodies import BertBody
from .heads import *

import torch.nn as nn
import pdb

class SimpleGraph(SimpleGraphBaseModel):
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
        self.init_weights()

    @classmethod
    def code(cls):
        return 'simple_graph'

    
    def setUserItemRepFromGraph(self, user_rep_graph, item_rep_graph):
        """
        The representations from the LightGCN model; 
        user_rep_graph: (|users|, dim)
        item_rep_graph: (|items|, dim)
        """
        self.user_rep_graph = user_rep_graph
        self.item_rep_graph = item_rep_graph

        # #增加一行是mask token对应的表征
        # mask_emb1 = torch.zeros(1, user_rep_graph.size(-1)).cuda()
        # mask_emb2 = torch.zeros(1, user_rep_graph.size(-1)).cuda()
        # self.user_rep_graph = torch.cat()
        return

    def get_logits(self, d):
        x = d['tokens']
        # attn_mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1) #()
        attn_mask = (x > 0).unsqueeze(-1) #(bs, sl, 1)
        # e = self.token_embedding(d) + self.positional_embedding(d)
        # pdb.set_trace() #由于多特殊的mask token, 导致超出词表数量.
        #采用图模型输出的表征初始化序列推荐模型item lookup table, 初始化的效果增强;
        x_unsqueeenze = x.reshape(-1)
        graph_e = self.item_rep_graph[x_unsqueeenze, :].reshape(x.size(0), x.size(1), -1) #(bs, sl, dim) #use the hidden representation to init the embedding lookup table;
        # e = graph_e + self.positional_embedding(d)

        valid_e = graph_e * attn_mask
        e = self.ln(valid_e)
        e = self.dropout(e)  # B x T x H
        # pdb.set_trace()
        info = {} if self.output_info else None
        
        # b = self.body(e, attn_mask, info)  # B x T x H
        b = torch.sum(e, dim=1)
        return b, info

    def get_scores(self, d, logits):
        # logits : B x H or M x H
        if self.training:  # logits : M x H, returns M x V
            h = self.head(logits)  # M x V
        else:  # logits : B x H,  returns B x C
            candidates = d['candidates']  # B x C
            h = self.head(logits, candidates)  # B x C
        return h