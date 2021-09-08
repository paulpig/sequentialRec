from .bert_base import BertBaseModel
from .embeddings import *
from .bodies import BertBody
from .heads import *

import torch.nn as nn


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

        #add project layer
        self.project_layer = nn.Linear(args.hidden_units, 1, bias=True)
        self.dropout = nn.Dropout(p=args.dropout)
        self.init_weights()

    @classmethod
    def code(cls):
        return 'bert_binary'

    def calculate_binary_loss(self, hidden1, hidden2):
        """
        hidden1: (bs, sl, dim)
        hidden2: (bs, sl, dim)
        """
        positive_logits = self.project_layer(hidden1) #(bs, 1)
        negative_logits = self.project_layer(hidden2) #(bs, 1)
        loss_binary = - torch.mean(torch.log(nn.functional.sigmoid(positive_logits) + 1e-6) + torch.log(1.0 - nn.functional.sigmoid(negative_logits) + 1e-6))
        return loss_binary
    

    def get_logits(self, d, keyword='tokens', add_dropout=False, contrastive_flag=False):
        x = d['tokens']
        attn_mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        e = self.token_embedding(d) + self.positional_embedding(d)
        e = self.ln(e)
        e = self.dropout(e)  # B x T x H

        info = {} if self.output_info else None
        b = self.body(e, attn_mask, info)  # B x T x H
        return b, info

    def get_scores(self, d, logits):
        # logits : B x H or M x H
        if self.training:  # logits : M x H, returns M x V
            h = self.head(logits)  # M x V
        else:  # logits : B x H,  returns B x C
            candidates = d['candidates']  # B x C
            h = self.head(logits, candidates)  # B x C
        return h