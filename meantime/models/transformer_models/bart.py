import pdb
from .bart_base import BartBaseModel
from .embeddings import *
from .bodies import BertBody, SasBody
from .heads import *

import torch.nn as nn


class BartModel(BartBaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.output_info = args.output_info
        self.token_embedding = TokenEmbedding(args)
        self.positional_embedding = PositionalEmbedding(args)
        self.body = BertBody(args)

        #参考论文<Future Data Helps Training: Modeling Future Contexts for Session-based Recommendation>采用不同的embedding layer;
        self.token_embedding_decoder = TokenEmbedding(args)
        self.positional_embedding_decoder = PositionalEmbedding(args)
        self.body_decoder = SasBody(args)

        self.W1 = nn.Linear(args.hidden_units, args.hidden_units)
        self.W2 = nn.Linear(args.hidden_units, args.hidden_units)

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
        self.ln = nn.LayerNorm(args.hidden_units)
        self.dropout = nn.Dropout(p=args.dropout)

        self.ln_decoder = nn.LayerNorm(args.hidden_units)
        self.dropout_decoder = nn.Dropout(p=args.dropout)
        self.init_weights()

    @classmethod
    def code(cls):
        return 'bart'
    

    def get_logits(self, d, keyword='tokens', add_dropout=False, is_bidirection=True, encoder_hidden=None):
        """
        encoder_hidden: 在decoder中需要输入来预测mask; (bs, sl, hidden)
        """
        x = d[keyword]
        # pdb.set_trace()
        attn_mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        if is_bidirection == False:
            attn_mask.tril_()  # causal attention for sasrec
            # e = self.token_embedding_decoder(d, keyword='tokens') + self.positional_embedding_decoder(d, keyword='tokens')
            e = self.token_embedding(d, keyword='tokens') + self.positional_embedding_decoder(d, keyword='tokens')
            e = self.ln_decoder(e)
            e = self.dropout_decoder(e)  # B x T x H
        else:
            e = self.token_embedding(d, keyword='tokens_pair') + self.positional_embedding(d, keyword='tokens_pair')
            # if add_dropout == True:
            # e = nn.functional.dropout(e, self.args.dropout)
            e = self.ln(e)
            e = self.dropout(e)  # B x T x H
        
        if add_dropout:
            e = nn.functional.dropout(e, self.args.dropout)

        info = {} if self.output_info else None
        if is_bidirection:
            b = self.body(e, attn_mask, info)  # B x T x H
        else:
            # emb_ori = e
            # e = e + encoder_hidden
            # e = emb_ori + self.W2(nn.functional.relu(self.W1(e)))
            b = self.body_decoder(e, attn_mask, info)
            # b = encoder_hidden + b #再增加decoder的embedding;
        
        # 添加project layer
        return b, info

    def get_scores(self, d, logits):
        # logits : B x H or M x H
        if self.training:  # logits : M x H, returns M x V
            h = self.head(logits)  # M x V
        else:  # logits : B x H,  returns B x C
            candidates = d['candidates']  # B x C
            h = self.head(logits, candidates)  # B x C
        return h