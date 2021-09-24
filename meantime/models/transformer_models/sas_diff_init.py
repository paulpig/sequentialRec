from ..base import BaseModel
from .embeddings import *
# from .bodies import ExactSasBody
# from .embeddings import *
# from .bodies import ExactSasBody
from .bodies import SasBody as ExactSasBody
from .heads import *
from meantime.models.transformer_models.bodies.transformers.transformer_meantime import MixedAttention
from meantime.models.transformer_models.bodies.transformers.transformer_relative import RelAttention
from meantime.models.transformer_models.heads import BertDiscriminatorHead, BertDotProductPredictionHead

import torch
import torch.nn as nn


class SASModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.output_info = args.output_info
        self.token_embedding = TokenEmbedding(args)
        self.positional_embedding = PositionalEmbedding(args)
        self.body = ExactSasBody(args)
        self.ln = nn.LayerNorm(args.hidden_units)
        self.dropout = nn.Dropout(p=args.dropout)
        self.init_weights()

        self.sigmoid = nn.Sigmoid()
        self.vocab_size = args.num_items + 1

    @classmethod
    def code(cls):
        return 'sas_init'

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
            candidate_embeddings = self.token_embedding.emb(d['candidates'])  # B x C x H
            scores = (last_logits * candidate_embeddings).sum(-1)  # B x C
            ret['scores'] = scores
        return ret

    
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            # module.weight.data.normal_(mean=0.0, std=self.model_init_range) #修改model_init_range
            # nn.init.xavier_uniform(module.weight.data) #修改
            # nn.init.xavier_normal_(module.weight.data) #修改
            # nn.init.kaiming_uniform_(module.weight.data)
            # nn.init.kaiming_normal_(module.weight.data)
            # torch.nn.init.constant_(module.weight.data, 0.5)
            torch.nn.init.orthogonal_(module.weight.data)
            # torch.nn.init.uniform_(module.weight.data)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model_init_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, MixedAttention):
            for param in [module.rel_position_bias]:
                param.data.normal_(mean=0.0, std=self.model_init_range)
        elif isinstance(module, RelAttention):
            for param in [module.r_bias]:
                param.data.normal_(mean=0.0, std=self.model_init_range)
        elif isinstance(module, BertDiscriminatorHead):
            for param in [module.w]:
                param.data.normal_(mean=0.0, std=self.model_init_range)
        elif isinstance(module, BertDotProductPredictionHead):
            for param in [module.bias]:
                param.data.zero_()
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_logits(self, d):
        x = d['tokens']
        attn_mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        attn_mask.tril_()  # causal attention for sasrec
        e = self.token_embedding(d) + self.positional_embedding(d)
        e = self.dropout(e)
        info = None
        if self.output_info:
            info = {}
        b = self.body(e, attn_mask, info)  # B x T x H
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

        valid_labels_emb = self.token_embedding.emb(valid_labels)  # M x H
        valid_negative_labels_emb = self.token_embedding.emb(valid_negative_labels)  # M x H

        valid_labels_prob = self.sigmoid((valid_logits * valid_labels_emb).sum(-1))  # M
        valid_negative_labels_prob = self.sigmoid((valid_logits * valid_negative_labels_emb).sum(-1))  # M

        loss = -torch.log(valid_labels_prob + 1e-24) - torch.log((1-valid_negative_labels_prob) + 1e-24)
        loss = loss.mean()
        loss, loss_cnt = loss.unsqueeze(0), loss_cnt.unsqueeze(0)
        return loss, loss_cnt
