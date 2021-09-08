import pdb
from .bert_base import BertBaseModel
from .embeddings import *
from .bodies import BertBody
from .heads import *
import torch.nn.functional as F

import torch.nn as nn


class BertModel(BertBaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
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
        
        self.kl_loss = torch.nn.KLDivLoss(size_average=False)
        self.project_layer = nn.Sequential(
                nn.Linear(self.args.hidden_units, self.args.hidden_units),
                GELU(),
                nn.LayerNorm(self.args.hidden_units),
                nn.Linear(self.args.hidden_units, self.args.hidden_units)
            )
        self.ln = nn.LayerNorm(args.hidden_units)
        self.dropout = nn.Dropout(p=args.dropout)
        self.init_weights()

    @classmethod
    def code(cls):
        return 'bert_argu_input'

    
    def calculate_contrastive_loss_by_prob(self, prob_ori, pro_arg, tempurate=1.0):
        """
        计算两个prob的KL散度;
        prob_ori: (M, V)
        pro_arg: (M, V)
        """
        pro_arg = F.softmax(pro_arg/tempurate, dim=-1)
        prob_ori = F.softmax(prob_ori/tempurate, dim=-1)
        # pdb.set_trace()
        kl_mean_loss = self.kl_loss(pro_arg.log(), prob_ori)
        return kl_mean_loss

    def calculate_contrastive_loss(self, hidden1, hidden2, temperature=1.0):
        """
        当前item与相似item作为正样本对, 其余2N-2作为负样本对;
        """
        LARGE_NUM = 1e9
        # pdb.set_trace()
        batch_size, sl = hidden1.shape
        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = torch.arange(0, batch_size).to(device=hidden1.device)
        masks = torch.nn.functional.one_hot(torch.arange(0, batch_size), num_classes=batch_size).to(device=hidden1.device, dtype=torch.float)

        logits_aa = torch.matmul(hidden1, hidden1_large.transpose(0, 1)) / temperature  # shape (bsz, bsz)
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.transpose(0, 1)) / temperature  # shape (bsz, bsz)
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.transpose(0, 1)) / temperature  # shape (bsz, bsz)
        logits_ba = torch.matmul(hidden2, hidden1_large.transpose(0, 1)) / temperature  # shape (bsz, bsz)

        loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
        loss = loss_a + loss_b

        return loss

    # def calculate_userRep_and_itemRep(self, hidden1, hidden2, item_rep):
    #     """
    #     增强样本与item rep之间的距离相近;
    #     """

    #     return
    

    def get_logits(self, d, keyword='tokens', add_dropout=False, contrastive_flag=False):
        x = d[keyword]
        # pdb.set_trace()
        attn_mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        e = self.token_embedding(d) + self.positional_embedding(d)
        if add_dropout == True:
            e = nn.functional.dropout(e, self.args.dropout_rate)
        e = self.ln(e)
        e = self.dropout(e)  # B x T x H

        info = {} if self.output_info else None
        b = self.body(e, attn_mask, info)  # B x T x H

        # 添加project layer
        if contrastive_flag:
            # pdb.set_trace()
            b = self.project_layer(b)
        return b, info

    def get_scores(self, d, logits):
        # logits : B x H or M x H
        if self.training:  # logits : M x H, returns M x V
            h = self.head(logits)  # M x V
        else:  # logits : B x H,  returns B x C
            candidates = d['candidates']  # B x C
            h = self.head(logits, candidates)  # B x C
        return h