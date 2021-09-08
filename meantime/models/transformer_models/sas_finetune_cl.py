from ..base import BaseModel
from .embeddings import *
# from .bodies import ExactSasBody
from .bodies import SasBody
from .heads import *
from meantime.models.transformer_models.utils import PoolingLayer

import torch
import torch.nn as nn
import pdb

class SASModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.output_info = args.output_info
        self.token_embedding = TokenEmbedding(args)
        self.positional_embedding = PositionalEmbedding(args)
        self.body = SasBody(args)
        self.ln = nn.LayerNorm(args.hidden_units)
        self.dropout = nn.Dropout(p=args.dropout)
        self.pooling_type = args.pooling_type
        self.pooling_layer = PoolingLayer(self.pooling_type) #为什么save model时, 没有将poolingLayer保存？
        self.pooling_layer_arg = PoolingLayer(self.pooling_type)

        # self.head是为加载bert模型参数, 在此处不使用;
        if args.headtype == 'dot':
            self.head = BertDotProductPredictionHead(args, self.token_embedding.emb)
        elif args.headtype == 'linear':
            self.head = BertLinearPredictionHead(args)
        else:
            raise ValueError
        self.project_layer = nn.Sequential(
                nn.Linear(self.args.hidden_units, self.args.hidden_units),
                GELU(),
                nn.LayerNorm(self.args.hidden_units),
                nn.Linear(self.args.hidden_units, self.args.hidden_units)
            )
        self.init_weights()

        self.sigmoid = nn.Sigmoid()
        self.vocab_size = args.num_items + 1

    @classmethod
    def code(cls):
        return 'sas_finetune_cl'


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
        loss = (loss_a + loss_b) / 2.0
        # pdb.set_trace()
        return loss

    def forward(self, d):
        logits, info = self.get_logits(d)

        #add cl contrastive loss
        if self.args.constrastive_input_flag and self.training:
            # pdb.set_trace()
            # logits_ori, info_ori = self.get_logits(d, keyword="tokens_ori", contrastive_flag=True)
            logits_arg, info_arg = self.get_logits(d, keyword='tokens_pair1', contrastive_flag=True) #tokens_pair是经过变换的数据;
            logits_arg2, info_arg2 = self.get_logits(d, keyword='tokens_pair2', contrastive_flag=True) #tokens_pair是经过变换的数据;
            #在原始模型的基础上添加dropout
            # logits_arg, info_arg = self.get_logits(d, keyword='tokens', add_dropout=True)
            # d['sub_tokens'], d['shuffle_subseq'], d['sub_tokens'] 
            ret = {'logits':logits, 'info':info, 'logit_arg': logits_arg, 'info_arg': info_arg}
        else:
            ret = {'logits':logits, 'info':info}
        if self.training:
            labels = d['labels']
            negative_labels = d['negative_labels']
            loss, loss_cnt = self.get_loss(logits, labels, negative_labels)

            #对比损失函数, pooling层 + 对比学习, 实现pooling_layer: 考虑mean pooling, max pooling, last items;
            if self.pooling_type == 'last-pooling':
                sequence_length = (d['tokens'] > 0).sum(dim=-1)  #(bs)
                self.pooling_layer.sequence_len = sequence_length
            
            if self.pooling_type in ['max-pooling', 'mean-pooling']:
                seq_mask = (d['tokens'] > 0).float()
                self.pooling_layer.seq_mask = seq_mask
                self.pooling_layer_arg.seq_mask = (d['tokens_pair'] > 0).float()
            # pdb.set_trace()
            if self.args.constrastive_flag:
                logit_pooling_arg1, logit_pooling_arg2 = self.pooling_layer(logits_arg), self.pooling_layer(logits_arg2)
                # logit_pooling, logit_pooling_arg = self.pooling_layer(logits_ori), self.pooling_layer_arg(logits_arg)
                # logit_pooling, logit_pooling_arg = self.pooling_layer(logits_ori), self.pooling_layer(logits)
                # loss_contrastive = self.calculate_contrastive_loss(logit_pooling_arg1, logit_pooling_arg2, temperature=0.5)
                loss_contrastive = self.calculate_contrastive_loss(logit_pooling_arg1, logit_pooling_arg2, temperature=1.0)

                #只计算在mask输出的prob, 不考虑计算的loss;
                # loss_2, _, prob_arg = self.get_loss(d, logits_arg, labels)
                # pdb.set_trace()
                # loss_contrastive = self.calculate_contrastive_loss_by_prob(prob_ori, prob_arg, tempurate=0.5)
                # pdb.set_trace()
                # ret['loss_cons'] = loss_contrastive
                # pdb.set_trace()
                ret['loss'] = loss + self.args.constrastive_weight * loss_contrastive
                # pdb.set_trace()
                # ret['loss'] = loss_contrastive
                # ret['loss'] = loss
            elif self.args.cl_binary_flag:
                # logit_pooling, logit_pooling_arg = self.pooling_layer(logits_ori), self.pooling_layer_arg(logits_arg)
                # loss_contrastive = self.calculate_binary_loss(logit_pooling, logit_pooling_arg)
                # ret['loss'] = loss + self.args.cl_binary_weight*loss_contrastive
                # pdb.set_trace()
                pass
            else:
                ret['loss'] = loss
            ret['loss_cnt'] = loss_cnt
        else:
            # get scores (B x C) for validation
            last_logits = logits[:, -1, :].unsqueeze(1)  # B x 1 x H
            candidate_embeddings = self.token_embedding.emb(d['candidates'])  # B x C x H
            scores = (last_logits * candidate_embeddings).sum(-1)  # B x C
            ret['scores'] = scores
        return ret

    def get_logits(self, d, keyword='tokens', contrastive_flag=False):
        x = d[keyword]
        attn_mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        attn_mask.tril_()  # causal attention for sasrec
        # pdb.set_trace()
        if contrastive_flag == False:
            e = self.token_embedding(d, keyword=keyword) + self.positional_embedding(d, keyword=keyword) #positional_embedding是否构建错误哎;
        else:
            e = self.token_embedding(d, keyword=keyword)
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
