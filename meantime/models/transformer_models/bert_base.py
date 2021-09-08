from meantime.models.base import BaseModel

import torch.nn as nn
from meantime.models.transformer_models.utils import PoolingLayer
from abc import *
import pdb


class BertBaseModel(BaseModel, metaclass=ABCMeta):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.ce = nn.CrossEntropyLoss()
        self.pooling_type = args.pooling_type
        self.pooling_layer = PoolingLayer(self.pooling_type) #为什么save model时, 没有将poolingLayer保存？
        self.pooling_layer_arg = PoolingLayer(self.pooling_type)

    # custom added functions for data augmentation
    def set_flag(self, key: str, value):
        assert f"flag__{key}" not in self.__dict__
        self.__dict__[f"flag__{key}"] = value
    
    def unset_flag(self, key: str):
        assert f"flag__{key}" in self.__dict__
        del self.__dict__[f"flag__{key}"]
    
    def exists_flag(self, key: str):
        return f"flag__{key}" in self.__dict__
    
    def get_flag(self, key: str):
        assert f"flag__{key}" in self.__dict__
        return self.__dict__[f"flag__{key}"]

    def calculate_contrastive_loss(self, hidden1, hidden2, temperature=1.0):
        pass

    def calculate_binary_loss(self, hidden1, hidden2):
        pass

    def calculate_contrastive_loss_by_prob(self, prob_ori, pro_arg):
        pass
    
    def forward(self, d):
        """
        d: dict, contains the 'tokens' key; 
        """
        # pdb.set_trace()
        # 数据增强的方法;
        logits, info = self.get_logits(d) #logits: (bs, sl, dim) #for MLM loss
        if self.args.constrastive_model_flag and self.training:
            logits_ori, info_ori = self.get_logits(d, "tokens_pair")
            self.set_flag('cut_off', True)
            self.set_flag("cut_off.rate", self.args.cutoff_rate)
            self.set_flag("cut_off.direction", self.args.cutoff_type)
            logits_arg, info_arg = self.get_logits(d) #logits: (bs, sl, dim)

            self.unset_flag('cut_off')
            self.unset_flag("cut_off.rate")
            self.unset_flag("cut_off.direction")
            # tmp = d
            # pdb.set_trace()
            ret = {'logits':logits, 'info':info, 'logit_arg': logits_arg, 'info_arg': info_arg}
        elif self.args.constrastive_input_flag and self.training:
            # pdb.set_trace()
            # logits_ori, info_ori = self.get_logits(d, keyword="tokens_ori", contrastive_flag=True)
            # logits_arg, info_arg = self.get_logits(d, keyword='tokens_pair', contrastive_flag=True)
            #在原始模型的基础上添加dropout
            logits_arg, info_arg = self.get_logits(d, keyword='tokens', add_dropout=True)
            # d['sub_tokens'], d['shuffle_subseq'], d['sub_tokens'] 
            ret = {'logits':logits, 'info':info, 'logit_arg': logits_arg, 'info_arg': info_arg}
        else:
            ret = {'logits':logits, 'info':info}
        if self.training: #训练时则是在整个词表空间通过交叉熵计算损失函数;
            labels = d['labels']
            # pdb.set_trace()
            loss, loss_cnt, prob_ori = self.get_loss(d, logits, labels)
            #对比损失函数, pooling层 + 对比学习, 实现pooling_layer: 考虑mean pooling, max pooling, last items;
            if self.pooling_type == 'last-pooling':
                sequence_length = (d['tokens'] > 0).sum(dim=-1)  #(bs)
                self.pooling_layer.sequence_len = sequence_length
            
            if self.pooling_type in ['max-pooling', 'mean-pooling']:
                seq_mask = (d['tokens'] > 0).float()
                self.pooling_layer.seq_mask = seq_mask

                self.pooling_layer_arg.seq_mask = (d['tokens_pair'] > 0).float()

            if self.pooling_type == 'item_level_mean_pooling':
                self.pooling_layer.labels = labels
            
            if self.pooling_type == 'cls_pooling':
                tmp = d
                sequence_length = (d['tokens'] > 0).sum(dim=-1)  #(bs)
                sequence_length = self.args.max_len - sequence_length #for get the 'cls' token;
                self.pooling_layer.sequence_len = sequence_length
            # pdb.set_trace()
            if self.args.constrastive_flag:
                # logit_pooling, logit_pooling_arg = self.pooling_layer(logits), self.pooling_layer(logits_arg)
                # logit_pooling, logit_pooling_arg = self.pooling_layer(logits_ori), self.pooling_layer_arg(logits_arg)
                # logit_pooling, logit_pooling_arg = self.pooling_layer(logits_ori), self.pooling_layer(logits)
                # loss_contrastive = self.calculate_contrastive_loss(logit_pooling, logit_pooling_arg, temperature=0.02)

                #只计算在mask输出的prob, 不考虑计算的loss;
                loss_2, _, prob_arg = self.get_loss(d, logits_arg, labels)
                # pdb.set_trace()
                # loss_contrastive = self.calculate_contrastive_loss_by_prob(prob_ori, prob_arg, tempurate=0.5)
                # pdb.set_trace()
                # ret['loss_cons'] = loss_contrastive
                # pdb.set_trace()
                ret['loss'] = loss + self.args.constrastive_weight*loss_2
                # ret['loss'] = loss_contrastive
                # ret['loss'] = loss
            elif self.args.cl_binary_flag:
                logit_pooling, logit_pooling_arg = self.pooling_layer(logits_ori), self.pooling_layer_arg(logits_arg)
                loss_contrastive = self.calculate_binary_loss(logit_pooling, logit_pooling_arg)
                ret['loss'] = loss + self.args.cl_binary_weight*loss_contrastive
                # pdb.set_trace()
            else:
                ret['loss'] = loss
            # ret['loss'] = loss
            ret['loss_cnt'] = loss_cnt
        else:
            # get scores (B x V) for validation, 测试时, 只需要采样100条来计算metric;
            last_logits = logits[:, -1, :]  # B x H
            ret['scores'] = self.get_scores(d, last_logits)  # B x C
        return ret

    @abstractmethod
    def get_logits(self, d):
        pass

    @abstractmethod
    def get_scores(self, d, logits):  # logits : B x H or M x H, returns B x C or M x V
        pass

    def get_loss(self, d, logits, labels):
        """
        return:
            valid_scores: (M, V)
        """

        # pdb.set_trace()
        _logits = logits.view(-1, logits.size(-1))  # BT x H
        _labels = labels.view(-1)  # BT

        valid = _labels > 0
        loss_cnt = valid.sum()  # = M
        valid_index = valid.nonzero().squeeze()  # M: mask的数量;

        valid_logits = _logits[valid_index]  # M x H
        valid_scores = self.get_scores(d, valid_logits)  # M x V, V是词表的数量;
        valid_labels = _labels[valid_index]  # M

        loss = self.ce(valid_scores, valid_labels)
        loss, loss_cnt = loss.unsqueeze(0), loss_cnt.unsqueeze(0)
        return loss, loss_cnt, valid_scores
