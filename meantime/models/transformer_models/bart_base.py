from meantime.models.base import BaseModel

import torch.nn as nn
from meantime.models.transformer_models.utils import PoolingLayer
from abc import *
from .heads import *
import pdb


class BartBaseModel(BaseModel, metaclass=ABCMeta):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.ce = nn.CrossEntropyLoss()
        self.pooling_type = args.pooling_type
        self.pooling_layer = PoolingLayer(self.pooling_type) #为什么save model时, 没有将poolingLayer保存？
        self.pooling_layer_arg = PoolingLayer(self.pooling_type)

        self.W1 = nn.Linear(args.hidden_units, args.hidden_units)
        self.W2 = nn.Linear(args.hidden_units, args.hidden_units)

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

    # def calculate_contrastive_loss(self, hidden1, hidden2, temperature=1.0):
    #     pass

    # def calculate_binary_loss(self, hidden1, hidden2):
    #     pass
    
    def forward(self, d):
        """
        d: dict, contains the 'tokens' key; 
        """
        # pdb.set_trace()
        # 数据增强的方法;
        # pdb.set
        logits_encoder, info_encoder = self.get_logits(d, 'tokens_pair') #输入增强样本; encoder的输入;
        # logits_encoder, info_encoder = self.get_logits(d, keyword='tokens_pair', is_bidirection=True) #输入增强样本; encoder的输入;
        
        # logits_encoder_arg, _ = self.get_logits(d, 'tokens_pair', add_dropout=True)
        # logits_decoder, info_decoder = self.get_logits(d, keyword='tokens', is_bidirection=False, encoder_hidden=logits_encoder) # n-1, 最前面的token是special token;

        # activate_tensor = torch.sigmoid(self.W1(logits_encoder) + self.W2(logits_decoder))
        # merge_logits = activate_tensor * logits_encoder + (1.0 - activate_tensor) * logits_decoder
        merge_logits = self.W1(logits_encoder) + self.W2(logits_decoder)
        # merge_logits = activate_tensor * logits_decoder
        merge_logits = nn.functional.dropout(merge_logits, self.args.dropout)

        # merge_logits = logits_decoder #在上层再一层单向的transformer;
        # merge_logits = logits_encoder #只考虑bert4rec模型;

        ret = {'logits_encoder':logits_encoder, 'info_encoder':info_encoder, 'logits_decoder': logits_decoder, 'info_decoder': info_decoder}
        if self.training: #训练时则是在整个词表空间通过交叉熵计算损失函数;
            labels = d['labels']
            #merge encoder rep and decoder rep, gate;
            # merge_logits = logits_encoder + logits_decoder
            # activate_tensor = torch.sigmoid(self.W1(logits_encoder) + self.W2(logits_decoder))
            # merge_logits = activate_tensor * logits_encoder + (1.0 - activate_tensor) * logits_decoder
            # pdb.set_trace()
            loss, loss_cnt = self.get_loss(merge_logits, labels, d) #修改为似然损失函数;
            # loss, loss_cnt = self.get_loss(logits_decoder, labels, d) #修改为似然损失函数;
           
            ret['loss'] = loss
            # ret['loss'] = loss
            ret['loss_cnt'] = loss_cnt
        else:
            # get scores (B x V) for validation, 测试时, 只需要采样100条来计算metric;
            last_logits = merge_logits[:, -1, :]  # B x H
            # last_logits = logits_decoder[:, -1, :]  # B x H
            ret['scores'] = self.get_scores(d, last_logits)  # B x C
        return ret

    @abstractmethod
    def get_logits(self, d):
        pass

    @abstractmethod
    def get_scores(self, d, logits):  # logits : B x H or M x H, returns B x C or M x V
        pass

    def get_loss(self, logits, labels, d):
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
        return loss, loss_cnt

    # def get_loss(self, logits, labels, d):
    #     """
    #     different with sasrec model, using the log function;
    #     logtis: (bs, sl, hidden)
    #     labels: (bs, sl)
    #     """
    #     _logits = logits.view(-1, logits.size(-1))  # BT x H
    #     _labels = labels.view(-1)  # BT

    #     valid = _labels > 0
    #     loss_cnt = valid.sum()  # = M
    #     valid_index = valid.nonzero().squeeze()  # M

    #     valid_logits = _logits[valid_index]  # M x H
    #     valid_scores = self.get_scores(d, valid_logits)  # M x V, V是词表的数量;
    #     valid_labels = _labels[valid_index]  # M

    #     loss = self.ce(valid_scores, valid_labels)
    #     loss, loss_cnt = loss.unsqueeze(0), loss_cnt.unsqueeze(0)
    #     return loss, loss_cnt