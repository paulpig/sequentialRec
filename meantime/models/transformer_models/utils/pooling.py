import torch.nn as nn
import torch
import math
import pdb


class PoolingLayer(nn.Module):
    """
    PoolingLayer: convert a 3-dim tensor to a 2-dim tensor;
    """ 

    def __init__(self, pooling_type='last-pooling', sequence_len=None, seq_mask=None):
        super().__init__()
        self.pooling_type = pooling_type
        self.sequence_len = sequence_len
        self.seq_mask = seq_mask #(bs, sl)
        self.labels = None

    def set_pooling_type(self, pooling_type):
        self.pooling_type = pooling_type
        return
    
    def set_sequence_len(self, sequence_length):
        self.sequence_len = sequence_length
        return

    def forward(self, x):
        """
        input: 
            x: (bs, sl, dim)

        return 
            result: (bs, dim)
        """
        if self.pooling_type == 'max-pooling':
            seq_mask_expand = self.seq_mask.unsqueeze(-1).expand(x.size()).float()
            x[seq_mask_expand == 0.] = -1e9 #(bs, sl, dim)
            return torch.max(x, 1)[0]
        if self.pooling_type == 'mean-pooling':
            # seq_mask_expand = self.seq_mask.unsqueeze(-1).expand(x.size()).float() #(bs, sl, dim)
            seq_mask_expand = self.seq_mask.unsqueeze(-1).repeat(1, 1, x.size()[-1]).float() #(bs, sl, dim)
            x_sum_embeddings = torch.sum(x * seq_mask_expand, 1) #(bs, dim)
            x_len = torch.sum(self.seq_mask, -1).unsqueeze(-1) #(bs)
            # pdb.set_trace()
            return x_sum_embeddings/x_len
        if self.pooling_type == 'last-pooling':
            assert self.sequence_len != None
            # pdb.set_trace()
            # return torch.index_select(x, 1, self.sequence_len -1) #(bs, dim)
            # return x[torch.arange(x.size(0)), self.sequence_len-1] #(bs, dim)
            # indices = torch.unsqueeze(self.sequence_len-1, 1)
            # indices = torch.unsqueeze(indices, 2)
            # indices = torch.repeat_interleave(indices, x.size(-1), dim=2) #(bs, 1, dim)
            # return torch.gather(x, 1, indices).squeeze() #(bs, dim)
            #由于是左padding, 因此只选取最后一个item即可;
            return x[:, -1, :]
        
        if self.pooling_type == 'item_level_mean_pooling':
            assert self.labels != None #(bs, sl)
            labels_mask = self.labels.unsqueeze(-1).float() #(bs, sl, 1)
            x_sum_embeddings = torch.sum(labels_mask * x, 1) #(bs, dim)
            return x_sum_embeddings / torch.sum(self.labels + 0.1, -1).unsqueeze(-1)

        
        if self.pooling_type == 'cls_pooling':
            assert self.sequence_len != None #(bs)
            return x[torch.arange(x.size(0)), self.sequence_len] #(bs, dim)
        return None