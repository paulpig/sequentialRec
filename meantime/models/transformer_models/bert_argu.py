from .bert_base import BertBaseModel
from .embeddings import *
from .bodies import BertBody
from .heads import *
import torch
import torch.nn as nn
import pdb


class BertModelArg(BertBaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.output_info = args.output_info
        self.token_embedding = TokenEmbedding(args)
        # self.positional_embedding = PositionalEmbedding(args)
        self.position_embeddings = nn.Embedding(args.max_len, args.hidden_units)
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
        return 'bert_argu'
    
    def _cutoff_embeddings(self, embedding_output, attention_mask, direction, rate):
        """
        embedding_output: (bs, sl, dim)
        attention_mask: (bs, sl)
        """
        bsz, seq_len, emb_size = embedding_output.shape
        cutoff_embeddings = []
        for bsz_id in range(bsz):
            sample_embedding = embedding_output[bsz_id]
            sample_mask = attention_mask[bsz_id]
            # pdb.set_trace()
            if direction == "row":
                num_dimensions = sample_mask.sum().int().item()  # number of tokens
                dim_index = 0
            elif direction == "column":
                num_dimensions = emb_size  # number of features
                dim_index = 1
            elif direction == "random":
                num_dimensions = sample_mask.sum().int().item() * emb_size
                dim_index = 0
            else:
                raise ValueError(f"direction should be either row or column, but got {direction}")
            num_cutoff_indexes = int(num_dimensions * rate)
            if num_cutoff_indexes < 0 or num_cutoff_indexes > num_dimensions:
                raise ValueError(f"number of cutoff dimensions should be in (0, {num_dimensions}), but got {num_cutoff_indexes}")
            indexes = list(range(num_dimensions))
            import random
            random.shuffle(indexes)
            cutoff_indexes = indexes[:num_cutoff_indexes]
            if direction == "random":
                sample_embedding = sample_embedding.reshape(-1)
            cutoff_embedding = torch.index_fill(sample_embedding, dim_index, torch.tensor(cutoff_indexes, dtype=torch.long).to(device=embedding_output.device), 0.0)
            if direction == "random":
                cutoff_embedding = cutoff_embedding.reshape(seq_len, emb_size)
            cutoff_embeddings.append(cutoff_embedding.unsqueeze(0))
        cutoff_embeddings = torch.cat(cutoff_embeddings, 0)
        assert cutoff_embeddings.shape == embedding_output.shape, (cutoff_embeddings.shape, embedding_output.shape)
        return cutoff_embeddings

    def argument_embeddings(self, item_embeddings, item_mask):
        """
        item_embeddings: (bs, sl, dim); type: long
        item_mask: (bs, sl); type:boolean
        cut_off.direction is in ['row', 'column', 'random']
        cut_off.rate is a float;
        """
        bsz, seq_len, emb_size = item_embeddings.shape
        if self.exists_flag("cut_off"):
            rate = self.get_flag("cut_off.rate")
            direction = self.get_flag("cut_off.direction")
            assert direction in ("row", "column", "random")
            assert isinstance(rate, float) and 0.0 < rate < 1.0
            embedding_after_cutoff = self._cutoff_embeddings(item_embeddings, item_mask, direction, rate)
            return embedding_after_cutoff
        elif self.exists_flag("shuffle_embeddings"):
            self.unset_flag("shuffle_embeddings")
            shuffled_embeddings = []
            for bsz_id in range(bsz):
                sample_embedding = item_embeddings[bsz_id] #(sl, dim)
                sample_mask = item_embeddings[bsz_id] #(sl)
                num_tokens = sample_mask.sum().int().item()
                indexes = list(range(num_tokens))
                import random
                random.shuffle(indexes)
                rest_indexes = list(range(num_tokens, seq_len))
                total_indexes = indexes + rest_indexes
                shuffled_embeddings.append(torch.index_select(sample_embedding, 0, torch.tensor(total_indexes).to(device=item_embeddings.device)).unsqueeze(0))
            return torch.cat(shuffled_embeddings, 0)
        else:
            return item_embeddings

    def _replace_position_ids(self, input_ids, attention_mask=None,  position_ids=None):
        bsz, seq_len = input_ids.shape
        if position_ids is None:
                # position_ids = torch.arange(512).expand((bsz, -1))[:, :seq_len].to(device=input_ids.device)
                position_ids = torch.arange(seq_len).expand((bsz, -1)).to(device=input_ids.device) #(bs, seq_len) 
        if self.exists_flag("shuffle_position"):
            self.unset_flag("shuffle_position")
            # shuffle position_ids
            shuffled_pid = []
            for bsz_id in range(bsz):
                sample_pid = position_ids[bsz_id]
                sample_mask = attention_mask[bsz_id]
                num_tokens = sample_mask.sum().int().item()
                indexes = list(range(num_tokens))
                import random
                random.shuffle(indexes)
                rest_indexes = list(range(num_tokens, seq_len)) #将有效部分的position ids给shuffle;s
                total_indexes = indexes + rest_indexes
                shuffled_pid.append(torch.index_select(sample_pid, 0, torch.tensor(total_indexes).to(device=input_ids.device)).unsqueeze(0))
            return torch.cat(shuffled_pid, 0)
        else:
            return position_ids 
        
    def get_logits(self, d):
        """
        每次调用该方法时, 需要手动通过set_flag来设置数据增强的标记符号;
        input:
        d is a dict; 包含'tokens'和'candidates'
        return: 
        (bs, sl, dim)
        """
        x = d['tokens'] #(bs, sl)
        attn_mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1) #(bs, 1, sl, sl); 
        self.item_embedding = self.token_embedding(d)
        # self.position_embedding = self.positional_embedding(d)

        #argument embedding 
        item_attn_mask = (x > 0) #(bs, sl)
        self.argument_item_embedding = self.argument_embeddings(self.item_embedding, item_attn_mask)
        # pdb.set_trace()
        #shuffle position embedding 
        shuffle_position_ids = self._replace_position_ids(x, attn_mask) #()
        # pdb.set_trace()
        self.shuffle_position_embedding = self.position_embeddings(shuffle_position_ids)

        e = self.argument_item_embedding + self.shuffle_position_embedding
        e = self.ln(e)
        e = self.dropout(e)  # B x T x H

        info = {} if self.output_info else None
        b = self.body(e, attn_mask, info)  # B x T x H
        return b, info


    def calculate_contrastive_loss(self, hidden1, hidden2, temperature=1.0):
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
    

    def get_scores(self, d, logits):
        """
        如果是CE方法,则不用到D;
        如果是logit方法, 则需要D中的候选items;
        d就是候选实体
        """
        # logits : B x H or M x H
        if self.training:  # logits : M x H, returns M x V
            h = self.head(logits)  # M x V
        else:  # logits : B x H,  returns B x C
            candidates = d['candidates']  # B x C
            h = self.head(logits, candidates)  # B x C, C是候选的items数量;
        return h
