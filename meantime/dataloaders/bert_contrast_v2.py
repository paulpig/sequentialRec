from .base import AbstractDataloader

import torch
import torch.utils.data as data_utils
import pdb
import copy
import numpy as np

class BertContrastDataloader(AbstractDataloader):
    @classmethod
    def code(cls):
        return 'bert_pair_argument'

    def _get_dataset(self, mode):
        if mode == 'train':
            return self._get_train_dataset()
        elif mode == 'val':
            return self._get_eval_dataset('val')
        else:
            return self._get_eval_dataset('test')

    def _get_train_dataset(self):
        train_ranges = self.train_targets
        dataset = BertConstrastTrainDataset(self.args, self.dataset, self.train_negative_samples, self.rng, train_ranges)
        return dataset

    def _get_eval_dataset(self, mode):
        positions = self.validation_targets if mode=='val' else self.test_targets
        dataset = BertConstrastDevDataset(self.args, self.dataset, self.test_negative_samples, positions)
        return dataset

#构建训练数据, 在原始bert模型基础上, 额外构建两条数据, 其中一条数据是低评分数据, 另一条数据是高评分数据; 
class BertConstrastTrainDataset(data_utils.Dataset):
    def __init__(self, args, dataset, negative_samples, rng, train_ranges):
        self.args = args
        self.user2dict = dataset['user2dict']
        self.users = sorted(self.user2dict.keys())
        self.train_window = args.train_window
        self.max_len = args.max_len
        self.mask_prob = args.mask_prob
        self.special_tokens = dataset['special_tokens']
        self.num_users = len(dataset['umap'])
        self.num_items = len(dataset['smap'])
        self.rng = rng
        self.train_ranges = train_ranges

        self.index2user_and_offsets_ori = self.populate_indices()

        #增强数据;
        self.dataset_len = len(self.index2user_and_offsets)
        self.index2user_and_offsets = self.argument_dataset()

        self.output_timestamps = args.dataloader_output_timestamp
        self.output_days = args.dataloader_output_days
        self.output_user = args.dataloader_output_user

        self.negative_samples = negative_samples
        # pdb.set_trace()

    def argument_dataset(self):
        self.index2user_and_offsets = {}
        for index_i in range(self.args.dupe_len):
            for index_j in range(len(self.index2user_and_offsets_ori)):
                self.index2user_and_offsets[index_i*self.dataset_len + index_j] = self.index2user_and_offsets_ori[index_j]
        return self.index2user_and_offsets
    
    def get_rng_state(self):
        return self.rng.getstate()

    def set_rng_state(self, state):
        return self.rng.setstate(state)

    def populate_indices(self):
        """
        对于长序列, 切割成多个序列;
        """
        index2user_and_offsets = {}
        i = 0
        T = self.max_len
        W = self.train_window

        # offset is exclusive
        for user, pos in self.train_ranges:
            if W is None or W == 0:
                offsets = [pos]
            else:
                offsets = list(range(pos, T-1, -W))  # pos ~ T, 选取多个有效的行为序列; 
                if len(offsets) == 0:
                    offsets = [pos]
            for offset in offsets:
                index2user_and_offsets[i] = (user, offset)
                i += 1
        return index2user_and_offsets #一共是8700条数据;

    def __len__(self):
        return len(self.index2user_and_offsets)

    def __getitem__(self, index):
        """
        构建sequence pair;
        """
        user, offset = self.index2user_and_offsets[index]
        seq = self.user2dict[user]['items']
        beg = max(0, offset-self.max_len)
        end = offset  # exclude offset (meant to be)
        seq = seq[beg:end]
        d = {}

        #在最后设置一个标记位: CLS
        # seq.append(self.special_tokens.cls)
        
        # (1) 随机crop items, 不保证连续性; 
        # if self.args.argument_type == 'cutoff':
        if self.args.argument_type == "cutoff":
            tokens_pair = copy.deepcopy(seq)
            # s_len = len(seq) - 1 #最后一个不能mask;
            s_len = len(seq) #最后一个不能mask;
            valid_len = int(self.args.cutoff_rate * s_len)
            import random
            shuffle_index = [index for index in range(s_len)]
            random.shuffle(shuffle_index)
            mask_index = shuffle_index[:valid_len]
            # pdb.set_trace()
            tokens_pair =np.array(tokens_pair)
            tokens_pair[mask_index] = self.special_tokens.mask #用padding替换合理吗? 尝试用MASK替换;
            # pdb.set_trace()
            tokens_pair = tokens_pair.tolist()
            tokens_pair = tokens_pair[-self.max_len:]
            padding_len = self.max_len - len(tokens_pair)
            #开头添加CLS;
            # if padding_len == 0:
            #     tokens_pair[0] = self.special_tokens.cls
            # else:
            #     tokens_pair = [self.special_tokens.cls] + tokens_pair
            #     tokens_pair = [0] * (padding_len-1) + tokens_pair

            #不添加CLS
            tokens_pair = [0] * padding_len + tokens_pair
            d['tokens_pair'] = torch.LongTensor(tokens_pair)
        
        # (2) 筛选出部分连续的items; 开头待添加CLS
        if self.args.argument_type == "cutoff_subseq":
            # s_len = len(seq) - 1
            s_len = len(seq)
            valid_len = int(self.args.cutoff_subseq_rate * s_len)
            import random
            start_index = random.randint(0, s_len - valid_len + 1)
            sub_tokens = seq[start_index:start_index + valid_len]
            sub_tokens.append(self.special_tokens.cls) #重新添加cls标记位;
            sub_tokens = sub_tokens[-self.max_len:]
            padding_len = self.max_len - len(sub_tokens)
            sub_tokens = [0] * padding_len + sub_tokens
            d['sub_tokens'] = torch.LongTensor(sub_tokens)
        # (3) 随机mask一些items, 用[mask] token来代替; MLM中的mask作为输入;
        # if self.args.argument_type == ""
        # (4) items乱序, 分为两个方法: position_ids的乱序和items的乱序(只乱序子序列); 开头待添加CLS
        if self.args.argument_type == "shuffle_subseq":
            shuffle_subseq = copy.deepcopy(seq)
            s_len = len(shuffle_subseq) - 1 #保持最后一个item为cls;
            valid_len = int(self.args.shuffle_subseq_rate * s_len)
            import random
            start_index = random.randint(0, s_len - valid_len + 1)
            sub_tokens = shuffle_subseq[start_index:start_index + valid_len]
            random.shuffle(sub_tokens)
            shuffle_subseq[start_index:start_index + valid_len] = sub_tokens #打乱部分items;
            shuffle_subseq = shuffle_subseq[-self.max_len:]
            padding_len = self.max_len - len(shuffle_subseq)
            shuffle_subseq = [0] * padding_len + shuffle_subseq
            d['shuffle_subseq'] = torch.LongTensor(shuffle_subseq)
        

        if (index+1) % self.dataset_len == 0:
            #构建mask输入数据;
            tokens = []
            labels = []
            # for s in seq:
            for index, s in enumerate(seq):
                if (index + 1) == len(seq):
                    tokens.append(self.special_tokens.mask)
                    labels.append(0) 
                else:
                    tokens.append(s)
                    labels.append(0)
        else:
            #构建mask输入数据;
            tokens = []
            labels = []
            # for s in seq:
            for index, s in enumerate(seq):
                prob = self.rng.random()
                # if prob < self.mask_prob:
                if prob < self.mask_prob:
                    prob /= self.mask_prob
                    if prob < 0.8:
                        tokens.append(self.special_tokens.mask)
                    elif prob < 0.9:
                        tokens.append(self.rng.randint(1, self.num_items))
                    else:
                        tokens.append(s)

                    labels.append(s)
                else:
                    tokens.append(s)
                    labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        tokens_ori = seq[-self.max_len:]

        padding_len = self.max_len - len(tokens)
        valid_len = len(tokens)

        #添加CLS
        # if padding_len == 0:
        #     tokens[0] = self.special_tokens.cls
        #     labels[0] = 0
        #     tokens_ori[0] = self.special_tokens.cls
        # else:
        #     tokens = [0] * (padding_len-1) + [self.special_tokens.cls] + tokens
        #     labels = [0] * (padding_len) + labels
        #     tokens_ori = [0] * (padding_len-1) + [self.special_tokens.cls] + tokens_ori

        #不添加CLS
        tokens_ori = [0] * padding_len + tokens_ori
        tokens = [0] * padding_len + tokens
        labels = [0] * (padding_len) + labels

        d['tokens'] = torch.LongTensor(tokens)
        d['tokens_ori'] = torch.LongTensor(tokens_ori)
        d['labels'] = torch.LongTensor(labels)

        # pdb.set_trace()
        if self.output_timestamps:
            timestamps = self.user2dict[user]['timestamps']
            timestamps = timestamps[beg:end]
            timestamps = [0] * padding_len + timestamps
            d['timestamps'] = torch.LongTensor(timestamps)

        if self.output_days:
            days = self.user2dict[user]['days']
            days = days[beg:end]
            days = [0] * padding_len + days
            d['days'] = torch.LongTensor(days)

        if self.output_user:
            d['users'] = torch.LongTensor([user])
        return d


class BertConstrastDevDataset(data_utils.Dataset):
    def __init__(self, args, dataset, negative_samples, positions):
        self.user2dict = dataset['user2dict']
        self.positions = positions
        self.max_len = args.max_len
        self.num_items = len(dataset['smap'])
        self.special_tokens = dataset['special_tokens']
        self.negative_samples = negative_samples

        self.output_timestamps = args.dataloader_output_timestamp
        self.output_days = args.dataloader_output_days
        self.output_user = args.dataloader_output_user

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        user, pos = self.positions[index]
        seq = self.user2dict[user]['items']

        beg = max(0, pos + 1 - self.max_len)
        end = pos + 1
        seq = seq[beg:end]

        negs = self.negative_samples[user]
        answer = [seq[-1]]
        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq[-1] = self.special_tokens.mask
        padding_len = self.max_len - len(seq)

        #添加[cls]
        if padding_len == 0:
            seq[0] = self.special_tokens.cls
        else:
            seq = [0] * (padding_len-1) + [self.special_tokens.cls] + seq #左padding;

        tokens = torch.LongTensor(seq)
        candidates = torch.LongTensor(candidates)
        labels = torch.LongTensor(labels)
        d = {'tokens':tokens, 'candidates':candidates, 'labels':labels}

        if self.output_timestamps:
            timestamps = self.user2dict[user]['timestamps']
            timestamps = timestamps[beg:end]
            timestamps = [0] * padding_len + timestamps
            d['timestamps'] = torch.LongTensor(timestamps)

        if self.output_days:
            days = self.user2dict[user]['days']
            days = days[beg:end]
            days = [0] * padding_len + days
            d['days'] = torch.LongTensor(days)

        if self.output_user:
            d['users'] = torch.LongTensor([user])
        return d
