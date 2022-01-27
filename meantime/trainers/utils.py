import torch
from time import time
import numpy as np
import pdb
import random


def recall(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / labels.sum(1).float()).mean().item()


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(n, k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}

    scores = scores.cpu()
    labels = labels.cpu()
    answer_count = labels.sum(1)
    answer_count_float = answer_count.float()
    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
       cut = cut[:, :k]
       hits = labels_float.gather(1, cut)
       metrics['Recall@%d' % k] = (hits.sum(1) / answer_count_float).mean().item()

       position = torch.arange(2, 2+k)
       weights = 1 / torch.log2(position.float())
       dcg = (hits * weights).sum(1)
       idcg = torch.Tensor([weights[:min(n, k)].sum() for n in answer_count])
       ndcg = (dcg / idcg).mean().item()
       metrics['NDCG@%d' % k] = ndcg

    return metrics


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


def UniformSample_original_v2(dataset, rel_type=None):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array: (trainDataSize, 3), each element is <user, positem, negitem>
    The parameter 'dataset' is from ./dataloaders/graph.py, class Loader;
    """
    # pdb.set_trace()
    total_start = time()
    # dataset : BasicDataset
    print("trainData: ", dataset.trainDataSize)
    print("allPos: ", len(dataset.allPos))
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    if rel_type == None:
        allPos = dataset.allPos
        allNeg = dataset.allNeg
    # pdb.set_trace()
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    
    for i, user in enumerate(users): 
        start = time() 
        # all_voc = [index for index in range(dataset.m_items)]
        posForUser = allPos[user]
        negForUser = allNeg[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]

        #删除正样本
        # negativeItems = list(set(all_voc) - set(posForUser))
        # print("posForUser:", len(posForUser), dataset.m_items)

        negindex = np.random.randint(0, len(negForUser))
        negitem = negForUser[negindex]

        # while True:
        #     negitem = np.random.randint(0, dataset.m_items)
        #     if negitem in posForUser:
        #         continue
        #     else:
        #         break
        S.append([user, positem, negitem]) #<user, positem, negitem>, 1:1:1
        end = time()
        sample_time1 += end - start
        # print("{}/{}".format(i, len(users)))
    total = time() - total_start
    print("sample time:", total)
    return np.array(S)

def UniformSample_original(dataset, rel_type=None):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array: (trainDataSize, 3), each element is <user, positem, negitem>
    The parameter 'dataset' is from ./dataloaders/graph.py, class Loader;
    """
    # pdb.set_trace()
    total_start = time()
    # dataset : BasicDataset
    print("trainData: ", dataset.trainDataSize)
    print("allPos: ", len(dataset.allPos))
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    if rel_type == None:
        allPos = dataset.allPos
    # pdb.set_trace()
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    
    for i, user in enumerate(users): 
        start = time() 
        # all_voc = [index for index in range(dataset.m_items)]
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]

        #删除正样本
        # negativeItems = list(set(all_voc) - set(posForUser))
        # print("posForUser:", len(posForUser), dataset.m_items)

        # negitem = np.random.randint(0, dataset.m_items)
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem]) #<user, positem, negitem>, 1:1:1
        end = time()
        sample_time1 += end - start
        # print("{}/{}".format(i, len(users)))
    total = time() - total_start
    print("sample time:", total)
    return np.array(S)

def sample_pos_triples_for_h(kg_dict, head, n_sample_pos_triples):
        pos_triples = kg_dict[head]
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == n_sample_pos_triples:
                break

            pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
            # tail = pos_triples[pos_triple_idx][0]
            # relation = pos_triples[pos_triple_idx][1]
            tail = pos_triples[pos_triple_idx][1]
            relation = pos_triples[pos_triple_idx][0]

            if relation not in sample_relations and tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails


def sample_neg_triples_for_h(kg_dict, head, relation, n_sample_neg_triples, attribute_voc):
        pos_triples = kg_dict[head]

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break

            tail = np.random.randint(low=0, high=attribute_voc, size=1)[0]
            if (relation, tail) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails


def UniformSample_original_KGE(dataset, rel_type=None):
    """
    The input of the KGE.
    :return:
        np.array: (trainDataSize, 4), each element is <user, rels, positem, negitem>
    The parameter 'dataset' is from ./dataloaders/graph.py, class Loader;
    """
 
    total_start = time()
    # dataset : BasicDataset
    graph_kge_dict = dataset.graph_kge
    userNum = len(graph_kge_dict.keys())
    all_head = list(graph_kge_dict.keys())
    trainNumber = len(dataset.all_head_list)
    attribute_voc = max([item[1] for item in dataset.attribute2id.items()]) + 1
    # attribute_voc = len()
    # print("trainData: ", trainNumber, trainNumber//1000)
    # trainNumber = trainNumber // 10000
    # pdb.set_trace()
    users = np.random.randint(0, userNum, trainNumber)
    S = []
    for user_index in users:
        head = all_head[user_index]
        sample_relations, sample_pos_tails = sample_pos_triples_for_h(graph_kge_dict, head, 1)
        sample_neg_tails = sample_neg_triples_for_h(graph_kge_dict, head, sample_relations[0], 1, attribute_voc)
        S.append([head, sample_relations[0], sample_pos_tails[0], sample_neg_tails[0]])
    
    return np.array(S)


def UniformSample_original_kgat_item2item(dataset, rel_type=None):
    """
    The input of the item2item loss.
    :return:
        np.array: (trainDataSize, 4), each element is <user, rels, positem, negitem>
    The parameter 'dataset' is from ./dataloaders/graph.py, class Loader;
    """

    total_start = time()
    # dataset : BasicDataset
    graph_kge_dict = dataset.graph_kge
    userNum = len(graph_kge_dict.keys())
    all_head = list(graph_kge_dict.keys())
    trainNumber = len(dataset.all_head_list)
    attribute_voc = len(dataset.attribute2id)
    allPos = dataset.allPos
    # pdb.set_trace()
    users = np.random.randint(0, userNum, trainNumber)
    S = []
    for user_index in users:
        head = all_head[user_index]
        sample_relations, sample_pos_tails = sample_pos_triples_for_h(graph_kge_dict, head, 1)
        sample_neg_tails = sample_neg_triples_for_h(graph_kge_dict, head, sample_relations[0], 1, attribute_voc)

        #according to head to sample posuser and neguser
        # pdb.set_trace()
        posForUserRel = allPos[head]
        posindex = np.random.randint(0, len(posForUserRel))
        positem, pos_rel = posForUserRel[posindex]
        posForUser = [item[0] for item in posForUserRel]
        
        while True:
            negitem = np.random.randint(0, dataset.n_users) #都是从user维度采样得到;
            if negitem in posForUser:
                continue
            else:
                break
        S.append([head, sample_relations[0], sample_pos_tails[0], sample_neg_tails[0], positem, negitem, pos_rel]) 
    
    return np.array(S)


def UniformSample_original_DisMulti(dataset, rel_type=None):
    """
    The input of the KGE.
    :return:
        np.array: (trainDataSize, 4), each element is <user, rels, positem, negitem>
    The parameter 'dataset' is from ./dataloaders/graph.py, class Loader;
    """

    total_start = time()
    # dataset : BasicDataset
    graph_kge_dict = dataset.graph_kge
    userNum = len(graph_kge_dict.keys())
    all_head = list(graph_kge_dict.keys())
    trainNumber = len(dataset.all_head_list) #测试使用;
    attribute_voc = len(dataset.item2id)
    relvalue2type = dataset.relvalue2reltype

    # pdb.set_trace()
    users = np.random.randint(0, userNum, trainNumber)
    S = []
    for index, user_index in enumerate(users):
        if index % 1000000 == 0:
            print("{}/{}".format(index, len(users)))
        head = all_head[user_index]
        sample_relations, sample_pos_tails = sample_pos_triples_for_h(graph_kge_dict, head, 1)
        sample_neg_tails = sample_neg_triples_for_h(graph_kge_dict, head, sample_relations[0], 1, attribute_voc)
        
        S.append([head, relvalue2type[sample_relations[0]], sample_relations[0], sample_pos_tails[0], sample_neg_tails[0]])
    
    return np.array(S)
    # attribute2id = dataset.attribute2id

    # S = []
    # for i, head in enumerate(all_head_list):
    #     while True:
    #         negitem = np.random.randint(0, len(attribute2id))
    #         if negitem == all_tail_list[i]:
    #             continue
    #         else:
    #             break
    #     S.append([head, all_rel_list[i], all_tail_list[i], negitem])

    # return np.array(S)
    # user_num = dataset.trainDataSize
    # users = np.random.randint(0, dataset.n_users, user_num)
    # if rel_type == None:
    #     allPos = dataset.allPos
    # # pdb.set_trace()
    # S = []
    # sample_time1 = 0.
    # sample_time2 = 0.
    # for i, user in enumerate(users):
    #     start = time()
    #     posForUser = allPos[user]
    #     if len(posForUser) == 0:
    #         continue
    #     sample_time2 += time() - start
    #     posindex = np.random.randint(0, len(posForUser))
    #     positem = posForUser[posindex]
    #     while True:
    #         negitem = np.random.randint(0, dataset.m_items)
    #         if negitem in posForUser:
    #             continue
    #         else:
    #             break
    #     S.append([user, positem, negitem]) #<user, positem, negitem>, 1:1:1
    #     end = time()
    #     sample_time1 += end - start
    # total = time() - total_start
    # return np.array(S)



def UniformSample_original_add_rel(dataset, S, rel_type):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array: (trainDataSize, 5), each element is <user, positem, negitem, posRel, negRel>
    The parameter 'dataset' is from ./dataloaders/graph.py, class Loader;
    """
    total_start = time()
    # dataset : BasicDataset
    if rel_type == 'buy':
        user_num = dataset.traindataSize_buy
    elif rel_type == 'view':
        user_num = dataset.traindataSize_view
    
    users = np.random.randint(0, dataset.n_users, user_num)
    if rel_type == 'buy':
        allPos = dataset._allPos_buy
    else:
        allPos = dataset._allPos_view

    allPos_total = dataset._allPos
    #选取不在任意一个图中的items;
    # allPos = dataset._allPos
    # pdb.set_trace()
    # S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        posForUser_total = allPos_total[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        #get the posRel
        if rel_type == 'buy':
            posRel = 0
        elif rel_type == 'view':
            posRel = 1
        
        # pdb.set_trace()
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            # if negitem in posForUser:
            if negitem in posForUser_total:
                continue
            else:
                break
        
        negRel = random.randint(0,1)

        S.append([user, positem, negitem, posRel, negRel]) #<user, positem, negitem>, 1:1:1
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    # print("Sampling time:", total)
    return S


def minibatch_add_rel(*tensors, batch_size):

    # batch_size = kwargs.get('batch_size', config['bpr_batch_size'])
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def minibatch(*tensors, batch_size):

    # batch_size = kwargs.get('batch_size', config['bpr_batch_size'])
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


class BPRLoss:
    def __init__(self, recmodel, config):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result