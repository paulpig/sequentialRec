import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time
import pdb

class GraphLoader():
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset

    注意: item ids和序列推荐模型的item ids必须一致;
    """

    # def __init__(self,config ,path="../data/gowalla", user2id=None, item2id=None):
    def __init__(self, config, user2id=None, item2id=None):
        """
            user2id, item2id: This is from sequential behaviors;
        """
        # train or test
        # cprint(f'loading [{path}]')
        path = config.graph_path
        print(f'loading [{path}]')
        self.config = config
        # self.split = config['A_split']
        self.split = False #不分割处理;
        # self.split = config.A_split
        # self.folds = config['A_n_fold']
        # self.folds = config.A_n_fold
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        # train_file = path + '/graph.txt'
        # train_file = path + '/new_graph.txt' #未保证target item没有提前泄露;
        # train_file = path + '/new_cocurrence.txt' #未保证target item没有提前泄露;
        # train_file = path + '/new_cocurrence_v2.txt' #未保证target item没有提前泄露;
        # train_file = path + '/new_cocurrence_correct.txt' #未保证target item没有提前泄露; #效果最佳的模型;
        # train_file = path + '/rm_low_items_cocurrence_correct.txt' #未保证target item没有提前泄露;
        # train_file = path + '/rm_2_low_items_cocurrence_correct_rm_valid.txt' #未保证target item没有提前泄露;
        # train_file = path + '/rm_5_low_items_cocurrence_correct_rm_valid.txt' #未保证target item没有提前泄露;
        # train_file = path + config.graph_filename #重新构建, 每行由<head, rel, tail> 构成;
        train_file = path + config.graph_filename_kgat #重新构建, 每行由<head, rel, tail> 构成;
          
        print("Loading datafile is: ", train_file)
        # train_file = path
        # test_file = path + '/test.txt' #不需要测试集, 在整个数据集中pretrain来获取每个item的表征;
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        # testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0
        single_edge_number = 0 

        self.rel2id = {}
        self.attribute2id = {}

        self.all_head_list = []
        self.all_rel_list = []
        self.all_tail_list = []

        #重写构建邻接矩阵代码这段逻辑, 不仅获取初始化邻接矩阵, 而且得到head, rel and tail list, 用于构建KGE loss and updating the adjacent matrix.
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    rel = l[1]
                    attribute = l[2]

                    if rel not in self.rel2id:
                        self.rel2id[rel] = len(self.rel2id)
                    
                    if attribute not in self.attribute2id:
                        self.attribute2id[attribute] = len(self.attribute2id)

                    rel_id = int(self.rel2id[rel])
                    attribute_id = int(self.attribute2id[attribute])
                    uid = int(item2id[l[0]])

                    
                    self.all_head_list.append(uid)
                    self.all_rel_list.append(rel_id)
                    self.all_tail_list.append(attribute_id)


                    trainUser.append(uid)
                    trainItem.append(attribute_id)
                    self.m_item = max(self.m_item, attribute_id)
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += 1
        # self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser) #item数量;
        self.trainItem = np.array(trainItem) #attribute数量;

        # self.m_item += 1 #为什么要加add 1
        # self.n_user += 1
        
        # self.m_item = len(item2id) + 1
        self.m_item = len(self.attribute2id) + 1
        self.n_user = len(item2id) + 1

        print("single_edge_number: {}".format(single_edge_number))
        print("item number: {}, user number:{}".format(self.m_item, self.n_user))
        if 'bert' in self.config.model_code:
            self.m_item += 1
            self.n_user += 1
        # pdb.set_trace()
        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        # print(f"{self.testDataSize} interactions for testing")
        # print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")
        print(f"Graph Sparsity: {(self.trainDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item)) #第一参数: value, 第二个采纳数: index;
        # pdb.set_trace()
        # assert (self.UserItemNet.transpose() == self.UserItemNet).toarray().all() #必须是对称矩阵才能通过;
        # 共现item relation时, 不需要是对称矩阵, 因此also_review和also_bought是非对称关系;

        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        # self.__testDict = self.__build_test()
        print("Success to create the graph dataloader.")
        # print(f"{world.dataset} is ready to go")

        #{user: (rel, item)}
        graph_kge = {}
        for i, head in enumerate(self.all_head_list):
            if head not in graph_kge:
                graph_kge[head] = [(self.all_rel_list[i], self.all_tail_list[i])]
            else:
                graph_kge[head].append((self.all_rel_list[i], self.all_tail_list[i]))
        self.graph_kge = graph_kge
        # pdb.set_trace()

    
    def conductHeadRelTailList(self, ):
        """
        conducting the head list, the rel list and the tail list.
        """

        return

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    # @property
    # def testDict(self):
    #     return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self,A):
        """
        矩阵分块操作;
        """
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(self.config.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        """
        图转为Tensor形式;
        """
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        """
            构建归一化矩阵;
        """
        print("loading kgat adjacency matrix")
        # pdb.set_trace()
        if self.Graph is None:
            try:
                # pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat_{}.npz'.format(self.config.model_code))
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat_{}_kgat.npz'.format(self.config.experiment_name))
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix. Both Buy and view.")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32) #为什么构建(user_num + item_num, user_num + item_num)矩阵;
                adj_mat = adj_mat.tolil() #convert list of lists format;
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok() #convert dictionary of Keys format;
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv) #(user_num + item_num)
                
                """
                乘以两次对角矩阵的原因是分别除以入度的平方根和出度的平方根;
                """ 
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                # sp.save_npz(self.path + '/s_pre_adj_mat_{}.npz'.format(self.config.model_code), norm_adj)
                sp.save_npz(self.path + '/s_pre_adj_mat_{}_kgat.npz'.format(self.config.experiment_name), norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(self.config.device)
                print("don't split the matrix")
        return self.Graph


    # def __build_test(self):
    #     """
    #     return:
    #         dict: {user: [items]}
    #     """
    #     test_data = {}
    #     for i, item in enumerate(self.testItem):
    #         user = self.testUser[i]
    #         if test_data.get(user):
    #             test_data[user].append(item)
    #         else:
    #             test_data[user] = [item]
    #     return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1]) #将正向items筛选出;
        return posItems


# if __name__ == "__main__":
#     GPU = torch.cuda.is_available()
#     device = torch.device('cuda' if GPU else "cpu")
#     graph_path = "/pub/data/kyyx/wbc/MEANTIME/Data/beauty/graph.txt"
#     config = {"graph_path": graph_path, "A_split": 2, "A_n_fold": 4, "device": device}

#     train_loader = 
#     dataset = train_loader._get_dataset()
#     user2id = dataset['umap']
#     item2id = dataset['smap']
#     graph_dataset = GraphLoader(config)
#     dataset = graph_dataset.getSparseGraph()
#     pdb.set_trace()
#     print("Finish to conduct the graph dataset;")