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

class GraphLoaderHeterogeneous():
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
        self.n_user = len(item2id) + 1
        self.m_item = len(item2id) + 1
        
        # train_file = path + '/graph.txt'
        # train_file = path + '/new_graph.txt' #未保证target item没有提前泄露;
        # train_file = path + '/new_cocurrence.txt' #未保证target item没有提前泄露;
        # train_file = path + '/new_cocurrence_v2.txt' #未保证target item没有提前泄露;
        # train_file = path + '/new_cocurrence_correct.txt' #未保证target item没有提前泄露; #效果最佳的模型;
        # train_file = path + '/rm_low_items_cocurrence_correct.txt' #未保证target item没有提前泄露;
        # train_file = path + '/rm_2_low_items_cocurrence_correct_rm_valid.txt' #未保证target item没有提前泄露;
        # train_file = path + '/rm_5_low_items_cocurrence_correct_rm_valid.txt' #未保证target item没有提前泄露;
        train_file_buy = path + config.graph_filename_buy
        train_file_view = path + config.graph_filename_view
        train_file_both = path + config.graph_filename
        
        print("Loading datafile is: ", train_file_buy, train_file_view)
        # train_file = path
        # test_file = path + '/test.txt' #不需要测试集, 在整个数据集中pretrain来获取每个item的表征;
        self.path = path
        m_item = len(item2id) + 1#是否需要+1？

        self.UserItemNet_buy, self.users_D_buy, self.items_D_buy, self._allPos_buy, self.traindataSize_buy = self.conductGraph(train_file_buy, item2id, m_item)
        self.UserItemNet_view, self.users_D_view, self.items_D_view, self._allPos_view, self.traindataSize_view = self.conductGraph(train_file_view, item2id, m_item)
        self.UserItemNet_both, self.users_D_both, self.items_D_both, self._allPos_both, self.traindataSize_both = self.conductGraph(train_file_both, item2id, m_item)
        print("Success to create the graph dataloader.")

        # pdb.set_trace()
        trainData_all_size = self.traindataSize_buy + self.traindataSize_view
        self.traindataSizes = trainData_all_size
        #融合allPos_buy和allPos_view
        self._allPos = []
        # pdb.set_trace()
        for buy_items, veiw_items in zip(self._allPos_buy, self._allPos_view):
            self._allPos.append(np.array(list(set(buy_items.tolist() + veiw_items.tolist()))))
        # pdb.set_trace()
        # print(f"{world.dataset} is ready to go")


    def conductGraph(self, train_file_buy, item2id, m_item):
        trainUniqueUsers, trainItem, trainUser = [], [], []
        # testUniqueUsers, testItem, testUser = [], [], []
        traindataSize = 0
        testDataSize = 0
        n_user = m_item
        m_item = m_item
        user_item_tuple = []

        with open(train_file_buy) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    # items = [int(i) for i in l[1:]]
                    # pdb.set_trace()
                    items = [int(item2id[i]) for i in l[1:]]
                    #convert string to int
                    # uid = user2id[l[0]] #暂时不考虑user对模型的影响, 只考虑items共现的影响;
                    uid = int(item2id[l[0]])
                    # uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    # self.m_item = max(self.m_item, max(items))
                    # self.n_user = max(self.n_user, uid)
                    traindataSize += len(items)
                    # user_item_tuple.extend([(uid, item_id) for item_id in items])
        trainUniqueUsers = np.array(trainUniqueUsers)
        trainUser = np.array(trainUser)
        trainItem = np.array(trainItem)

        # self.m_item += 1 #为什么要加add 1
        # self.n_user += 1
        
        print("item number: {}, user number:{}".format(m_item, m_item))
        # if 'bert' in self.config.model_code:
        #     self.m_item += 1
        #     self.n_user += 1
        # pdb.set_trace()
        # self.Graph = None
        print(f"{traindataSize} interactions for training")
        # print(f"{self.testDataSize} interactions for testing")
        # print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")
        print(f"Graph Sparsity: {(traindataSize) / n_user / m_item}")
        # pdb.set_trace()
        # (users,items), bipartite graph
        UserItemNet = csr_matrix((np.ones(len(trainUser)), (trainUser, trainItem)),
                                      shape=(n_user, m_item)) #第一参数: value, 第二个采纳数: index;
        # pdb.set_trace()
        # assert (self.UserItemNet.transpose() == self.UserItemNet).toarray().all() #必须是对称矩阵才能通过;
        # 共现item relation时, 不需要是对称矩阵, 因此also_review和also_bought是非对称关系;

        users_D = np.array(UserItemNet.sum(axis=1)).squeeze()
        users_D[users_D == 0.] = 1
        items_D = np.array(UserItemNet.sum(axis=0)).squeeze()
        items_D[items_D == 0.] = 1.
        # pre-calculate
        _allPos = self.getUserPosItems(list(range(n_user)), UserItemNet)
        # self.__testDict = self.__build_test()
        return UserItemNet, users_D, items_D, _allPos, traindataSize
    

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSizes
    
    # @property
    # def testDict(self):
    #     return self.__testDict

    @property
    def allPos(self):
        # return self._allPos
        return self._allPos_buy

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
        
    def getSparseGraph(self, UserItemNet, rel_type, UserItemNet_total, Graph=None):
        """
            构建归一化矩阵;
        """
        print("loading adjacency matrix")
        if Graph is None:
            try:
                # pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat_{}.npz'.format(self.config.model_code))
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat_{}_{}.npz'.format(self.config.experiment_name, rel_type))
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32) #为什么构建(user_num + item_num, user_num + item_num)矩阵;
                adj_mat = adj_mat.tolil() #convert list of lists format;
                R = UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok() #convert dictionary of Keys format;
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                
                # rowsum = np.array(adj_mat.sum(axis=1))
                # d_inv = np.power(rowsum, -0.5).flatten()
                # d_inv[np.isinf(d_inv)] = 0.
                # d_mat = sp.diags(d_inv) #(user_num + item_num)

                #归一化采用多类型构建的矩阵;
                adj_mat_total = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32) #为什么构建(user_num + item_num, user_num + item_num)矩阵;
                adj_mat_total = adj_mat_total.tolil() #convert list of lists format;
                R = UserItemNet_total.tolil()
                adj_mat_total[:self.n_users, self.n_users:] = R
                adj_mat_total[self.n_users:, :self.n_users] = R.T
                adj_mat_total = adj_mat_total.todok() #convert dictionary of Keys format;
                rowsum = np.array(adj_mat_total.sum(axis=1))
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
                sp.save_npz(self.path + '/s_pre_adj_mat_{}_{}.npz'.format(self.config.experiment_name, rel_type), norm_adj)

            if self.split == True:
                Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                Graph = Graph.coalesce().to(self.config.device)
                print("don't split the matrix")
        return Graph

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

    def getUserPosItems(self, users, UserItemNet):
        posItems = []
        for user in users:
            posItems.append(UserItemNet[user].nonzero()[1]) #将正向items筛选出;
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