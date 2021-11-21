import pdb
import torch
from .bert_base import BertBaseModel
from torch import nn
import numpy as np


class LightGCN(BertBaseModel):
    def __init__(self, config:dict, dataset):
        super(LightGCN, self).__init__(config)
        self.config = config
        self.dataset = dataset  #dataloader.BasicDataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        # self.num_users  = len(self.dataset['smap'])
        self.num_items  = self.dataset.m_items
        # self.num_items  = len(self.dataset['smap'])
        # self.latent_dim = self.config['latent_dim_rec']
        self.latent_dim = self.config.latent_dim_rec
        # self.n_layers = self.config['lightGCN_n_layers']
        self.n_layers = self.config.lightGCN_n_layers
        # self.keep_prob = self.config['keep_prob']
        self.keep_prob = self.config.keep_prob
        # self.A_split = self.config['A_split']
        self.A_split = self.config.A_split
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        # if self.config['pretrain'] == 0:
        if self.config.graph_pretrain == False:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            # world.cprint('use NORMAL distribution initilizer')
        else:
            #暂时不pretrain, 之后可考虑pretrain;
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        # print(f"lgn is already to go(dropout:{self.config['dropout']})")
        print(f"lgn is already to go(dropout:{self.config.graph_dropout})")


    @classmethod
    def code(cls):
        return 'lightGCN'

    def get_logits(self, d):
        pass

    def get_scores(self, d, logits):
        pass
    
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        # pdb.set_trace()
        embs = [all_emb]
        # if self.config['dropout']:
        if self.config.graph_dropout:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob) #随机丢弃一些节点;
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        # pdb.set_trace()
        for layer in range(self.n_layers):
            if self.A_split: #按行切分，完成矩阵乘法后，再按照行拼接;
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                # pdb.set_trace()
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users) #embedding layers;
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        # pdb.set_trace()
        #以user表征为中心, item分别正负样本;
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss


    def regular_loss(self, users, pos, neg):
        all_users, all_items = self.computer()
        users_emb_in = all_users[users]
        users_emb_out = all_items[users]

        loss = self.loss_dependence_hisc(torch.cat((users_emb_in, users_emb_out), dim=-1), 2, self.config.latent_dim_rec)

        return loss

    
    def loss_dependence_hisc(self, zdata_trn, ncaps, nhidden):
        loss_dep = torch.zeros(1).cuda()
        hH = (-1/nhidden)*torch.ones(nhidden, nhidden).cuda() + torch.eye(nhidden).cuda()
        kfactor = torch.zeros(ncaps, nhidden, nhidden).cuda()

        for mm in range(ncaps):
            data_temp = zdata_trn[:, mm * nhidden:(mm + 1) * nhidden]
            kfactor[mm, :, :] = torch.mm(data_temp.t(), data_temp)

        for mm in range(ncaps):
            for mn in range(mm + 1, ncaps):
                mat1 = torch.mm(hH, kfactor[mm, :, :])
                mat2 = torch.mm(hH, kfactor[mn, :, :])
                mat3 = torch.mm(mat1, mat2)
                teststat = torch.trace(mat3) / zdata_trn.size(0)
                # pdb.set_trace()
                loss_dep = loss_dep + teststat
        return loss_dep

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb) #(user_num, item_num)
        gamma     = torch.sum(inner_pro, dim=1) #(user_num)
        return gamma

    
    def getUserItemEmb(self):
        # pdb.set_trace()
        all_users, all_items = self.computer()
        return all_users, all_items