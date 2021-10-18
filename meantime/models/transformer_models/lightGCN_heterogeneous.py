import pdb
import torch
from .bert_base import BertBaseModel
from torch import nn
import numpy as np


class LightGCNHeterogeneous(BertBaseModel):
    def __init__(self, config:dict, dataset):
        super(LightGCNHeterogeneous, self).__init__(config)
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

        #初始化线性层
        self.initWeight()

        UserItemNet_buy = self.dataset.UserItemNet_buy
        UserItemNet_view = self.dataset.UserItemNet_view
        UserItemNet_both = self.dataset.UserItemNet_both
        rel_type = 'buy'
        self.Graph_buy = self.dataset.getSparseGraph(UserItemNet_buy, rel_type, UserItemNet_both) 
        rel_type = 'view'
        self.Graph_view = self.dataset.getSparseGraph(UserItemNet_view, rel_type, UserItemNet_both)


        #初始化异构关系表征;
        self.embedding_rel_buy = torch.nn.Embedding(
            num_embeddings=1, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_rel_buy.weight, std=0.1)
        self.rel_buy = self.embedding_rel_buy.weight

        self.embedding_rel_view = torch.nn.Embedding(
            num_embeddings=1, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_rel_view.weight, std=0.1)
        self.rel_view = self.embedding_rel_view.weight


        self.W_graph_para_1 = nn.Linear(self.config.latent_dim_rec, 1).cuda()
        self.W_graph_para_2 = nn.Linear(self.config.latent_dim_rec, 1).cuda()
        # self.rel_view = None
        # print(f"lgn is already to go(dropout:{self.config['dropout']})")
        print(f"lgn is already to go(dropout:{self.config.graph_dropout})")


    @classmethod
    def code(cls):
        return 'lightGCN_heterogeneous'


    def initWeight(self):
        # all_weights = dict()
        self.gcn_linears = nn.ModuleList([ nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.n_layers)])
        self.rel_linears = nn.ModuleList([ nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.n_layers)])

        for layer in self.gcn_linears:
            nn.init.xavier_uniform_(layer.weight)

        for layer in self.rel_linears:
            nn.init.xavier_uniform_(layer.weight)
        
        return
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
    
    def __dropout(self, keep_prob, Graph):
        if self.A_split:
            graph = []
            for g in Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(Graph, keep_prob)
        return graph

    def computerMergeHetero(self):
        """
        合并异构图的表征
        """
        users, items, rels_emb, users_buy, items_buy, users_view, items_view = self.computer(self.Graph_buy, self.Graph_view, self.rel_buy, self.rel_view)
        # embs_view, rels_view = self.computer(self.Graph_view, self.rel_view)
        #融合两个表征;
        # rels_buy = torch.stack(rels_buy, dim=1)
        # rel_buy_emb = torch.mean(rels_buy, dim=1)

        # rels_view = torch.stack(rels_view, dim=1)
        # rels_view_emb = torch.mean(rels_view, dim=1)

        # embs_total = []
        # for layer in range(self.n_layers):
        #     emb_merge = embs_buy[layer] + embs_view[layer] #后续可以采用gate等方式融合
        #     embs_total.append(emb_merge)
        
        # embs = torch.stack(embs_total, dim=1)
        # light_out = torch.mean(embs, dim=1)
        # users, items = torch.split(light_out, [self.num_users, self.num_items])

        # return users, items, rels_emb
        return users, items, rels_emb, users_buy, items_buy, users_view, items_view

    def computer(self, Graph_buy,  Graph_view, relationship_buy, relationship_view):
        """
        propagate methods for lightGCN
        relationship: (1, dim)

        return:
        embs: list of tensor, the shape of tensor is (n_user + n_item, dim), the shape of list is the number of graph layers;
        rels is the same shape as the embs;
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        # pdb.set_trace()
        embs = [all_emb]
        embs_buy = [all_emb]
        embs_view = [all_emb]
        rels_buy = [relationship_buy]
        rels_view = [relationship_view]
        # if self.config['dropout']:
        if self.config.graph_dropout:
            if self.training:
                print("droping")
                g_droped_buy = self.__dropout(self.keep_prob, Graph_buy) #随机丢弃一些节点;
                g_droped_view = self.__dropout(self.keep_prob, Graph_view) #随机丢弃一些节点;
            else:
                g_droped_buy = Graph_buy
                g_droped_view = Graph_view

        else:
            g_droped_buy = Graph_buy
            g_droped_view = Graph_view
        
        # pdb.set_trace()
        for layer in range(self.n_layers):
            if self.A_split: #按行切分，完成矩阵乘法后，再按照行拼接;
                temp_emb = []
                for f in range(len(g_droped_buy)):
                    temp_emb.append(torch.sparse.mm(g_droped_buy[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb_buy = side_emb
            else:
                # pdb.set_trace()
                all_emb_buy = torch.sparse.mm(g_droped_buy, all_emb) #(node_number, dim)
            
            #添加边信息
            # all_emb = nn.functional.leaky_relu(torch.matmul(torch.mul(all_emb, relationship), self.gcn_linears[layer]))
            # all_emb_buy = self.gcn_linears[layer](torch.mul(all_emb_buy, relationship_buy))
            # relationship_buy = self.rel_linears[layer](relationship_buy)
            # all_emb_buy = torch.mul(all_emb_buy, relationship_buy)
            # relationship_buy = self.rel_linears[layer](relationship_buy)

            if self.A_split: #按行切分，完成矩阵乘法后，再按照行拼接;
                temp_emb = []
                for f in range(len(g_droped_view)):
                    temp_emb.append(torch.sparse.mm(g_droped_view[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb_view = side_emb
            else:
                # pdb.set_trace()
                all_emb_view = torch.sparse.mm(g_droped_view, all_emb) #(node_number, dim)
            
            #添加边信息
            # all_emb = nn.functional.leaky_relu(torch.matmul(torch.mul(all_emb, relationship), self.gcn_linears[layer]))
            # all_emb_view = self.gcn_linears[layer](torch.mul(all_emb_view, relationship_view))
            # relationship_view = self.rel_linears[layer](relationship_view)
            # all_emb_view = torch.mul(all_emb_view, relationship_view)
            # relationship_view = self.rel_linears[layer](relationship_view)

            #整合来自不同关系的表征;  后续可以尝试不同的融合策略;
            candidate_embeddings_gate = torch.sigmoid(self.W_graph_para_1(all_emb_view) + self.W_graph_para_2(all_emb_buy))
            all_emb = candidate_embeddings_gate * all_emb_view + (1. - candidate_embeddings_gate) * all_emb_buy

            # all_emb = nn.functional.leaky_relu(all_emb) #删除激活函数;
            # all_emb = all_emb_view

            # add dropout layer;
            # all_emb = nn.functional.dropout(all_emb)
            #分别构建
            embs.append(all_emb)
            embs_buy.append(all_emb_buy)
            embs_view.append(all_emb_view)
            rels_buy.append(relationship_buy)
            rels_view.append(relationship_view)
        #分割到函数外面分割，比较灵活;
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        #print(embs.size())
        embs_buy = torch.stack(embs_buy, dim=1)
        light_out_buy = torch.mean(embs_buy, dim=1)
        users_buy, items_buy = torch.split(light_out_buy, [self.num_users, self.num_items])

        #print(embs.size())
        embs_view = torch.stack(embs_view, dim=1)
        light_out_view = torch.mean(embs_view, dim=1)
        users_view, items_view = torch.split(light_out_view, [self.num_users, self.num_items])

        rels_buy = torch.stack(rels_buy, dim=1)
        rel_buy_emb = torch.mean(rels_buy, dim=1)

        rels_view = torch.stack(rels_view, dim=1)
        rels_view_emb = torch.mean(rels_view, dim=1) #(1, dim)

        rels_emb = torch.cat((rel_buy_emb, rels_view_emb), 0) #(2, dim), buy:0, view: 1;
        return users, items, rels_emb, users_buy, items_buy, users_view, items_view
    
    def getUsersRating(self, users):
        all_users, all_items, _, users_buy, items_buy, users_view, items_view = self.computerMergeHetero()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items, _, users_buy, items_buy, users_view, items_view = self.computerMergeHetero()
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


    def getEmbedding_add_rel(self, users, pos_items, neg_items, rel_pos, rel_neg):
        all_users, all_items, all_rels, users_buy, items_buy, users_view, items_view = self.computerMergeHetero()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        pos_rel = all_rels[rel_pos]
        # pdb.set_trace()
        neg_rel = all_rels[rel_neg]
        users_emb_ego = self.embedding_user(users) #embedding layers;
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, pos_rel, neg_rel
    
    def bpr_loss_add_rel(self, users, pos, neg, pos_rel, neg_rel):
        # pdb.set_trace()
        #以user表征为中心, item分别正负样本;
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0, pos_rel, neg_rel) = self.getEmbedding_add_rel(users.long(), pos.long(), neg.long(), pos_rel.long(), neg_rel.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb * pos_rel) #添加关系;
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb * neg_rel)
        neg_scores = torch.sum(neg_scores, dim=1)
        # + 
        # self.rel_buy.norm(2).pow(2) +
        # self.rel_view.norm(2).pow(2)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items, _, users_buy, items_buy, users_view, items_view = self.computerMergeHetero()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb) #(batch_size, dim)
        gamma     = torch.sum(inner_pro, dim=1) #(batch_size)
        return gamma

    
    def getUserItemEmb(self):
        all_users, all_items, _, users_buy, items_buy, users_view, items_view = self.computerMergeHetero()
        # return all_users, all_items
        # return users_buy, items_buy, users_view, items_view
        return all_users, all_items, users_view, items_view #只采用整合的表征;