import pdb
import torch
from .bert_base import BertBaseModel
from torch import nn
import numpy as np


class LightGCNAttention(BertBaseModel):
    def __init__(self, config:dict, dataset):
        super(LightGCNAttention, self).__init__(config)
        self.config = config
        self.dataset = dataset  #dataloader.BasicDataset
        self.__init_weight()
        self.initWeight()

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


        #初始化异构关系表征;
        self.embedding_rel = torch.nn.Embedding(
            num_embeddings=3, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_rel.weight, std=0.1)

        # linear

        self.f = nn.Sigmoid()
        self.Graph_cate3 = self.dataset.getSparseGraph("cate3")
        self.Graph_brand = self.dataset.getSparseGraph("brand")
        self.Graph_price = self.dataset.getSparseGraph("price")
        # print(f"lgn is already to go(dropout:{self.config['dropout']})")
        print(f"lgn is already to go(dropout:{self.config.graph_dropout})")


    @classmethod
    def code(cls):
        return 'lightGCNAttention'

    def get_logits(self, d):
        pass

    def get_scores(self, d, logits):
        pass

    def initWeight(self):
        # all_weights = dict()
        self.gcn_linears_1 = nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.n_layers)])
        self.gcn_linears_2 = nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.n_layers)])
        self.gcn_linears_3 = nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.n_layers)])
        self.rel_linears_1 = nn.ModuleList([nn.Linear(self.latent_dim, 1) for _ in range(self.n_layers)])
        self.rel_linears_2 = nn.ModuleList([nn.Linear(self.latent_dim, 1) for _ in range(self.n_layers)])
        self.rel_linears_3 = nn.ModuleList([nn.Linear(self.latent_dim, 1) for _ in range(self.n_layers)])

        for layer in self.gcn_linears_1:
            nn.init.normal_(layer.weight, std=0.1)
        for layer in self.gcn_linears_2:
            # nn.init.xavier_uniform_(layer.weight)
            nn.init.normal_(layer.weight, std=0.1)
        for layer in self.gcn_linears_3:
            # nn.init.xavier_uniform_(layer.weight)
            nn.init.normal_(layer.weight, std=0.1)

        for layer in self.rel_linears_1:
            # nn.init.xavier_uniform_(layer.weight)
            nn.init.normal_(layer.weight, std=0.1)
        for layer in self.rel_linears_2:
            # nn.init.xavier_uniform_(layer.weight)
            nn.init.normal_(layer.weight, std=0.1)
        for layer in self.rel_linears_3:
            # nn.init.xavier_uniform_(layer.weight)
            nn.init.normal_(layer.weight, std=0.1)
        return
    
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

    
    def computerMergeHetero(self):
        """
        合并异构图的表征
        """
        users, items = self.computer_inner(self.Graph_cate3, self.Graph_brand, self.Graph_price)
        # users, items = self.computer_outside(self.Graph_cate3, self.Graph_brand, self.Graph_price)
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
        return users, items

    def computer_inner(self, Graph_cate3, Graph_brand, Graph_price):
        """
        propagate methods for lightGCN, inner attention. 
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
        # embs_cate3 = [all_emb]
        # embs_brand = [all_emb]
        # embs_price = [all_emb]

        # if self.config['dropout']:
        if self.config.graph_dropout:
            if self.training:
                print("droping")
                g_droped_cate3 = self.__dropout(self.keep_prob, Graph_cate3) #随机丢弃一些节点;
                g_droped_brand = self.__dropout(self.keep_prob, Graph_brand) #随机丢弃一些节点;
                g_droped_price = self.__dropout(self.keep_prob, Graph_price) #随机丢弃一些节点;
            else:
                g_droped_cate3 = Graph_cate3
                g_droped_brand = Graph_brand
                g_droped_price = Graph_price

        else:
            g_droped_cate3 = Graph_cate3
            g_droped_brand = Graph_brand
            g_droped_price = Graph_price
        
        # cate3
        for layer in range(self.n_layers):
            if self.A_split: #按行切分，完成矩阵乘法后，再按照行拼接;
                temp_emb = []
                for f in range(len(g_droped_cate3)):
                    temp_emb.append(torch.sparse.mm(g_droped_cate3[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb_cate3 = side_emb
            else:
                # pdb.set_trace()
                all_emb_cate3 = torch.sparse.mm(g_droped_cate3, all_emb) #(node_number, dim)
            
            #添加边信息
            # all_emb = nn.functional.leaky_relu(torch.matmul(torch.mul(all_emb, relationship), self.gcn_linears[layer]))
            # all_emb_buy = self.gcn_linears[layer](torch.mul(all_emb_buy, relationship_buy))
            # relationship_buy = self.rel_linears[layer](relationship_buy)
            # all_emb_buy = torch.mul(all_emb_buy, relationship_buy)
            # relationship_buy = self.rel_linears[layer](relationship_buy)


            #brand
            if self.A_split: #按行切分，完成矩阵乘法后，再按照行拼接;
                temp_emb = []
                for f in range(len(g_droped_brand)):
                    temp_emb.append(torch.sparse.mm(g_droped_brand[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb_brand = side_emb
            else:
                # pdb.set_trace()
                all_emb_brand = torch.sparse.mm(g_droped_brand, all_emb) #(node_number, dim)
            

            #price
            if self.A_split: #按行切分，完成矩阵乘法后，再按照行拼接;
                temp_emb = []
                for f in range(len(g_droped_price)):
                    temp_emb.append(torch.sparse.mm(g_droped_price[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb_price = side_emb
            else:
                # pdb.set_trace()
                all_emb_price = torch.sparse.mm(g_droped_price, all_emb) #(node_number, dim)
            #添加边信息
            # all_emb = nn.functional.leaky_relu(torch.matmul(torch.mul(all_emb, relationship), self.gcn_linears[layer]))
            # all_emb_view = self.gcn_linears[layer](torch.mul(all_emb_view, relationship_view))
            # relationship_view = self.rel_linears[layer](relationship_view)
            # all_emb_view = torch.mul(all_emb_view, relationship_view)
            # relationship_view = self.rel_linears[layer](relationship_view)

            #整合来自不同关系的表征;  后续可以尝试不同的融合策略;
            # candidate_embeddings_gate = torch.sigmoid(self.W_graph_para_1(all_emb_view) + self.W_graph_para_2(all_emb_buy))
            # all_emb = candidate_embeddings_gate * all_emb_view + (1. - candidate_embeddings_gate) * all_emb_buy

            # attention 方式整合
            seq_1 = self.rel_linears_1[layer](torch.tanh(self.gcn_linears_1[layer](all_emb_cate3)))#(bs, sl, 1)
            seq_2 = self.rel_linears_2[layer](torch.tanh(self.gcn_linears_2[layer](all_emb_brand))) #(bs, sl, 1)
            seq_3 = self.rel_linears_3[layer](torch.tanh(self.gcn_linears_3[layer](all_emb_price))) #(bs, sl, 1)
            seq_merge = torch.cat([seq_1, seq_2, seq_3], dim=-1)
            seq_merge_weight = nn.functional.softmax(seq_merge, dim=-1) #(bs, sl, 3)
            seq_merge_tensor = torch.cat([torch.unsqueeze(all_emb_cate3, dim=-2), torch.unsqueeze(all_emb_brand, dim=-2), torch.unsqueeze(all_emb_price, dim=-2)], dim=-2) #(bs, sl, 3, dim)
            # pdb.set_trace()
            all_emb = (seq_merge_tensor * torch.unsqueeze(seq_merge_weight, dim=-1)).sum(-2) #(bs, sl, dim)
            # all_emb = nn.functional.leaky_relu(all_emb) #删除激活函数;
            # all_emb = all_emb_view

            # add dropout layer;
            # all_emb = nn.functional.dropout(all_emb)
            #分别构建
            embs.append(all_emb)
            # embs_buy.append(all_emb_buy)
            # embs_view.append(all_emb_view)
            # rels_buy.append(relationship_buy)
            # rels_view.append(relationship_view)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items



    def computer_outside(self, Graph_cate3, Graph_brand, Graph_price):
        """
        propagate methods for lightGCN, outside attention. 
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
        embs_cate3 = [all_emb]
        embs_brand = [all_emb]
        embs_price = [all_emb]

        # if self.config['dropout']:
        if self.config.graph_dropout:
            if self.training:
                print("droping")
                g_droped_cate3 = self.__dropout(self.keep_prob, Graph_cate3) #随机丢弃一些节点;
                g_droped_brand = self.__dropout(self.keep_prob, Graph_brand) #随机丢弃一些节点;
                g_droped_price = self.__dropout(self.keep_prob, Graph_price) #随机丢弃一些节点;
            else:
                g_droped_cate3 = Graph_cate3
                g_droped_brand = Graph_brand
                g_droped_price = Graph_price

        else:
            g_droped_cate3 = Graph_cate3
            g_droped_brand = Graph_brand
            g_droped_price = Graph_price
        
        # cate3
        for layer in range(self.n_layers):
            if self.A_split: #按行切分，完成矩阵乘法后，再按照行拼接;
                temp_emb = []
                for f in range(len(g_droped_cate3)):
                    temp_emb.append(torch.sparse.mm(g_droped_cate3[f], embs_cate3[-1]))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb_cate3 = side_emb
            else:
                # pdb.set_trace()
                all_emb_cate3 = torch.sparse.mm(g_droped_cate3, embs_cate3[-1]) #(node_number, dim)
            
            #添加边信息
            # all_emb = nn.functional.leaky_relu(torch.matmul(torch.mul(all_emb, relationship), self.gcn_linears[layer]))
            # all_emb_buy = self.gcn_linears[layer](torch.mul(all_emb_buy, relationship_buy))
            # relationship_buy = self.rel_linears[layer](relationship_buy)
            # all_emb_buy = torch.mul(all_emb_buy, relationship_buy)
            # relationship_buy = self.rel_linears[layer](relationship_buy)
            #添加边信息


            #brand
            if self.A_split: #按行切分，完成矩阵乘法后，再按照行拼接;
                temp_emb = []
                for f in range(len(g_droped_brand)):
                    temp_emb.append(torch.sparse.mm(g_droped_brand[f], embs_brand[-1]))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb_brand = side_emb
            else:
                # pdb.set_trace()
                all_emb_brand = torch.sparse.mm(g_droped_brand, embs_brand[-1]) #(node_number, dim)
            

            #price
            if self.A_split: #按行切分，完成矩阵乘法后，再按照行拼接;
                temp_emb = []
                for f in range(len(g_droped_price)):
                    temp_emb.append(torch.sparse.mm(g_droped_price[f], embs_price[-1]))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb_price = side_emb
            else:
                # pdb.set_trace()
                all_emb_price = torch.sparse.mm(g_droped_price, embs_price[-1]) #(node_number, dim)
            #添加边信息
            # all_emb = nn.functional.leaky_relu(torch.matmul(torch.mul(all_emb, relationship), self.gcn_linears[layer]))
            # all_emb_view = self.gcn_linears[layer](torch.mul(all_emb_view, relationship_view))
            # relationship_view = self.rel_linears[layer](relationship_view)
            # all_emb_view = torch.mul(all_emb_view, relationship_view)
            # relationship_view = self.rel_linears[layer](relationship_view)

            #整合来自不同关系的表征;  后续可以尝试不同的融合策略;
            # candidate_embeddings_gate = torch.sigmoid(self.W_graph_para_1(all_emb_view) + self.W_graph_para_2(all_emb_buy))
            # all_emb = candidate_embeddings_gate * all_emb_view + (1. - candidate_embeddings_gate) * all_emb_buy

            # attention 方式整合
            # seq_1 = self.rel_linears_1[layer](torch.tanh(self.gcn_linears_1[layer](all_emb_cate3)))#(bs, sl, 1)
            # seq_2 = self.rel_linears_2[layer](torch.tanh(self.gcn_linears_2[layer](all_emb_brand))) #(bs, sl, 1)
            # seq_3 = self.rel_linears_3[layer](torch.tanh(self.gcn_linears_3[layer](all_emb_price))) #(bs, sl, 1)
            # seq_merge = torch.cat([seq_1, seq_2, seq_3], dim=-1)
            # seq_merge_weight = nn.functional.softmax(seq_merge, dim=-1) #(bs, sl, 3)
            # seq_merge_tensor = torch.cat([torch.unsqueeze(all_emb_cate3, dim=-2), torch.unsqueeze(all_emb_brand, dim=-2), torch.unsqueeze(all_emb_price, dim=-2)], dim=-2) #(bs, sl, 3, dim)
            # all_emb = (seq_merge_tensor * torch.unsqueeze(seq_merge_weight, dim=-1)).sum(-2) #(bs, sl, dim)


            # all_emb = nn.functional.leaky_relu(all_emb) #删除激活函数;
            # all_emb = all_emb_view

            # add dropout layer;
            # all_emb = nn.functional.dropout(all_emb)
            #分别构建
            embs.append(all_emb)
            embs_cate3.append(all_emb_cate3)
            embs_brand.append(all_emb_brand)
            embs_price.append(all_emb_price)
            # embs_buy.append(all_emb_buy)
            # embs_view.append(all_emb_view)
            # rels_buy.append(relationship_buy)
            # rels_view.append(relationship_view)
        
        # attention 方式整合
        # attention 方式整合
        embs_cate3 = torch.stack(embs_cate3, dim=1) #(N, dim)
        embs_cate3 = torch.mean(embs_cate3, dim=1)

        embs_brand = torch.stack(embs_brand, dim=1)
        embs_brand = torch.mean(embs_brand, dim=1)

        embs_price = torch.stack(embs_price, dim=1)
        embs_price = torch.mean(embs_price, dim=1)


        seq_1 = self.rel_linears_1[0](torch.tanh(self.gcn_linears_1[0](embs_cate3)))#(bs, sl, 1)
        seq_2 = self.rel_linears_2[0](torch.tanh(self.gcn_linears_2[0](embs_brand))) #(bs, sl, 1)
        seq_3 = self.rel_linears_3[0](torch.tanh(self.gcn_linears_3[0](embs_price))) #(bs, sl, 1)
        seq_merge = torch.cat([seq_1, seq_2, seq_3], dim=-1)
        seq_merge_weight = nn.functional.softmax(seq_merge, dim=-1) #(bs, sl, 3)
        seq_merge_tensor = torch.cat([torch.unsqueeze(all_emb_cate3, dim=-2), torch.unsqueeze(all_emb_brand, dim=-2), torch.unsqueeze(all_emb_price, dim=-2)], dim=-2) #(bs, sl, 3, dim)
        light_out = (seq_merge_tensor * torch.unsqueeze(seq_merge_weight, dim=-1)).sum(-2) #(bs, sl, dim)

        # embs = torch.stack(embs, dim=1)
        # #print(embs.size())
        # light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items



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
        # all_users, all_items = self.computer()
        all_users, all_items = self.computerMergeHetero()
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

    def bpr_loss_add_rel(self, users, pos, neg, rel):
        # pdb.set_trace()
        #以user表征为中心, item分别正负样本;
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        
        #get relation embedding
        rel_emb = self.embedding_rel[rel] #(bs, dim)
        pos_scores = torch.mul(users_emb, (pos_emb + rel_emb))
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, (neg_emb + rel_emb))
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        # all_users, all_items = self.computer()
        # print('forward')
        all_users, all_items = self.computerMergeHetero()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb) #(user_num, item_num)
        gamma     = torch.sum(inner_pro, dim=1) #(user_num)
        return gamma

    
    def getUserItemEmb(self):
        # pdb.set_trace()
        # all_users, all_items = self.computer()
        all_users, all_items = self.computerMergeHetero()
        return all_users, all_items