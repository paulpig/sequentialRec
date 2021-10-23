import pdb
import torch
from .bert_base import BertBaseModel
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp


class KGATV2(BertBaseModel):
    def __init__(self, config:dict, dataset):
        super(KGATV2, self).__init__(config)
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

        
        self.attribute_nums = len(self.dataset.attribute2id)
        self.rel_nums = len(self.dataset.rel2id)

        self.embedding_rel = torch.nn.Embedding(
            num_embeddings=self.rel_nums, embedding_dim=self.latent_dim)
        # self.embedding_attribute = torch.nn.Embedding(
        #     num_embeddings=self.attribute_nums, embedding_dim=self.latent_dim)
        if self.config.graph_pretrain == False:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_rel.weight, std=0.1)
            # nn.init.normal_(self.embedding_attribute.weight, std=0.1)

        self.W_R = nn.Parameter(torch.Tensor(self.rel_nums, self.latent_dim, self.latent_dim))
        nn.init.normal_(self.W_R, std=0.1)

        self.W_R_hidden = nn.Parameter(torch.Tensor(self.rel_nums, self.latent_dim, self.latent_dim))
        nn.init.normal_(self.W_R_hidden, std=0.1)

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
        self.Graph = self.dataset.getSparseGraph() #初始化图模型的参数, 后续得通过Attention步骤更新临界矩阵;
        
        #all triple 
        self.kg_l2loss_lambda = self.config.kg_l2loss_lambda
        self.all_head_list = self.dataset.all_head_list
        self.all_rel_list = self.dataset.all_rel_list
        self.all_tail_list = self.dataset.all_tail_list

        #convert tensor
        self.all_head_tensor = torch.Tensor(self.all_head_list ).long().to(dtype=torch.long, device=self.config.device)
        self.all_rel_tensor = torch.Tensor(self.all_rel_list ).long().to(dtype=torch.long, device=self.config.device)
        self.all_tail_tensor = torch.Tensor(self.all_tail_list ).long().to(dtype=torch.long, device=self.config.device)


        self.W_graph_para_1 = nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.n_layers)])
        self.W_graph_para_2 = nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.n_layers)])
        # self.W_graph_para_1 = nn.Linear(self.latent_dim, self.latent_dim)
        # self.W_graph_para_2 = nn.Linear(self.latent_dim, self.latent_dim)
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
                all_emb_neighbor = side_emb
            else:
                # pdb.set_trace()
                all_emb_neighbor = torch.sparse.mm(g_droped, all_emb)

            
            if self.config.kgat_merge == "bilinear":
                all_emb = F.leaky_relu(self.W_graph_para_1[layer](all_emb_neighbor + all_emb)) + F.leaky_relu(self.W_graph_para_2[layer](all_emb_neighbor * all_emb_neighbor))
            elif self.config.kgat_merge == "lightgcn":
                all_emb = F.leaky_relu(self.W_graph_para_1[layer](all_emb_neighbor + all_emb))
            elif self.config.kgat_merge == "add":
                all_emb = all_emb_neighbor + all_emb
            elif self.config.kgat_merge == "max":
                all_emb = torch.cat([all_emb_neighbor.unsqueeze(-2), all_emb.unsqueeze(-2)], dim=-2).max(dim=-2).values
            elif self.config.kgat_merge == "mean":
                all_emb = torch.cat([all_emb_neighbor.unsqueeze(-2), all_emb.unsqueeze(-2)], dim=-2).mean(dim=-2)

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
    
    def bpr_loss(self, users, pos, neg, rel):
        # pdb.set_trace()
        #以user表征为中心, item分别正负样本;
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        
        W_R_param = self.W_R_hidden[rel] #(bs, dim, dim)
        # users_emb = torch.bmm(users_emb.unsqueeze(1), W_R_param).squeeze(1) #(bs, dim)
        pos_emb = torch.bmm(pos_emb.unsqueeze(1), W_R_param).squeeze(1)
        neg_emb = torch.bmm(neg_emb.unsqueeze(1), W_R_param).squeeze(1)

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss

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
        # all_users, all_items = self.computer()
        if self.config.kgat_output == "emb":
            all_users = self.embedding_user.weight
            all_items = self.embedding_item.weight
        elif self.config.kgat_output == "hidden":
            all_users, all_items = self.computer()
            
        return all_users, all_items


    #=====================KGE loss =================================

    def _L2_loss_mean(self, x):
        return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

    def tranR_loss(self, head, rel, pos_tail, neg_tail):
        """
        Calculating the tranR loss.
        input:
            head, rel, pos_tail, neg_tail: (bs)
        ouput:
            loss: scalar;
        """
        # pdb.set_trace()
        head_embeddings = self.embedding_user(head)
        rel_embeddings = self.embedding_rel(rel)
        tail_pos_embeddings = self.embedding_item(pos_tail) #(bs, dim)
        tail_neg_embeddings = self.embedding_item(neg_tail)

        W_R_param = self.W_R[rel] #(bs, dim, dim)
        
        head_embeddings_h = torch.bmm(head_embeddings.unsqueeze(1), W_R_param).squeeze(1) #(bs, dim)
        tail_embeddings_pos_h = torch.bmm(tail_pos_embeddings.unsqueeze(1), W_R_param).squeeze(1) #(bs, dim)
        tail_embeddings_neg_h = torch.bmm(tail_neg_embeddings.unsqueeze(1), W_R_param).squeeze(1) #(bs, dim)

        pos_score = torch.sum(torch.pow(head_embeddings_h + rel_embeddings - tail_embeddings_pos_h, 2), dim=1)     # (kg_batch_size)
        neg_score = torch.sum(torch.pow(head_embeddings_h + rel_embeddings - tail_embeddings_neg_h, 2), dim=1)     # (kg_batch_size)

        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = self._L2_loss_mean(head_embeddings_h) + self._L2_loss_mean(rel_embeddings) + self._L2_loss_mean(tail_embeddings_pos_h) + self._L2_loss_mean(tail_embeddings_neg_h)
        # pdb.set_trace()
        # loss = kg_loss + self.kg_l2loss_lambda * l2_loss

        return kg_loss,  l2_loss

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

    
    def updateAttentionScore(self):
        """"
        After training the KGE and the recommendation, updating the attention score (adjacent matrix)
        output:
            attention_score: the sparse matrix, shape:(|U|+|V|, |U|+|V|)
        """

        all_head = self.all_head_tensor
        all_r = self.all_rel_tensor
        all_tail = self.all_tail_tensor

        # the embedding layer
        head_embeddings = self.embedding_user(all_head)
        rel_embeddings = self.embedding_rel(all_r)
        tail_pos_embeddings = self.embedding_item(all_tail) #(all, dim)
        W_R_param = self.W_R[all_r] #(all, dim, dim)


        # the attention operation
        head_embeddings_h = torch.bmm(head_embeddings.unsqueeze(1), W_R_param).squeeze(1) #(all, dim)
        tail_embeddings_h = torch.bmm(tail_pos_embeddings.unsqueeze(1), W_R_param).squeeze(1) #(all, dim)

        attention_score = torch.sum(head_embeddings_h * torch.tanh(tail_embeddings_h + rel_embeddings), dim=-1) #(all)
        # conducting the adjacent matrix in the sparse format

        
        attention_score = attention_score.cpu().detach().numpy()
        attention_matrix_score = csr_matrix((attention_score, (self.all_head_list, self.all_tail_list)),
                                      shape=(self.num_users, self.num_items)) #第一参数: value, 第二个采纳数: index;

        adj_mat = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items), dtype=np.float32) #为什么构建(user_num + item_num, user_num + item_num)矩阵;
        adj_mat = adj_mat.tolil() #convert list of lists format;
        R = attention_matrix_score.tolil()
        adj_mat[:self.num_users, self.num_users:] = R
        adj_mat[self.num_users:, :self.num_users] = R.T
        adj_mat = adj_mat.tocsr() #convert dictionary of Keys format;

        attention_matrix_score_tensor = self._convert_sp_mat_to_sp_tensor(adj_mat)
        
        # the softmax function
        # pdb.set_trace()
        attention_matrix_score_tensor = torch.sparse.softmax(attention_matrix_score_tensor, dim=1)

        return attention_matrix_score_tensor.to(device=self.config.device)