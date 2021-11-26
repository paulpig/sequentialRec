from time import time
from meantime.loggers import *
# from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY, TRAIN_LOADER_RNG_STATE_DICT_KEY
from meantime.config import *
from meantime.utils import AverageMeterSet
from meantime.utils import fix_random_seed_as
from meantime.analyze_table import find_saturation_point
from meantime.dataloaders import get_dataloader
from .utils import recalls_and_ndcgs_for_ks
import json
import time

from .base import AbstractTrainer
from meantime.trainers.utils import UniformSample_original, timer, minibatch, shuffle, UniformSample_original_KGE
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import numpy as np
from meantime.models.transformer_models.lightGCN import LightGCN
from meantime.models.transformer_models.GraphGAT import KGAT

from abc import *
from pathlib import Path
import os
import pdb
# from meantime.dataloaders.graph import GraphLoader
from meantime.dataloaders.graph_add_mi_rm_sq import GraphLoader
# from meantime.dataloaders.graph_cate_brand import GraphLoaderCateBrand
from meantime.dataloaders.graphGAT import GraphLoader as GATLoader
# from meantime.dataloaders.graphGAT 

class GraphTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, local_export_root):
        """
        train_loader, val_loader, test_loader are objects of the pytorch:

        we need to use 'get_dataloader' to get the dataloader object;
        """
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)

        self.kg_l2loss_lambda = self.args.kg_l2loss_lambda
        #graph-based model and loader;
        # if graph_loader != None and graph_model != None:
            # self.graph_loader = graph_loader
        # pdb.set_trace()
        # mode = 'train'
        dataset = get_dataloader(args).dataset
        # pdb.set_trace()
        user2id = dataset['umap']
        item2id = dataset['smap']


        json_str = json.dumps(item2id)
        with open('item2id.json', 'w') as json_file:
            json_file.write(json_str)
        
        self.graph_loader = GraphLoader(self.args, user2id, item2id)
        #add cate
        # self.graph_loader_cate = GraphLoaderCateBrand(self.args, user2id, item2id)
        self.graph_loader_kgat = GATLoader(self.args, user2id, item2id)
        
        # pdb.set_trace()
        # self.graph_model = graph_model
        self.graph_model = LightGCN(self.args, self.graph_loader).to(self.device)
        #add cate
        self.graph_model_kgat = KGAT(self.args, self.graph_loader_kgat).to(self.device)

        self.graph_epochs = args.graph_epochs
        # self.graph_cate_epochs = args.graph_cate_epochs
        self.graph_attribute_epochs = args.graph_attribute_epochs
        # self.graph_optimizer = self._create_graph_optimizer() #创建图模型优化器;
        
        self.use_parallel = args.use_parallel
        if self.use_parallel:
            self.model = nn.DataParallel(self.model)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        # self.optimizer = self._create_optimizer()
        self.optimizer = self._create_optimizer_total()
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_step, gamma=args.gamma)
        self.clip_grad_norm = args.clip_grad_norm
        self.epoch_start = 0
        self.best_epoch = self.epoch_start - 1
        self.best_metric_at_best_epoch = -1
        self.accum_iter_start = 0

        self.num_epochs = args.num_epochs
        if self.num_epochs == -1:
            self.num_epochs = 987654321  # Technically Infinite
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        self.saturation_wait_epochs = args.saturation_wait_epochs

        self.pilot = args.pilot
        if self.pilot:
            self.num_epochs = 1
            self.pilot_batch_cnt = 1

        self.local_export_root = local_export_root
        # pdb.set_trace()
        self.train_loggers, self.val_loggers, self.test_loggers = self._create_loggers() if not self.pilot else (None, None, None)
        self.add_extra_loggers()
        
        #
        self.logger_service = LoggerService(args, self.train_loggers, self.val_loggers, self.test_loggers)
        self.log_period_as_iter = args.log_period_as_iter
        # pdb.set_trace()
        self.resume_training = args.resume_training
        if self.resume_training:
            print('Restoring previous training state')
            # self._restore_training_state()
            self._restore_best_state_model(train_type='pretrain', is_best=False) #加载预模型;
            print('Finished restoring')


    @classmethod
    def code(cls):
        # return 'graph_sasrec_improve_add_cate_brand'
        return 'graph_sasrec_improve_lightgcn_kgat_add_mi_rm_square'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    # def calculate_loss(self, batch):
    #     pass

    def calculate_loss(self, batch):
        # loss = self.model(batch, loss=True)
        # loss = loss.mean()
        # return loss
        d = self.model(batch)
        loss, loss_cnt = d['loss'], d['loss_cnt']
        loss = (loss * loss_cnt).sum() / loss_cnt.sum()
        return loss

    
    def saveGraphOutputTensor(self):
        user_hidden_rep, item_hidden_rep = self.graph_model.getUserItemEmb() #(bs, sl, dim)

        valid_labels_gate = torch.sigmoid(self.model.W_graph_para_1(item_hidden_rep) + self.model.W_graph_para_2(user_hidden_rep))
        lightgcn_rep = valid_labels_gate * item_hidden_rep + (1. - valid_labels_gate) * user_hidden_rep

        # self.user_hidden_rep_cate, self.item_hidden_rep_cate = self.graph_model_cate.getUserItemEmb()
        kgat_rep, _ = self.graph_model_kgat.getUserItemEmb() #(bs, sl, dim)

        def save_tensor(tensor_input, file_name):
            tensor_input = tensor_input.detach().cpu().numpy()
            np.savetxt(file_name, tensor_input)

        save_tensor(lightgcn_rep, "lightgcn.txt")
        save_tensor(kgat_rep, "kgat.txt")

        return

    def calculate_metrics(self, batch):
        labels = batch['labels']
        scores = self.model(batch)['scores']  # B x C
        # scores = scores.gather(1, candidates)  # B x C

        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics

    def trainGraphModelOneEpochKGAT(self, optim_graph):
        # Recmodel = self.graph_model
        # Recmodel.train()
        self.graph_model_kgat.train()
        # bpr: utils.BPRLoss = loss_class

        # self.weight_decay = config['decay']
        self.weight_decay = self.args.weight_decay
        # self.lr = config['lr']
        
        with timer(name="Sample"):
            # S = UniformSample_original(self.graph_loader_kgat)
            S = UniformSample_original_KGE(self.graph_loader_kgat)
            
        users = torch.Tensor(S[:, 0]).long()
        rels = torch.Tensor(S[:, 1]).long() #(len(train_items))
        posItems = torch.Tensor(S[:, 2]).long()
        negItems = torch.Tensor(S[:, 3]).long() #(len(train_items))
        

        users = users.to(self.args.device)
        rels = rels.to(self.args.device)
        posItems = posItems.to(self.args.device)
        negItems = negItems.to(self.args.device)
        users, rels, posItems, negItems = shuffle(users, rels, posItems, negItems)
        # total_batch = len(users) // world.config['bpr_batch_size'] + 1
        total_batch = len(users) // self.args.bpr_batch_size + 1
        aver_loss = 0.
        # pdb.set_trace()
        for (batch_i,
            (batch_users,
            batch_rels,
            batch_pos,
            batch_neg)) in enumerate(minibatch(users, rels, posItems, negItems, batch_size=self.args.bpr_batch_size)):
            # cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
            # loss, reg_loss = self.model.bpr_loss(batch_users, batch_pos, batch_neg)
            loss, reg_loss = self.graph_model_kgat.bpr_loss(batch_users, batch_pos, batch_neg, batch_rels)
            reg_loss = reg_loss*self.weight_decay
            loss = loss + reg_loss

            optim_graph.zero_grad()
            loss.backward(retain_graph=True)
            optim_graph.step()
            cri = loss.cpu().item()
            aver_loss += cri
            # if world.tensorboard:
            #     w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
        aver_loss = aver_loss / total_batch
        time_info = timer.dict()
        timer.zero()

        # add the KGE loss and update the adjacent matrix (TO DO)
        optim_graph_kge = optim_graph #同一个optim;

        with timer(name="SampleKGE"):
            S = UniformSample_original_KGE(self.graph_loader_kgat) #return <centor_node, rel, posItems, negItems>
        users = torch.Tensor(S[:, 0]).long()
        rels = torch.Tensor(S[:, 1]).long()
        posItems = torch.Tensor(S[:, 2]).long()
        negItems = torch.Tensor(S[:, 3]).long() #(len(train_items))

        # pdb.set_trace()
        users = users.to(dtype=torch.long, device=self.args.device)
        rels = rels.to(dtype=torch.long, device=self.args.device)
        posItems = posItems.to(dtype=torch.long, device=self.args.device)
        negItems = negItems.to(dtype=torch.long, device=self.args.device)

        users, rels, posItems, negItems = shuffle(users, rels, posItems, negItems)
        # total_batch = len(users) // world.config['bpr_batch_size'] + 1
        total_batch = len(users) // self.args.bpr_batch_size + 1
        tranR_aver_loss = 0.
        # pdb.set_trace()
        for (batch_i,
            (batch_users,
            batch_rels,
            batch_pos,
            batch_neg)) in enumerate(minibatch(users, rels, posItems, negItems, batch_size=self.args.bpr_batch_size)):

            tranR_loss, reg_loss = self.graph_model_kgat.tranR_loss(batch_users, batch_rels, batch_pos, batch_neg)
            
            # reg_loss = reg_loss*self.kg_l2loss_lambda
            reg_loss = reg_loss*self.weight_decay
            tranR_loss = tranR_loss + reg_loss

            optim_graph_kge.zero_grad()
            tranR_loss.backward(retain_graph=True)
            optim_graph_kge.step()
            cri = tranR_loss.cpu().item()
            tranR_aver_loss += cri
            # if world.tensorboard:
            #     w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
        tranR_aver_loss = tranR_aver_loss / total_batch
        tranR_time_info = timer.dict()
        timer.zero()

        # updating attention scores
        with torch.no_grad():
            # pdb.set_trace()
            att = self.graph_model_kgat.updateAttentionScore()
            self.graph_model_kgat.Graph = att

        return f"loss{aver_loss:.4f}-{time_info}" + "----------" + f"loss{tranR_aver_loss:.4f}-{tranR_time_info}"
    

    # def trainGraphModelOneEpochCate(self, optim_graph):
    #     """
    #     训练基于cate的图模型;
    #     """
    #     # Recmodel = self.graph_model
    #     # Recmodel.train()
    #     self.graph_model_cate.train()
    #     # bpr: utils.BPRLoss = loss_class

    #     # self.weight_decay = config['decay']
    #     self.weight_decay = self.args.weight_decay
    #     # self.lr = config['lr']
        
    #     with timer(name="Sample"):
    #         S = UniformSample_original(self.graph_loader_cate)
    #     users = torch.Tensor(S[:, 0]).long()
    #     posItems = torch.Tensor(S[:, 1]).long()
    #     negItems = torch.Tensor(S[:, 2]).long() #(len(train_items))

    #     users = users.to(self.args.device)
    #     posItems = posItems.to(self.args.device)
    #     negItems = negItems.to(self.args.device)
    #     users, posItems, negItems = shuffle(users, posItems, negItems)
    #     # total_batch = len(users) // world.config['bpr_batch_size'] + 1
    #     total_batch = len(users) // self.args.bpr_batch_size + 1
    #     aver_loss = 0.
    #     # pdb.set_trace()
    #     for (batch_i,
    #         (batch_users,
    #         batch_pos,
    #         batch_neg)) in enumerate(minibatch(users, posItems, negItems, batch_size=self.args.bpr_batch_size)):
    #         # cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
    #         # loss, reg_loss = self.model.bpr_loss(batch_users, batch_pos, batch_neg)
    #         loss, reg_loss = self.graph_model_cate.bpr_loss(batch_users, batch_pos, batch_neg)
    #         reg_loss = reg_loss*self.weight_decay
    #         loss = loss + reg_loss

    #         optim_graph.zero_grad()
    #         loss.backward(retain_graph=True)
    #         optim_graph.step()
    #         cri = loss.cpu().item()
    #         aver_loss += cri
    #         # if world.tensorboard:
    #         #     w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    #     aver_loss = aver_loss / total_batch
    #     time_info = timer.dict()
    #     timer.zero()
    #     return f"loss{aver_loss:.4f}-{time_info}"


    def trainGraphModelOneEpoch(self, optim_graph):
        # Recmodel = self.graph_model
        # Recmodel.train()
        self.graph_model.train()
        # bpr: utils.BPRLoss = loss_class

        # self.weight_decay = config['decay']
        self.weight_decay = self.args.weight_decay
        # self.lr = config['lr']
        
        with timer(name="Sample"):
            S = UniformSample_original(self.graph_loader)
        users = torch.Tensor(S[:, 0]).long()
        posItems = torch.Tensor(S[:, 1]).long()
        negItems = torch.Tensor(S[:, 2]).long() #(len(train_items))

        users = users.to(self.args.device)
        posItems = posItems.to(self.args.device)
        negItems = negItems.to(self.args.device)
        users, posItems, negItems = shuffle(users, posItems, negItems)
        # total_batch = len(users) // world.config['bpr_batch_size'] + 1
        total_batch = len(users) // self.args.bpr_batch_size + 1
        aver_loss = 0.
        # pdb.set_trace()
        for (batch_i,
            (batch_users,
            batch_pos,
            batch_neg)) in enumerate(minibatch(users, posItems, negItems, batch_size=self.args.bpr_batch_size)):
            # cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
            # loss, reg_loss = self.model.bpr_loss(batch_users, batch_pos, batch_neg)
            loss, reg_loss = self.graph_model.bpr_loss(batch_users, batch_pos, batch_neg)
            reg_loss = reg_loss*self.weight_decay
            loss = loss + reg_loss

            optim_graph.zero_grad()
            loss.backward(retain_graph=True)
            optim_graph.step()
            cri = loss.cpu().item()
            aver_loss += cri
            # if world.tensorboard:
            #     w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
        aver_loss = aver_loss / total_batch
        time_info = timer.dict()
        timer.zero()
        return f"loss{aver_loss:.4f}-{time_info}"
    

    def train(self):
        epoch = self.epoch_start
        best_epoch = self.best_epoch
        accum_iter = self.accum_iter_start
        # self.validate(epoch-1, accum_iter, self.val_loader)
        best_metric = self.best_metric_at_best_epoch
        stop_training = False

        # add graph-based training
        self.lr = self.args.lr
        self.graph_opt = optim.Adam(self.graph_model.parameters(), lr=self.lr)
        # self.graph_opt_cate = optim.Adam(self.graph_model_cate.parameters(), lr=self.lr)
        self.graph_opt_attribute = optim.Adam(self.graph_model_kgat.parameters(), lr=self.lr)

        #预训练graph模型;
        for epoch in range(self.graph_epochs):
            info_train_loss = self.trainGraphModelOneEpoch(self.graph_opt)
            print("Both buy and view loss:", info_train_loss)

        #预预先cate_brand graph模型
        for epoch in range(self.graph_attribute_epochs):
            # info_train_loss = self.trainGraphModelOneEpochCate(self.graph_opt_cate)
            info_train_loss = self.trainGraphModelOneEpochKGAT(self.graph_opt_attribute)
            print("cate_and_graph_loss:", info_train_loss)       

        print("Finish training the LightGCN model;")

        #加载模型的额外的参数
        self.model.createMergeParameter() #创建merge参数;
        
        #pdb.set_trace()
        #get user and item embeddings
        self.user_hidden_rep, self.item_hidden_rep = self.graph_model.getUserItemEmb()
        # self.user_hidden_rep_cate, self.item_hidden_rep_cate = self.graph_model_cate.getUserItemEmb()
        self.user_hidden_rep_cate, self.item_hidden_rep_cate = self.graph_model_kgat.getUserItemEmb()

        #setting representations to sequential models; 由于user embedding不参与模型, 两个输出的均是item表征;
        self.model.setUserItemRepFromGraph(self.user_hidden_rep, self.item_hidden_rep, self.user_hidden_rep_cate, self.item_hidden_rep_cate) #每次加载相同的hidden representatin, 不合理;
        
        print("Finish setting user embeddings and item embeddings;")

        for epoch in range(self.epoch_start, self.num_epochs):
            if self.pilot:
                print('epoch', epoch)
            
            fix_random_seed_as(epoch)  # fix random seed at every epoch to make it perfectly resumable
            accum_iter = self.train_one_epoch(epoch, accum_iter, self.train_loader)

            # #经过一次epoch, 重新获取item representation;
            # self.user_hidden_rep, self.item_hidden_rep = self.graph_model.getUserItemEmb()
            # #setting representations to sequential models; 由于user embedding不参与模型, 两个输出的均是item表征;
            # self.model.setUserItemRepFromGraph(self.user_hidden_rep, self.item_hidden_rep) #每次加载相同的hidden representatin, 不合理;

            self.optimizer.step()
            # self.lr_scheduler.step()  # step before val because state_dict is saved at val. it doesn't affect val result
            start_time = time.time()
            val_log_data = self.validate(epoch, accum_iter, mode='val') #用验证代码, 每次保存模型, 调用log_val方法;
            # print("val time: {}".format(time.time()-start_time))
            metric = val_log_data[self.best_metric] #默认是NDCG指标;
            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch
            elif (self.saturation_wait_epochs is not None) and\
                    (epoch - best_epoch >= self.saturation_wait_epochs):
                stop_training = True  # stop training if val perf doesn't improve for saturation_wait_epochs

            if stop_training:
                # load best model
                best_model_logger = self.val_loggers[-1] #最后一个存放的是bestModel;
                assert isinstance(best_model_logger, BestModelLogger)
                weight_path = best_model_logger.filepath() #检索最有模型路径;
                if self.use_parallel:
                    self.model.module.load(weight_path)
                else:
                    self.model.load(weight_path) #从valid集最优参数中加载模型来测试;
                # self.validate(epoch, accum_iter, mode='test')  # test result at best model
                self.validate(best_epoch, accum_iter, mode='test')  # test result at best model
                break
        
        train_type = ''
        if self.args.finetune_flag != True:
            train_type = 'pretrain'
        else:
            train_type = 'finetune'
        # pdb.set_trace()
        self.logger_service.complete({
            'state_dict': (self._create_state_dict(epoch, accum_iter)),
            'train_type': train_type #指明是预训练还是微调;
        }) #循环调用每次logger中的complete方法, 其中对于valid方法, 保存最后一次的模型; 以final为后缀;

    def just_validate(self, mode):
        dummy_epoch, dummy_accum_iter = 0, 0
        self.validate(dummy_epoch, dummy_accum_iter, mode)

    def train_one_epoch(self, epoch, accum_iter, train_loader, **kwargs):
        self.model.train()
        self.graph_model.train()
        # self.graph_model_cate.train()
        self.graph_model_kgat.train()

        average_meter_set = AverageMeterSet()
        num_instance = 0
        tqdm_dataloader = tqdm(train_loader) if not self.pilot else train_loader
        # pdb.set_trace()
        for batch_idx, batch in enumerate(tqdm_dataloader):
            if self.pilot and batch_idx >= self.pilot_batch_cnt:
                break
                
            #经过一次epoch, 重新获取item representation; 修改为每次batch就重新获取item representation;
            self.user_hidden_rep, self.item_hidden_rep = self.graph_model.getUserItemEmb()
            # self.user_hidden_rep_cate, self.item_hidden_rep_cate = self.graph_model_cate.getUserItemEmb()
            self.user_hidden_rep_cate, self.item_hidden_rep_cate = self.graph_model_kgat.getUserItemEmb()
            #setting representations to sequential models; 由于user embedding不参与模型, 两个输出的均是item表征;
            self.model.setUserItemRepFromGraph(self.user_hidden_rep, self.item_hidden_rep, self.user_hidden_rep_cate, self.item_hidden_rep_cate) #每次加载相同的hidden representatin, 不合理;

            batch_size = next(iter(batch.values())).size(0)
            batch = {k:v.to(self.device) for k, v in batch.items()}
            # pdb.set_trace() # 参照原始bert模型构建输入数据, 随机mask任意的词, 注意此处没有特意去mask next item;
            num_instance += batch_size

            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)
            if isinstance(loss, tuple):
                loss, extra_info = loss
                for k, v in extra_info.items():
                    average_meter_set.update(k, v)
            loss.backward(retain_graph=True)

            if self.clip_grad_norm is not None:
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                # torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.graph_model.parameters()) + list(self.graph_model_cate.parameters()), self.clip_grad_norm)
                # torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.graph_model.parameters()) + list(self.graph_model_kgat.parameters()), self.clip_grad_norm)
                torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.graph_model.parameters()), self.clip_grad_norm)
            
            
            # pdb.set_trace()
            # for name, parameters in self.graph_model.named_parameters():
            #     print(name, ':', parameters.size())
            self.optimizer.step()
            # pdb.set_trace()
            # for name, parameters in self.graph_model.named_parameters():
            #     print(name, ':', parameters.size())
            
            average_meter_set.update('loss', loss.item())
            if not self.pilot:
                tqdm_dataloader.set_description(
                    'Epoch {}, loss {:.4f} '.format(epoch, average_meter_set['loss'].avg))

            accum_iter += batch_size

            if self._needs_to_log(accum_iter):
                if not self.pilot:
                    tqdm_dataloader.set_description('Logging')
                log_data = {
                    # 'state_dict': (self._create_state_dict()),
                    'epoch': epoch,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                log_data.update(kwargs)
                self.log_extra_train_info(log_data)
                self.logger_service.log_train(log_data)

        log_data = {
            # 'state_dict': (self._create_state_dict()),
            'epoch': epoch,
            'accum_iter': accum_iter,
            'num_train_instance': num_instance,
        }
        log_data.update(average_meter_set.averages())
        log_data.update(kwargs)
        self.log_extra_train_info(log_data)
        self.logger_service.log_train(log_data)
        return accum_iter

    def validate(self, epoch, accum_iter, mode, doLog=True, **kwargs):
        """
            根据model来预测, 测试时不需要图模型的forward步骤;
        """
        if mode == 'val':
            loader = self.val_loader
        elif mode == 'test':
            loader = self.test_loader
        else:
            raise ValueError

        self.model.eval()
        self.graph_model.eval()
        # self.graph_model_cate.eval()
        self.graph_model_kgat.eval()

        average_meter_set = AverageMeterSet()
        num_instance = 0

        train_type = ''
        if self.args.finetune_flag != True:
            train_type = 'pretrain'
        else:
            train_type = 'finetune'


        self.user_hidden_rep, self.item_hidden_rep = self.graph_model.getUserItemEmb()
        # self.user_hidden_rep_cate, self.item_hidden_rep_cate = self.graph_model_cate.getUserItemEmb()
        self.user_hidden_rep_cate, self.item_hidden_rep_cate = self.graph_model_kgat.getUserItemEmb()

        #setting representations to sequential models; 由于user embedding不参与模型, 两个输出的均是item表征;
        self.model.setUserItemRepFromGraph(self.user_hidden_rep, self.item_hidden_rep, self.user_hidden_rep_cate, self.item_hidden_rep_cate) #每次加载相同的hidden representatin, 不合理;


        with torch.no_grad():
            tqdm_dataloader = tqdm(loader) if not self.pilot else loader
            # pdb.set_trace()
            for batch_idx, batch in enumerate(tqdm_dataloader):
                # pdb.set_trace()
                if self.pilot and batch_idx >= self.pilot_batch_cnt:
                    # print('Break validation due to pilot mode')
                    break
                batch = {k:v.to(self.device) for k, v in batch.items()}
                batch_size = next(iter(batch.values())).size(0)
                # pdb.set_trace() #只mask最后一个token, 并且负采样100个items, 负采样步骤在哪儿实现呢?
                num_instance += batch_size

                metrics = self.calculate_metrics(batch) #计算指标, 由于self.model之前已经加载权重了;

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                if not self.pilot:
                    description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                          ['Recall@%d' % k for k in self.metric_ks[:3]]
                    description = '{}: '.format(mode.capitalize()) + ', '.join(s + ' {:.4f}' for s in description_metrics)
                    description = description.replace('NDCG', 'N').replace('Recall', 'R')
                    description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                    tqdm_dataloader.set_description(description)

            log_data = {
                'state_dict': (self._create_state_dict(epoch, accum_iter)),
                'epoch': epoch,
                'accum_iter': accum_iter,
                'num_eval_instance': num_instance,
                'train_type': train_type
            }
            log_data.update(average_meter_set.averages())
            log_data.update(kwargs)
            if doLog:
                if mode == 'val':
                    self.logger_service.log_val(log_data) #保存模型, 不仅保存当前模型, 同时保存最有模型; 索引是-1;
                elif mode == 'test':
                    self.logger_service.log_test(log_data) #保存最优模型;
                    self.saveGraphOutputTensor()
                else:
                    raise ValueError
        return log_data

    def _create_optimizer(self):
        args = self.args
        if args.optimizer.lower() == 'adam':
            betas = (args.adam_beta1, args.adam_beta2)
            return optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=betas)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError


    def _create_optimizer_total(self):
        args = self.args
        if args.optimizer.lower() == 'adam':
            betas = (args.adam_beta1, args.adam_beta2)
            # return optim.Adam(list(self.model.parameters()) + list(self.graph_model.parameters()) + list(self.graph_model_cate.parameters()), lr=args.lr, weight_decay=args.weight_decay, betas=betas)
            # return optim.Adam(list(self.model.parameters()) + list(self.graph_model.parameters()) + list(self.graph_model_kgat.parameters()), lr=args.lr, weight_decay=args.weight_decay, betas=betas)
            return optim.Adam(list(self.model.parameters()) + list(self.graph_model.parameters()), lr=args.lr, weight_decay=args.weight_decay, betas=betas)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(list(self.model.parameters()) + list(self.graph_model.parameters()), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError
    
    # def _create_graph_optimizer(self):
    #     args = self.args
    #     if args.optimizer.lower() == 'adam':
    #         betas = (args.adam_beta1, args.adam_beta2)
    #         return optim.Adam(self.graph_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=betas)
    #     elif args.optimizer.lower() == 'sgd':
    #         return optim.SGD(self.graph_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    #     else:
    #         raise ValueError

    # def _create_graph_cate_optimizer(self):
    #     args = self.args
    #     if args.optimizer.lower() == 'adam':
    #         betas = (args.adam_beta1, args.adam_beta2)
    #         return optim.Adam(self.graph_model_cate.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=betas)
    #     elif args.optimizer.lower() == 'sgd':
    #         return optim.SGD(self.graph_model_cate.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    #     else:
    #         raise ValueError

    def _create_loggers(self):
        """
        val_loggers包含至少两类, 一类中通用的记录结果, 另一类是用于保存模型, 保存模型分为两类, RecentModelLogger和BestModelLogger(分别表示保存当前模型与保存最优模型);
        """
        train_table_definitions = [
            ('train_log', ['epoch', 'loss'])
        ]
        val_table_definitions = [
            ('val_log', ['epoch'] + \
             ['NDCG@%d' % k for k in self.metric_ks] +
             ['Recall@%d' % k for k in self.metric_ks]),
        ]
        test_table_definitions = [
            ('test_log', ['epoch'] + \
             ['NDCG@%d' % k for k in self.metric_ks] +
             ['Recall@%d' % k for k in self.metric_ks]),
        ]

        train_loggers = [TableLoggersManager(args=self.args, export_root=self.local_export_root, table_definitions=train_table_definitions)]
        val_loggers = [TableLoggersManager(args=self.args, export_root=self.local_export_root, table_definitions=val_table_definitions)]
        test_loggers = [TableLoggersManager(args=self.args, export_root=self.local_export_root, table_definitions=test_table_definitions)]

        if self.local_export_root is not None:
            root = Path(self.local_export_root)
            model_checkpoint = root.joinpath('models')
            val_loggers.append(RecentModelLogger(model_checkpoint))
            val_loggers.append(BestModelLogger(model_checkpoint, metric_key=self.best_metric))

        if USE_WANDB:
            train_loggers.append(WandbLogger(table_definitions=train_table_definitions))
            val_loggers.append(WandbLogger(table_definitions=val_table_definitions, prefix='val_'))
            test_loggers.append(WandbLogger(table_definitions=test_table_definitions, prefix='test_'))

        return train_loggers, val_loggers, test_loggers

    def _create_state_dict(self, epoch, accum_iter):
        return {
            STATE_DICT_KEY: self.model.module.state_dict() if self.use_parallel else self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
            SCHEDULER_STATE_DICT_KEY: self.lr_scheduler.state_dict(),
            TRAIN_LOADER_DATASET_RNG_STATE_DICT_KEY: self.train_loader.dataset.get_rng_state(),
            TRAIN_LOADER_SAMPLER_RNG_STATE_DICT_KEY: self.train_loader.sampler.get_rng_state(),
            STEPS_DICT_KEY: (epoch, accum_iter),
        }

    def _restore_best_state(self):
        ### restore best epoch
        df_path = os.path.join(self.local_export_root, 'tables', 'val_log.csv')
        df = pd.read_csv(df_path)
        sat, reached_end = find_saturation_point(df, self.saturation_wait_epochs, display=False)
        e = sat['epoch'].iloc[0]
        self.best_epoch = e
        print('Restored best epoch:', self.best_epoch)

        ###
        state_dict_path = os.path.join(self.local_export_root, 'models', BEST_STATE_DICT_FILENAME)
        chk_dict = torch.load(os.path.abspath(state_dict_path))

        ### sanity check
        _e, _ = chk_dict[STEPS_DICT_KEY]
        assert e == _e

        ### load weights
        d = chk_dict[STATE_DICT_KEY]
        model_state_dict = {}
        # this is for stupid reason
        # pdb.set_trace()
        for k, v in d.items():
            if k.startswith('model.'):
                model_state_dict[k[6:]] = v
            else:
                model_state_dict[k] = v
        if self.use_parallel:
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)

        ### restore best metric
        val_log_data = self.validate(0, 0, mode='val', doLog=False)
        metric = val_log_data[self.best_metric]
        self.best_metric_at_best_epoch = metric
        print('Restored best metric:', self.best_metric_at_best_epoch)

    
    def _restore_best_state_model(self, train_type='pretrain', is_best=False):
        ### restore best epoch
        # df_path = os.path.join(self.local_export_root, 'tables', 'val_log.csv')
        # df = pd.read_csv(df_path)
        # sat, reached_end = find_saturation_point(df, self.saturation_wait_epochs, display=False)
        # e = sat['epoch'].iloc[0]
        # self.best_epoch = e
        # print('Restored best epoch:', self.best_epoch)

        ###
        if is_best:
            state_dict_path = os.path.join(self.local_export_root, 'models', BEST_STATE_DICT_FILENAME + '_' + train_type)
        else:
            state_dict_path = os.path.join(self.local_export_root, 'models', RECENT_STATE_DICT_FILENAME + '_' + train_type + '.final')
        chk_dict = torch.load(os.path.abspath(state_dict_path))
        
        ### sanity check
        # _e, _ = chk_dict[STEPS_DICT_KEY]
        # assert e == _e

        ### load weights
        d = chk_dict[STATE_DICT_KEY]
        model_state_dict = {}
        model_tmp = d
        # pdb.set_trace()
        # this is for stupid reason
        for k, v in d.items():
            if k.startswith('model.'):
                model_state_dict[k[6:]] = v
            else:
                model_state_dict[k] = v
        if self.use_parallel:
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)

        ### restore best metric
        # val_log_data = self.validate(0, 0, mode='val', doLog=False)
        # metric = val_log_data[self.best_metric]
        # self.best_metric_at_best_epoch = metric
        # print('Restored best metric:', self.best_metric_at_best_epoch)

    def _restore_training_state(self):
        self._restore_best_state()

        ###
        state_dict_path = os.path.join(self.local_export_root, 'models', RECENT_STATE_DICT_FILENAME)
        chk_dict = torch.load(os.path.abspath(state_dict_path))

        ### restore epoch, accum_iter
        epoch, accum_iter = chk_dict[STEPS_DICT_KEY]
        self.epoch_start = epoch + 1
        self.accum_iter_start = accum_iter

        ### restore train dataloader rngs
        train_loader_dataset_rng_state = chk_dict[TRAIN_LOADER_DATASET_RNG_STATE_DICT_KEY]
        self.train_loader.dataset.set_rng_state(train_loader_dataset_rng_state)
        train_loader_sampler_rng_state = chk_dict[TRAIN_LOADER_SAMPLER_RNG_STATE_DICT_KEY]
        self.train_loader.sampler.set_rng_state(train_loader_sampler_rng_state)

        ### restore model
        d = chk_dict[STATE_DICT_KEY]
        model_state_dict = {}
        # this is for stupid reason
        for k, v in d.items():
            if k.startswith('model.'):
                model_state_dict[k[6:]] = v
            else:
                model_state_dict[k] = v
        if self.use_parallel:
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)

        ### restore optimizer
        self.optimizer.load_state_dict(chk_dict[OPTIMIZER_STATE_DICT_KEY])

        ### restore lr_scheduler
        self.lr_scheduler.load_state_dict(chk_dict[SCHEDULER_STATE_DICT_KEY])

    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.train_batch_size and accum_iter != 0
