# -*- coding: utf-8 -*-
# 统合数据集，网络架构与损失函数进行训练
import os
import random

import numpy as np

from write_data import *
import pandas as pd
import torch
from torch.autograd import Function
from act_fun import *
from DataSet import *
import torch.nn as nn
#from network import *#避免重复引用
from network_for_eigvalue import *
from model import *
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
import math
import time
import torch.nn.functional as F

from datetime import datetime
from logging_tool import *
from loss_function import *


class evigal_number_model_for_eigvalue:
    def __init__(self, network, train_dataset, test_dataset, loss, optim, tho_generator, test_tho_generator=None,
                 subdomain_batch_size=64, device=None,
                 output_dim=20, scheduler=None, start_time="", prediction_model="discrete",test_epoch=10):
        self.start_time = start_time  # 记录实验开始时间
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.prediction_model = prediction_model  # discrete时输出与标签为离散类别，continue时输出与标签为0-1之间的连续值
        self.optim = optim
        self.subdomain_batch_size = subdomain_batch_size
        self.tho_generator = tho_generator  # 训练用的tho_generator
        self.test_epoch=int(test_epoch)#每隔self.test_epoch轮次在测试集上进行一次测试，以及在训练集上测试一些指标

        #test_tho_generator暂时注释掉，因为前向运算被整合到forward的函数中了，self.test_tho_generator实质上是无效的
        # if test_tho_generator is None:
        #     self.test_tho_generator = self.tho_generator  # 测试用的tho_generator,测试与训练用的可能不一致
        # else:
        #     self.test_tho_generator = test_tho_generator
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.network = network.to(self.device)
        self.output_dim = output_dim
        self.loss = loss.to(self.device)
        self.scheduler = scheduler
        self.mseloss = nn.MSELoss()

        # 用于数据记录与分析的数据
        # self.loss_history=[]
        self.epoch=-1#记录训练轮次，用于在test中更新各个指标的最优轮次

        self.best_train_loss = np.inf
        self.best_train_loss_epoch = -1
        self.best_train_acc = 0.0
        self.best_train_acc_epoch = -1

        self.best_test_acc = 0.0
        self.best_test_acc_epoch=-1
        self.best_test_map = 0.0
        self.best_test_map_epoch = -1
        self.test_best_log = None
        self.best_test_recall = 0.0
        self.best_test_precision = 0.0
        self.best_test_precision_epoch=-1
        self.test_loss = []
        self.best_test_loss=np.inf
        self.best_test_loss_epoch=-1

        self.test_mean_recall_history = []
        self.test_mean_precision_history = []
        self.test_map_history = []


        self.best_train_map = 0.0
        self.best_train_map_epoch = -1
        self.train_best_log = None
        self.best_train_recall = 0.0
        self.best_train_precision = 0.0
        self.train_loss = []
        self.train_acc = []
        self.test_acc_history = []
        self.train_mean_recall_history = []
        self.train_mean_precision_history = []
        self.train_map_history = []
    def forward(self,hdf5dataset,dataloader,global_dataset,subdomain_data, subdomain_name,test_metris,kappa=None):
        """
        对各个子区域的数据进行运算与处理labels封装为前向运算函数,方便进行统一的调用与修改
        :param hdf5dataset:HDF5Dataset_for_eigenvale_number类的数据
        :param dataloader:通过HDF5Dataset_for_eigenvale_number类的get_batches得到的数据加载器
        :param global_dataset:
        :param kappa:波数
        :return:outputs与labels
        """
        if kappa is None:
            kappa = global_dataset["kappa"]
        batch_size = len(subdomain_name)  # 获取实际加载时的batch
        for keys in subdomain_data:
            subdomain_data[keys] = subdomain_data[keys].to(self.device)
        with torch.no_grad():
            tho_batch = self.tho_generator.gettho(batch_size, kappa)

        # tho_batch=torch.zeros_like(tho_batch)
        labels = hdf5dataset.get_labels(subdomain_name, tho_batch).to(self.device)

        # print("测试节点： labels:",labels)
        # print("测试节点 labels的形状为",labels.shape)
        outputs = self.network(subdomain_data, global_dataset, tho_batch)
        # print("test point outputs is",outputs)
        if self.prediction_model == "discrete":
            labels = get_discrete_labels(labels=labels, ml=subdomain_data['m_l'],
                                         feature_dim=self.output_dim).to(self.device)
            # test_metris.update(outputs, labels)
        elif self.prediction_model == "continue":
            # print("测试节点 除以m_l前的labels' shape is ",labels.shape)
            # print("测试节点 除以m_l前的labels' values is ",labels)
            # print("测试节点 m_l's shape is ",subdomain_data['m_l'].shape)
            # print("测试节点 m_l's values is ",subdomain_data['m_l'])
            dis_labels = get_discrete_labels(labels=labels, ml=subdomain_data['m_l'],
                                             feature_dim=self.output_dim).to(self.device)
            labels = labels / (subdomain_data['m_l'].unsqueeze(-1))
            with torch.no_grad():
                # dis_outputs=outputs*(subdomain_data['m_l'].unsqueeze(-1))
                # print("-------------正确性测试 此时dis_outputs=labes,如若程序正确无误应当具有高准确性-----------")
                # dis_outputs = labels * (subdomain_data['m_l'].unsqueeze(-1))
                dis_outputs = get_discrete_labels_one_hot(labels=outputs, ml=subdomain_data['m_l'],
                                                          feature_dim=self.output_dim).to(self.device)
                # print("测试节点 dislabels is ",dis_labels)
                # print("测试节点 dis_outputs is ",dis_outputs)
                # print("测试节点 outputs is ",outputs)
                test_metris.update(dis_outputs, dis_labels)
        return outputs,labels
    def train(self, epochs, if_save_model=False, model_pth_dir="model.pth"):
        self.network.train()
        path_dir = os.path.dirname(model_pth_dir)
        print("训练的设备为: ", self.device)
        self.best_train_max_L2_relative_error=np.inf
        self.best_train_max_L2_relative_error_epoch=-1
        self.best_train_mean_L2_relative_error = np.inf
        self.best_train_mean_L2_relative_error_epoch=-1
        for epoch in range(epochs):
            try:
                self.network.alternating_train(epochs=epoch)
            except AttributeError:
                if epoch == 0:
                    print("alternating_train函数不存在")
            self.epoch=epoch
            loss_every_epoch = 0.0
            start_time_each_epoch = time.time()
            print("测试节点 目前的epoch", epoch)
            test_metris = MetricAccumulator(self.output_dim)
            indicies = list(range(self.train_dataset.__len__()))
            random.shuffle(indicies)
            for j in indicies:
                # print("测试节点，目前的dataset",j)
                hdf5dataset = self.train_dataset[j]
                dataloader = hdf5dataset.get_batches(if_shuffle=True)
                global_dataset = hdf5dataset.get_globaldata()
                for keys in global_dataset:
                    global_dataset[keys] = global_dataset[keys].to(self.device)
                kappa = global_dataset["kappa"]
                self.optim.zero_grad()
                loss = torch.tensor([0.0]).to(self.device)
                for subdomain_data, subdomain_name in dataloader:


                    outputs,labels=self.forward(hdf5dataset=hdf5dataset,dataloader=dataloader,global_dataset=global_dataset,subdomain_data=subdomain_data,
                                                subdomain_name=subdomain_name,test_metris=test_metris,kappa=kappa)
                    #-----原来没有整合到forward函数时的前向计算部分的代码--------------
                    # batch_size = len(subdomain_name)  # 获取实际加载时的batch
                    # for keys in subdomain_data:
                    #     subdomain_data[keys] = subdomain_data[keys].to(self.device)
                    # with torch.no_grad():
                    #     tho_batch = self.tho_generator.gettho(batch_size, kappa)
                    #
                    # # tho_batch=torch.zeros_like(tho_batch)
                    # labels = hdf5dataset.get_labels(subdomain_name, tho_batch).to(self.device)
                    #
                    # # print("测试节点： labels:",labels)
                    # # print("测试节点 labels的形状为",labels.shape)
                    # outputs = self.network(subdomain_data, global_dataset, tho_batch)
                    # # print("test point outputs is",outputs)
                    # if self.prediction_model == "discrete":
                    #     labels = get_discrete_labels(labels=labels, ml=subdomain_data['m_l'],
                    #                                  feature_dim=self.output_dim).to(self.device)
                    #     #test_metris.update(outputs, labels)
                    # elif self.prediction_model == "continue":
                    #     # print("测试节点 除以m_l前的labels' shape is ",labels.shape)
                    #     # print("测试节点 除以m_l前的labels' values is ",labels)
                    #     # print("测试节点 m_l's shape is ",subdomain_data['m_l'].shape)
                    #     # print("测试节点 m_l's values is ",subdomain_data['m_l'])
                    #     dis_labels=get_discrete_labels(labels=labels, ml=subdomain_data['m_l'],
                    #                                  feature_dim=self.output_dim).to(self.device)
                    #     labels = labels / (subdomain_data['m_l'].unsqueeze(-1))
                    #     with torch.no_grad():
                    #         #dis_outputs=outputs*(subdomain_data['m_l'].unsqueeze(-1))
                    #         # print("-------------正确性测试 此时dis_outputs=labes,如若程序正确无误应当具有高准确性-----------")
                    #         # dis_outputs = labels * (subdomain_data['m_l'].unsqueeze(-1))
                    #         dis_outputs=get_discrete_labels_one_hot(labels=outputs, ml=subdomain_data['m_l'],
                    #                                  feature_dim=self.output_dim).to(self.device)
                    #         # print("测试节点 dislabels is ",dis_labels)
                    #         # print("测试节点 dis_outputs is ",dis_outputs)
                    #         # print("测试节点 outputs is ",outputs)
                    #         test_metris.update(dis_outputs, dis_labels)
                    # -----原来没有整合到forward函数时的前向计算部分的代码结束--------------







                        # print("测试节点 除以m_l后的labels' shape is ", labels.shape)
                        # print("测试节点 除以m_l后的labels' values is ", labels)
                        # print("test point labels' shape is ",labels.shape)
                        # print("test point output's shape is ",outputs.shape)
                    #dis_labels_onehot=get_discrete_labels_one_hot(labels=labels, ml=subdomain_data['m_l'], feature_dim=self.output_dim).to(self.device)
                    # print("测试节点 outputs形状",outputs.shape)
                    # print("测试节点 ")

                    # print("测试节点 output形状为",outputs.shape)
                    # mseloss=self.mseloss(torch.argmax(outputs,dim=-1),dis_labels.to(torch.float32))
                    # outputs = outputs.permute(0, 2, 1)#用于协调以计算交叉熵损失函数

                    # print("测试节点:batch_size,tho_batch.size(1)",batch_size,tho_batch.size(1))
                    loss += self.loss(outputs, labels)

                loss_every_epoch += loss.item()
                loss.backward()
                self.optim.step()
                # del hdf5dataset
                # del dataloader
            acc = test_metris.accuracy
            mean_recall = test_metris.mean_recall
            mean_precision = test_metris.mean_precision
            map = test_metris.mean_average_precision
            self.train_loss.append(loss_every_epoch)
            self.train_acc.append(acc)
            self.train_mean_recall_history.append(mean_recall)
            self.train_mean_precision_history.append(mean_precision)
            self.train_map_history.append(map)
            if loss_every_epoch < self.best_train_loss:
                self.best_train_loss = loss_every_epoch
                self.best_train_loss_epoch = epoch
            if acc > self.best_train_acc:
                self.best_train_acc = acc
                self.best_train_acc_epoch = epoch
            if map > self.best_train_map:
                self.best_train_map = map
                self.best_train_map_epoch = epoch
            self.best_train_recall = self.best_train_recall if self.best_train_recall > mean_recall else mean_recall
            self.best_train_precision = self.best_train_precision if self.best_train_precision > mean_precision else mean_precision
            self.best_train_map = self.best_train_map if self.best_train_map > map else map
            self.scheduler.step()
            if (epoch % self.test_epoch == 0 and if_save_model)or epoch==(epochs-1):
                torch.save(self.network.state_dict(), model_pth_dir)
            if (epoch % self.test_epoch == 0)or epoch==(epochs-1):
                #with torch.no_grad()
                self.test()

                #-------------测试在训练集上的相对L2误差------------
                train_mean_raletive_L2_error = 0.0  # 训练集上所有子区域的平均L2相对误差
                train_max_raletiver_L2_error = 0.0  # 训练集上所有子区域的最大L2相对误差
                criterion = nn.MSELoss(reduction='none')  # 不降维的MSE用于计算L2相对误差
                indicies = list(range(self.train_dataset.__len__()))
                random.shuffle(indicies)
                for j in indicies:
                    # print("测试节点，目前的dataset",j)
                    hdf5dataset = self.train_dataset[j]
                    dataloader = hdf5dataset.get_batches(if_shuffle=True)
                    global_dataset = hdf5dataset.get_globaldata()
                    for keys in global_dataset:
                        global_dataset[keys] = global_dataset[keys].to(self.device)
                    kappa = global_dataset["kappa"]

                    for subdomain_data, subdomain_name in dataloader:
                        with torch.no_grad():
                            outputs, labels = self.forward(hdf5dataset=hdf5dataset, dataloader=dataloader,
                                                           global_dataset=global_dataset, subdomain_data=subdomain_data,
                                                           subdomain_name=subdomain_name, test_metris=test_metris,
                                                           kappa=kappa)
                            L2_error = criterion(outputs, labels)
                            L2_error = L2_error.sum(dim=1)  # 求和得到$\sum_{i=1}{M}|y_{real}(x_i)-y_{pred}(x_i)|^2
                            L2_norm = torch.pow(input=labels, exponent=2)
                            L2_norm = L2_norm.sum(dim=1)
                            relative_L2_norm = L2_error / L2_norm
                            train_mean_raletive_L2_error += (torch.sqrt(relative_L2_norm).sum()).item()
                            max_relative_L2_norm = math.sqrt((relative_L2_norm.max()).item())
                            train_max_raletiver_L2_error = max_relative_L2_norm if max_relative_L2_norm > train_max_raletiver_L2_error else train_max_raletiver_L2_error
                # 在所有子域上进行平均
                train_mean_raletive_L2_error = train_mean_raletive_L2_error / self.train_dataset.number_of_domain

                print(
                    f"训练集上 最大相对L2误差为{train_max_raletiver_L2_error},平均L2相对误差为{train_mean_raletive_L2_error}")
                self.best_train_max_L2_relative_error_epoch=self.best_train_max_L2_relative_error_epoch if train_max_raletiver_L2_error>self.best_train_max_L2_relative_error else epoch
                self.best_train_max_L2_relative_error=self.best_train_max_L2_relative_error if train_max_raletiver_L2_error>self.best_train_max_L2_relative_error else train_max_raletiver_L2_error
                self.best_train_mean_L2_relative_error_epoch = self.best_train_mean_L2_relative_error_epoch if train_mean_raletive_L2_error > self.best_train_mean_L2_relative_error else epoch
                self.best_train_mean_L2_relative_error = self.best_train_mean_L2_relative_error if train_mean_raletive_L2_error > self.best_train_mean_L2_relative_error else train_mean_raletive_L2_error


                self.network.train()
                try:
                    if epoch>0:
                        self.network.update()
                except AttributeError:
                    if epoch==0:
                        print("update函数不存在")
            # if epoch % 5 == 0 and self.loss.if_update_weights:
            #     weights = test_metris.weights_for_celoss.to(self.device)
            #     self.loss.update_weight(weights, self.device)
            end_time_each_epoch = time.time()
            print('该轮次训练耗时为:', end_time_each_epoch - start_time_each_epoch)
            print("该轮总的训练损失为:", loss_every_epoch)
            if self.prediction_model == "discrete":
                print("测试的准确度为:", acc)
                print("测试的平均召回率为:", test_metris.mean_recall)
                print("测试的召回率为：", test_metris.recall_per_class)
                print("测试的精确度为：", test_metris.precision_per_class)
                print("测试的平均精确度为：", mean_precision)
                print("测试的mAP为", map)
                print("每类标签出现的次数:", test_metris.real_class_numbers)
                print("每类标签预测出现的次数", test_metris.pre_class_numbers)
            else:
                print("测试的准确度为:", acc)
                print("测试的平均召回率为:", test_metris.mean_recall)
                print("测试的召回率为：", test_metris.recall_per_class)
                print("测试的精确度为：", test_metris.precision_per_class)
                print("测试的平均精确度为：", mean_precision)
                print("测试的mAP为", map)
                print("每类标签出现的次数:", test_metris.real_class_numbers)
                print("每类标签预测出现的次数", test_metris.pre_class_numbers)
        print(f"在第{self.best_train_loss_epoch}轮次到达最优的训练损失误差{self.best_train_loss}")
        print(f"在第{self.best_train_acc_epoch}轮次到达最优的准确度为{self.best_train_acc}")
        print(
            f"在所有轮次中所得到的最优mAP为{self.best_train_map},最优的平均召回率为{self.best_train_recall},最优的平均准确率为{self.best_train_precision}")
        print(f"在测试集上第{self.best_test_acc_epoch}轮次得到的最优准确度为{self.best_test_acc},在第{self.best_test_loss_epoch}达到最优损失{self.best_test_loss}")

        self.train_log = pd.DataFrame({"train_loss": self.train_loss})
        self.test_log=pd.DataFrame({"test_loss": self.train_loss})
        # self.train_best_log=pd.DataFrame("best loss":[self.be])
        if if_save_model:
            self.save_log(path_dir)
            self.plt_history(if_show_plt=False, save_dir=path_dir)

    def test(self):
        self.network.eval()
        self.eval_metris = MetricAccumulator(num_classes=self.output_dim)

        test_loss_epoch=0.0
        test_mean_raletive_L2_error=0.0#测试集上所有子区域的平均L2相对误差
        test_max_raletiver_L2_error=0.0#测试集上所有子区域的最大L2相对误差
        criterion = nn.MSELoss(reduction='none')#不降维的MSE用于计算L2相对误差
        for j in range(self.test_dataset.__len__()):
            # print("测试节点，目前的dataset",j)
            hdf5dataset = self.test_dataset[j]
            dataloader = hdf5dataset.get_batches()
            global_dataset = hdf5dataset.get_globaldata()
            for keys in global_dataset:
                global_dataset[keys] = global_dataset[keys].to(self.device)
            kappa = global_dataset["kappa"]
            #self.optim.zero_grad()
            test_loss = torch.tensor([0.0]).to(self.device)
            for subdomain_data, subdomain_name in dataloader:
                # -----原来没有整合到forward函数时的前向计算部分的代码--------------
                # batch_size = len(subdomain_name)  # 获取实际加载时的batch
                # for keys in subdomain_data:
                #     subdomain_data[keys] = subdomain_data[keys].to(self.device)
                #
                # tho_batch = self.test_tho_generator.gettho(batch_size, kappa)
                #
                # # tho_batch=torch.zeros_like(tho_batch)
                # labels = hdf5dataset.get_labels(subdomain_name, tho_batch).to(self.device)
                #
                # # print("测试节点： labels:",labels)
                # # print("测试节点 labels的形状为",labels.shape)
                # with torch.no_grad():
                #     outputs = self.network(subdomain_data, global_dataset, tho_batch)
                # # outputs=torch.argmax(outputs,dim=-1)
                # #dis_labels = get_discrete_labels(labels=labels, ml=subdomain_data['m_l'],
                # #                                 feature_dim=self.output_dim).to(self.device)
                # # print("测试节点 dislabels为",dis_labels)
                # if self.prediction_model == "discrete":
                #     labels = get_discrete_labels(labels=labels, ml=subdomain_data['m_l'],
                #                                  feature_dim=self.output_dim).to(self.device)
                #
                #     self.eval_metris.update(preds=outputs, targets=labels)
                # elif self.prediction_model == "continue":
                #     dis_labels = get_discrete_labels(labels=labels, ml=subdomain_data['m_l'],
                #                                      feature_dim=self.output_dim).to(self.device)
                #
                #     with torch.no_grad():
                #         # dis_outputs=outputs*(subdomain_data['m_l'].unsqueeze(-1))
                #         # print("-------------正确性测试 此时dis_outputs=labes,如若程序正确无误应当具有高准确性-----------")
                #         # dis_outputs = labels * (subdomain_data['m_l'].unsqueeze(-1))
                #         dis_outputs = get_discrete_labels_one_hot(labels=outputs, ml=subdomain_data['m_l'],
                #                                                   feature_dim=self.output_dim).to(self.device)
                #         labels = labels / (subdomain_data['m_l'].unsqueeze(-1))
                        #self.eval_metris.update(dis_outputs, dis_labels)
                # -----原来没有整合到forward函数时的前向计算部分的代码结束--------------
                # print("测试节点 output形状为",outputs.shape)
                # outputs = outputs.permute(0, 2, 1)  # 用于协调以计算交叉熵损失函数
                with torch.no_grad():
                    outputs,labels=self.forward(hdf5dataset=hdf5dataset,dataloader=dataloader,global_dataset=global_dataset,subdomain_data=subdomain_data,
                                                subdomain_name=subdomain_name,test_metris=self.eval_metris,kappa=kappa)
                    test_loss+=self.loss(outputs,labels)
                    #print(f"测试节点 outputs' shape is {outputs.shape},labels' shape is {labels.shape}")
                    L2_error=criterion(outputs,labels)
                    L2_error=L2_error.sum(dim=1)#求和得到$\sum_{i=1}{M}|y_{real}(x_i)-y_{pred}(x_i)|^2
                    L2_norm=torch.pow(input=labels,exponent=2)
                    L2_norm=L2_norm.sum(dim=1)
                    relative_L2_norm=L2_error/L2_norm
                    test_mean_raletive_L2_error+=(torch.sqrt(relative_L2_norm).sum()).item()
                    max_relative_L2_norm=math.sqrt((relative_L2_norm.max()).item())
                    test_max_raletiver_L2_error=max_relative_L2_norm if max_relative_L2_norm>test_max_raletiver_L2_error else test_max_raletiver_L2_error


                    # test_loss_MSE=nn.MSELoss()
                    # test_loss+=test_loss_MSE(outputs,labels)
            test_loss_epoch+=test_loss.item()

        #在所有子域上进行平均
        test_mean_raletive_L2_error=test_mean_raletive_L2_error/self.test_dataset.number_of_domain

        print(f"测试集上 最大相对L2误差为{test_max_raletiver_L2_error},平均L2相对误差为{test_mean_raletive_L2_error}")
        print("在测试集中的损失函数为: ",test_loss_epoch)
        self.test_loss.append(test_loss_epoch)
        # if test_loss<self.best_test_loss:
        #     self.best_test_loss=test_loss
        #     self.best_test_loss_epoch=self.epoch
        if self.best_test_loss>test_loss_epoch:
            self.best_test_loss=test_loss_epoch
            self.best_test_loss_epoch=self.epoch
        if self.prediction_model == "discrete":
            print("在测试集中的准确度为：", self.eval_metris.accuracy)
            self.test_acc_history.append(self.eval_metris.accuracy)
            if self.eval_metris.accuracy > self.best_test_acc:
                self.best_test_acc = self.eval_metris.accuracy
        elif self.prediction_model == "continue":
            print("在测试集中的准确度为：", self.eval_metris.accuracy)
            print("在测试集中的平均召回率为 ",self.eval_metris.mean_recall)
            print("在测试集中的平均精度为 ", self.eval_metris.mean_precision)
            print("在测试集中的mAP为",self.eval_metris.mean_average_precision)
            self.test_acc_history.append(self.eval_metris.accuracy)
            self.test_map_history.append(self.eval_metris.mean_average_precision)
            self.test_mean_precision_history.append(self.eval_metris.mean_precision)
            self.test_mean_recall_history.append(self.eval_metris.mean_recall)
            if self.eval_metris.accuracy > self.best_test_acc:
                self.best_test_acc = self.eval_metris.accuracy
                self.best_test_acc_epoch=self.epoch
            if self.eval_metris.mean_recall>self.best_test_map:
                self.best_test_map_epoch=self.epoch
                self.best_test_map=self.eval_metris.mean_recall
            if self.eval_metris.mean_precision>self.best_test_precision:
                self.best_test_precision=self.eval_metris.mean_precision
                self.best_test_precision_epoch=self.epoch

        # 可视化混淆矩阵
        # class_names=[]
        # for i in range(self.output_dim):
        #     class_names.append(str(i))
        # plot_confusion_matrix(self.eval_metris.confusion,class_names=class_names)

    def save_log(self, path):
        self.train_log.to_csv(os.path.join(path, f'{self.start_time}train_loss_log.csv'))
        self.test_log.to_csv(os.path.join(path, f'{self.start_time}test_loss_log.csv'))
        self.train_best_log = pd.DataFrame(
            {"best_loss_epoch": [self.best_train_loss_epoch], "loss": [self.best_train_loss],
             'best_train_acc_epoch': [self.best_train_acc_epoch], 'best_train_acc': [self.best_train_acc],
             'best_test_acc': [self.best_test_acc]})
        self.train_best_log.to_csv(os.path.join(path, 'train_best_log.csv'))
        self.test_acc_log = pd.DataFrame({"test_acc": self.test_acc_history})
        self.test_acc_log.to_csv(os.path.join(path, 'test_acc_log.csv'))
        # print("测试节点 调用数据成功")
        # self.train_log.to_csv(os.path.join((path,"train_log.csv")))

    def plt_history(self, if_show_plt=False, save_dir=None):
        epochs = len(self.train_acc)
        epoch = range(epochs)
        plt.figure(figsize=(8, 4))
        plt.plot(epoch, self.train_map_history, label='map', color='blue', linestyle='-', linewidth=2)
        plt.plot(epoch, self.train_mean_precision_history, label='mean_precision', color='red', linestyle='-',
                 linewidth=2)
        plt.plot(epoch, self.train_mean_recall_history, label='mean_recall', color='black', linestyle='-',
                 linewidth=2)
        plt.plot(epoch, self.train_acc, label='mean_precision', color='yellow', linestyle='-',
                 linewidth=2)
        plt.title("index in training epochs")
        plt.xlabel("epochs")
        plt.ylabel("values")
        plt.grid(True)
        plt.legend()
        if save_dir is not None:
            save_path = os.path.join(save_dir, f"{self.start_time}training_data.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if if_show_plt:
            plt.show()
        epochs = len(self.test_loss)
        epoch = range(epochs)
        plt.figure(figsize=(8, 4))
        plt.plot(epoch, self.test_map_history, label='map', color='blue', linestyle='-', linewidth=2)
        plt.plot(epoch, self.test_mean_precision_history, label='mean_precision', color='red', linestyle='-',
                 linewidth=2)
        plt.plot(epoch, self.test_mean_recall_history, label='mean_recall', color='black', linestyle='-',
                 linewidth=2)
        # plt.plot(epoch, self.train_acc, label='mean_precision', color='yellow', linestyle='-',
        #          linewidth=2)
        plt.title("index in training epochs")
        plt.xlabel("epochs")
        plt.ylabel("values")
        plt.grid(True)
        plt.legend()
        if save_dir is not None:
            save_path = os.path.join(save_dir, f"{self.start_time}test_data.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if if_show_plt:
            plt.show()
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 记录实验开始时间
    logging.basicConfig(level=logging.INFO)

    #从命令行与参数文件中读取超参数与实验方法设置，base_config为最底层的参数设置，model_config为命令行传入参数，为当需要加载模型权重文件时(pth),模型权重文件对应的配置文件，通过读取对应的配置文件覆盖base_config里的相应参数实现加载模型的兼容性
    #model_modify_config，命令行参数，为在加载模型权重文件对应的配置的基础上若需要进一步修改将需要修改的参数的文件传入
    #最终得到参数的覆盖优先级：命令行传入参数最高，model_modify_config其次，model_config再次之，base_config文件最低
    base_parser = argparse.ArgumentParser()
    #base_parser.add_argument('--config', default='./configs/base_with_cnn_for_eigvalue.yaml', help="基础配置文件路劲")
    base_parser.add_argument('--config', default='./configs/test.yaml', help="基础配置文件路劲")
    print("------------------注意，现在是使用的test.yaml用于测试--------------------------")
    base_args, _ = base_parser.parse_known_args()
    base_config = load_config(base_args.config)

    # 主参数解析器
    parser = argparse.ArgumentParser(description="配置管理系统示例")
    # parser.add_argument("--config", type=str, default='./configs/base_with_cnn.yaml', help="基础配置文件路劲")
    # 一些常用的修改的参数配置
    parser.add_argument("--data.batch_size", type=int, help="训练的batch_size")
    parser.add_argument("--model.lr", type=float, help="覆盖学习率")
    parser.add_argument("--training.epochs", type=int, help="训练轮次")
    parser.add_argument("--model.tho_generator", type=str, help="tho生成器设置")
    parser.add_argument("--Experimental_purpose", type=str, help="实验目的")
    parser.add_argument("--Experimental_args", type=list, help="实验的参数")
    parser.add_argument("--model_pth", default="None", type=str, help="模型加载路劲,默认值为None,表示不加载模型权重")
    parser.add_argument("--model_config", default='None', type=str, help="模型权重文件对应的配置文件用于加载模型的")

    parser.add_argument("--preditor_network.update_c",  type=float,
                        help="温度系数的更新系数")  # 覆盖优先级：传入参数大于model_modify_config大于model_config大于base_parser中的config



    #parser.add_argument("--model_config", default='None', type=str, help="模型权重文件对应的配置文件用于加载模型的")# 收集现有参数
    existing = {opt for action in parser._actions for opt in action.option_strings}

    # 根据base_config中的Experimental_args中的元素动态添加命令行可传入的参数，并读取命令行的参数
    experimental_args = base_config.get('Experimental_args', [])
    for arg_path in experimental_args:
        option = f'--{arg_path}'
        if option in existing:
            continue #已经存在的参数不再添加

        value = get_nested_value(base_config, arg_path)
        if value is None:
            continue  # 路径无效则跳过

        # 确定参数类型
        if isinstance(value, bool):
            arg_type = str_to_bool
            # arg_type=bool
            # print(f"测试节点 option is {option},value is {value},arg_type is {arg_type}")
        elif isinstance(value, list):
            arg_type = lambda x: yaml.safe_load(x)
        else:
            arg_type = type(value)

        # 添加参数
        print(f"测试节点 value is {value}")
        print(f"测试节点 添加前的parser is {parser}")
        parser.add_argument(
            option,
            type=arg_type,#注：对字符型或者列表不能直接指定为bool或者list,在调用时实际上输入value为str，会使用arg_type(value)，进行类型转化，float,int,str能正确转化(例如float("1.2")=1.2),但bool会将所有非空字符串转化为true
            default=value,
            help=f'动态参数: {arg_path} (默认: {value})'
        )
        existing.add(option)
        print(f"测试节点 parser is {parser}")
        print(f"测试节点 parser参数为 is {parser.parse_args()}")

    args = parser.parse_args()

    cli_config = {}
    for arg_key, arg_value in vars(args).items():
        if arg_value is None:
            continue
        keys = arg_key.split('.')
        current = cli_config
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = arg_value

    print("test point args is ", args)
    if args.model_config != "None":

        if args.model_modify_config != "None":
            config_load = load_config(args.model_modify_config)
            print("test point config_load is ", config_load)
            cli_config = merge_configs(config_load, cli_config)
            print("test point cli_config is", cli_config)
            del config_load
        config_load = load_config(args.model_config)
        cli_config = merge_configs(config_load, cli_config)
    final_config = merge_configs(base_config, cli_config)  # 覆盖优先级：传入参数大于model_modify_config大于model_config大于base_config
    log_dir = setup_logging(final_config)
    log_filename = os.path.join(log_dir, f"output_{timestamp}.log")
    dual_output = DualOutput(log_filename)  # 准备获取控制台或者python输出作为日志文件
    # 重定向标准输出和错误输出,获取终端或者控制台的输出存储在log文件里
    sys.stdout = dual_output
    sys.stderr = dual_output

    logging.info("=" * 50)
    logging.info(" 程序启动 - 最终配置参数:")
    #logging.info(yaml.dump(final_config, default_flow_style=False))

    # 保存配置副本
    save_final_config(final_config, log_dir, timestamp)
    input_dim = 20  # 全局输入特征维度
    # d_model = final_config["network"]['d_model']# 主要特征维度
    # adaptive_dim=4
    # hidden_dim = 32  # 隐藏层维度
    output_dim = final_config["model"]["output_dim"]  # 预测输出维度
    test_epoch=final_config["model"]["test_epoch"]
    batch_size = final_config["training"]["batch_size"]  # 2 组数据
    k_list = final_config["model"]["k_list"]  # 每组数据的 tho 数量不同
    global_data_keys = final_config["data"]["global_data_keys"]
    subdomain_data_keys = final_config["data"]["subdomain_data_keys"]
    train_data_dir = final_config["data"]["train_dir"]
    test_data_dir = final_config["data"]["test_dir"]
    subdomain_network_information = final_config["subdomain_network"]
    weihgt_network_information = final_config["weight_network"]

    preditor_network_information = final_config["preditor_network"]
    # physics_network_information = final_config["physics_network"]
    # query_network_information = final_config["query_network"]
    feature_out_network_information=final_config["feature_output__network"]
    model_information = final_config["model"]
    weihgt_function_keys = weihgt_network_information["keys"]
    subdomain_information_keys = subdomain_network_information['keys']

    #physics_information_keys = physics_network_information['keys']
    if_save_model = True
    print(f"----------------------保存模型权重:{if_save_model}-----------------------")
    if_save_result = True
    epochs = final_config["training"]['epochs']
    save_model_pth = os.path.join(log_dir, f"test_model_{timestamp}.pth")
    # label_smoothing=0.1
    # cfg={
    #     "实验目的":
    # }
    # 创建模型
    # attention_network=AttentionModule(3*d_model)
    # print(
    #     "-------------------------------------测试节点目前测试不用sigmoid作为最后输出，采用tanh作为最后输出--------------------------------------------------")
    train_dataset = MultiHDF5Dataset_for_eigenvale_number(folder_path=train_data_dir,
                                                          file_prefix="data", global_data_keys=global_data_keys,
                                                          subdomain_data_keys=subdomain_data_keys,
                                                          batch_size=batch_size, if_big_data=final_config["data"]["if_big_data"])
    test_dataset = MultiHDF5Dataset_for_eigenvale_number(folder_path=test_data_dir,
                                                         file_prefix="data", global_data_keys=global_data_keys,
                                                         subdomain_data_keys=subdomain_data_keys, batch_size=batch_size, if_big_data=final_config["data"]["if_big_data"])
    # if final_config["attention_network"]["network"]=="MultiheadAttention"

    #读取不同网络的类型与参数进行组装网络
    if weihgt_network_information["network"] == "AdaptivePoolEncoder":
        weight_network = AdaptivePoolEncoder(
            feature_dim=weihgt_network_information[weihgt_network_information["network"]]["feature_dim"],
            output_dim=weihgt_network_information[weihgt_network_information["network"]]["output_dim"])
    elif weihgt_network_information["network"] == "AdaptiveCNN":
        weight_network = AdaptiveCNN(input_channels=1,
                                     output_size=weihgt_network_information[weihgt_network_information["network"]][
                                         "output_dim"],
                                     initial_cnn_arg=weihgt_network_information["AdaptiveCNN"]["initial_cnn_arg"],
                                     final_cnn_arg=weihgt_network_information["AdaptiveCNN"]["final_cnn_arg"],
                                     adaptive_layer_arg=weihgt_network_information["AdaptiveCNN"]["adaptive_layer_arg"],
                                     global_pool_size=weihgt_network_information["AdaptiveCNN"]["global_pool_size"])
    # weight_network=MLP(input_dim=len(weihgt_function_keys),output_dim=d_model,layer_number=2, layer_size=[16, 16])
    # query_network=MLP(input_dim=1,output_dim=d_model,layer_number=2,layer_size=[16,16])
    # query_network = ThoFeatureExtractor(d_model=3 * d_model, input_dim=2)

    if subdomain_network_information["network"]=="MLP":
        subdomain_network = MLP(input_dim=len(subdomain_information_keys),
                                output_dim=subdomain_network_information[subdomain_network_information["network"]][
                                    'output_dim'],
                                layer_number=subdomain_network_information[subdomain_network_information["network"]][
                                    'layer_number'],
                                layer_size=subdomain_network_information[subdomain_network_information["network"]][
                                    'layer_size'])
    elif subdomain_network_information["network"]=="FCResNet_Block":
        subdomain_network=FCResNet_Block(input_dim=len(subdomain_information_keys),
                                         output_dim=subdomain_network_information[subdomain_network_information["network"]][
                                    'output_dim'],
                                         layers=subdomain_network_information[subdomain_network_information["network"]]['layers'],
                                         layer_size=subdomain_network_information[subdomain_network_information["network"]]['layer_size'],
                                         if_batchnorm=subdomain_network_information[subdomain_network_information["network"]]['if_batchnorm'])
    feature_out_network_input_dim=subdomain_network_information[subdomain_network_information["network"]][
                                 'output_dim']+weihgt_network_information[weihgt_network_information["network"]]["output_dim"]
    if feature_out_network_information["network"]=="MLP":
        feature_out_network=MLP(input_dim=feature_out_network_input_dim,output_dim=model_information["feature_out_dim"],layer_number=feature_out_network_information['MLP']["layer_number"],layer_size=feature_out_network_information['MLP']["layer_size"],act_fun=feature_out_network_information['MLP']["act_fun"])
    elif feature_out_network_information["network"]=="FCResNet_Block":
        feature_out_network=FCResNet_Block(input_dim=feature_out_network_input_dim,
                                         output_dim=model_information["feature_out_dim"],
                                         layers=feature_out_network_information[feature_out_network_information["network"]]['layers'],
                                         layer_size=feature_out_network_information[feature_out_network_information["network"]]['layer_size'],
                                         if_batchnorm=feature_out_network_information[feature_out_network_information["network"]]['if_batchnorm'])
    # physicis_network = MLP(input_dim=len(physics_information_keys),
    #                        output_dim=physics_network_information["MLP"]["output_dim"],
    #                        layer_number=physics_network_information["MLP"]["layer_number"],
    #                        layer_size=physics_network_information["MLP"]["layer_size"])
    # atten_dim = physics_network_information["MLP"]["output_dim"] + \
    #             geometrial_network_information[geometrial_network_information["network"]]['output_dim'] + \
    #             weihgt_network_information[weihgt_network_information["network"]]["output_dim"]
    # attention_network = nn.MultiheadAttention(embed_dim=atten_dim,
    #                                           num_heads=final_config["attention_network"]["MultiheadAttention"][
    #                                               "num_heads"],
    #                                           batch_first=final_config["attention_network"]["MultiheadAttention"][
    #                                               "batch_first"])
    # query_network = MLP(input_dim=query_network_information["MLP"]["input_dim"], output_dim=atten_dim,
    #                     layer_number=query_network_information["MLP"]["layer_number"],
    #                     layer_size=query_network_information["MLP"]["layer_size"])
    # preditor_network = MLP(input_dim=atten_dim, output_dim=output_dim,
    #                        layer_number=preditor_network_information["MLP"]["layer_number"],
    #                        layer_size=preditor_network_information["MLP"]["layer_size"],
    #                        act_fun=preditor_network_information["MLP"]["act_fun"])

    # model = Feature_Number_Prediction_Network_with_subdomain_data(input_dim=input_dim, output_dim=output_dim,
    #                                                               weight_function_network=weight_network
    #                                                               , physics_information_network=physicis_network,
    #                                                               geometrical_information_network=geometrial_network,
    #                                                               query_information_network=query_network,
    #                                                               attention_network=attention_network,
    #                                                               predictor_network=preditor_network,
    #                                                               physics_information_keys=physics_information_keys,
    #                                                               weihgt_function_keys=weihgt_function_keys,
    #                                                               geometrial_information_keys=geometrial_information_keys)

    # model = Feature_Number_Prediction_Network_with_subdomain_data_with_weight_function_grid(input_dim=input_dim,
    #                                                                                         output_dim=output_dim,
    #                                                                                         weight_function_network=weight_network
    #                                                                                         ,
    #                                                                                         physics_information_network=physicis_network,
    #                                                                                         geometrical_information_network=geometrial_network,
    #                                                                                         query_information_network=query_network,
    #                                                                                         attention_network=attention_network,
    #                                                                                         predictor_network=preditor_network,
    #                                                                                         physics_information_keys=physics_information_keys,
    #                                                                                         weihgt_function_keys=weihgt_function_keys,
    #                                                                                         geometrial_information_keys=geometrial_information_keys)


    if model_information["model_network"]=="Step_function_network":
        preditor_network = CustomNet(feature_dim=model_information["feature_out_dim"],initial_c=preditor_network_information["initial_c"],update_c=preditor_network_information["update_c"],min_c=preditor_network_information["min_c"])
        model=Step_function_network(input_dim=input_dim,output_dim=output_dim,weight_function_network=weight_network
                                            ,feature_out_network=feature_out_network,predictor_network=preditor_network,weihgt_function_keys=['weight_function_grid'],subdomain_information_keys=subdomain_information_keys,subdomain_information_network=subdomain_network)
    elif model_information["model_network"]=="Step_function_network_with_omegann":
        omega_network_information = final_config["omega_network"]
        omega_network_information_keys = omega_network_information['keys']
        if omega_network_information["network"]=="MLP":
            omega_network = MLP(input_dim=len(omega_network_information_keys),
                                output_dim=model_information["feature_out_dim"],
                                layer_number=omega_network_information[omega_network_information["network"]][
                                    'layer_number'],
                                layer_size=omega_network_information[omega_network_information["network"]][
                                    'layer_size'], act_fun="leakyrelu")  # 后面需要将权重进行初始化，为了避免梯度消失，不使用relu
        elif omega_network_information["network"]=="FCResNet_Block":
            omega_network=FCResNet_Block(input_dim=len(omega_network_information_keys),
                                         output_dim=model_information["feature_out_dim"],
                                         layers=omega_network_information[omega_network_information["network"]]['layers'],
                                         layer_size=omega_network_information[omega_network_information["network"]]['layer_size'],
                                         if_batchnorm=omega_network_information[omega_network_information["network"]]['if_batchnorm'])

        init_last_linear_zero_weight_bias_one(model=omega_network)#将omega_network最后一层权重初始化为0偏置初始化为1，初始输出为1,经过softmax后得到最终初始\omega输出为-1/feature_dim
        preditor_network=CustomNet_with_omiga(feature_dim=model_information["feature_out_dim"],initial_c=preditor_network_information["initial_c"],update_c=preditor_network_information["update_c"],min_c=preditor_network_information["min_c"])
        if_train_independ=model_information["if_train_independ"]  # Step_function_network_with_omegann是否启用交替训练
        alternating_epoch=model_information["alternating_epoch"]  # Step_function_network_with_omegann
        model=Step_function_network_with_omegann(input_dim=input_dim,output_dim=output_dim,weight_function_network=weight_network,omega_network=omega_network,omega_information_keys=omega_network_information_keys
                                            ,feature_out_network=feature_out_network,predictor_network=preditor_network,weihgt_function_keys=['weight_function_grid'],subdomain_information_keys=subdomain_information_keys,subdomain_information_network=subdomain_network,if_train_independ=if_train_independ,alternating_epoch=alternating_epoch)
    if args.model_pth != "None":
        model.load_state_dict(torch.load(args.model_pth))
        print(f"加载路劲{args.model_pth}的权重成功")
    # 设置损失函数
    if model_information["loss"] == "cross_entropy_loss":
        loss = cross_entropy_loss(label_smoothing=final_config["loss"][model_information["loss"]]["label_smoothing"],
                                  if_update_weights=final_config["loss"][model_information["loss"]][
                                      "if_update_weights"])
    elif model_information["loss"] == "cross_entropy_loss_with_mse":
        loss = cross_entropu_loss_with_mse(
            label_smoothing=final_config["loss"][model_information["loss"]]["label_smoothing"],
            alpha=final_config["loss"][model_information["loss"]]["alpha"],
            if_update_weights=final_config["loss"][model_information["loss"]]["if_update_weights"])
    elif model_information["loss"] == "MSE_Loss":
        loss = MSE_Loss()
    # 测试网络架构与原来是否一致
    # state_dict = torch.load(r"./logs/测试网络结构的通畅性与加入区域的几何节点信息后学习率对网络训练稳定度的影响/model.lr=0.0005/test_model_20250423_182117.pth", map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    # print("Model expects:", model.state_dict().keys())
    #
    # # 打印权重文件的键
    # print("File contains:", state_dict.keys())
    # model.load_state_dict(state_dict)

    #tho_generator=Tho_generator_uniform(beta=-0.6,klist=k_list)
    if model_information["tho_generator"] == "Tho_generator_uniform_xiugai":
        tho_generator = Tho_generator_uniform_xiugai(
            beta=final_config["tho_generator"]["Tho_generator_uniform_xiugai"]["beta"], klist=k_list,
            low_rate=final_config["tho_generator"]["Tho_generator_uniform_xiugai"]["low_rate"],
            up_rate=final_config["tho_generator"]["Tho_generator_uniform_xiugai"]["up_rate"],
            mean_rate=final_config["tho_generator"]["Tho_generator_uniform_xiugai"]["mean_rate"])
    elif model_information["tho_generator"] == "Tho_generator_network":
        tho_generator_inforamtion = final_config["tho_generator"]["Tho_generator_network"]
        if tho_generator_inforamtion["network"] == "MLP":
            tho_network = MLP(input_dim=tho_generator_inforamtion["MLP"]["input_dim"],
                              output_dim=tho_generator_inforamtion['MLP']['output_dim'],
                              layer_size=tho_generator_inforamtion["MLP"]["layer_size"],
                              layer_number=tho_generator_inforamtion["MLP"]["layer_number"],
                              act_fun=tho_generator_inforamtion["MLP"]["act_fun"],
                              pow_k=tho_generator_inforamtion["MLP"]["pow_k"])
    elif model_information["tho_generator"] == "Tho_generator_uniform_0_1":#0-1上均匀随机数,用于测试
        tho_generator = Tho_generator_uniform_0_1(beta=0.0,klist=k_list)
    if model_information["test_tho_generator"] == "Tho_generator_uniform_xiugai":
        test_tho_generator = Tho_generator_uniform_xiugai(
            beta=final_config["tho_generator"]["Tho_generator_uniform_xiugai"]["beta"], klist=k_list,
            low_rate=final_config["tho_generator"]["Tho_generator_uniform_xiugai"]["low_rate"],
            up_rate=final_config["tho_generator"]["Tho_generator_uniform_xiugai"]["up_rate"],
            mean_rate=final_config["tho_generator"]["Tho_generator_uniform_xiugai"]["mean_rate"])
    elif model_information["test_tho_generator"] == "Tho_generator_network":
        test_tho_generator_inforamtion = final_config["tho_generator"]["Tho_generator_network"]
        if tho_generator_inforamtion["network"] == "MLP":
            tho_network = MLP(input_dim=tho_generator_inforamtion["MLP"]["input_dim"],
                              output_dim=tho_generator_inforamtion['MLP']['output_dim'],
                              layer_size=tho_generator_inforamtion["MLP"]["layer_size"],
                              layer_number=tho_generator_inforamtion["MLP"]["layer_number"],
                              act_fun=tho_generator_inforamtion["MLP"]["act_fun"],
                              pow_k=tho_generator_inforamtion["MLP"]["pow_k"])
    elif model_information["test_tho_generator"] == "Tho_generator_uniform_0_1":#0-1上均匀随机数,用于测试
        test_tho_generator = Tho_generator_uniform_0_1(beta=0.0,klist=k_list)
    elif model_information["test_tho_generator"] == "None":
        test_tho_generator=None
    optim = torch.optim.Adam(params=model.parameters(), lr=final_config["model"]["lr"])
    MultiStepLR_information = final_config["scheduler"]["MultiStepLR"]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optim,
        milestones=MultiStepLR_information["milestones"],  # 指定衰减节点
        gamma=MultiStepLR_information["gamma"]
    )
    test_model = evigal_number_model_for_eigvalue(network=model, train_dataset=train_dataset, test_dataset=test_dataset, loss=loss,
                                     optim=optim, tho_generator=tho_generator, test_tho_generator=test_tho_generator,output_dim=output_dim,
                                     scheduler=scheduler, start_time=timestamp,
                                     prediction_model=final_config["model"]["prediction"],
                                     test_epoch=test_epoch)
    test_model.train(epochs, if_save_model=if_save_model, model_pth_dir=save_model_pth)
    test_model.save_log(path=log_dir)

    # torch.save(model,"test_model.pth")
    if if_save_result:

        # 记录本次实验数据
        experiment_path = os.path.join(final_config["logging"]["log_dir"], final_config["Experimental_purpose"],
                                       "最优数据汇总.csv")
        args_path = os.path.join(log_dir, "最优数据汇总.csv")
        heads = ['实验时间']
        data = [timestamp]
        for item in final_config['Experimental_args']:
            heads.append(item)
            keys = item.split(".")
            data_item = final_config[keys[0]]
            for i in range(len(keys) - 1):
                data_item = data_item[keys[i + 1]]
            data.append(data_item)
        heads += ['配置文件保存路劲', '总共训练轮次', '最优训练损失轮次', '最优训练损失', '最优训练准确度轮次',
                  '最优训练准确度', '在测试集上的最优测试准确度', "最优平均召回率", "最优平均准确度", "最优mAP","训练最优平均L2相对误差","训练最优平均L2相对误差轮次","训练最优最大L2相对误差","训练最优最大L2相对误差轮次","测试集上最优mAP","测试集上最优mAP轮次","测试集上的最优损失函数","测试集上最优损失函数轮次"]
        data += [log_dir, final_config["training"]['epochs'], test_model.best_train_loss_epoch,
                 test_model.best_train_loss, test_model.best_train_acc_epoch,
                 test_model.best_train_acc, test_model.best_test_acc, test_model.best_train_recall,
                 test_model.best_train_precision, test_model.best_train_map,test_model.best_train_mean_L2_relative_error,test_model.best_train_mean_L2_relative_error_epoch,
                 test_model.best_train_max_L2_relative_error,test_model.best_train_max_L2_relative_error_epoch,test_model.best_test_map,test_model.best_test_map_epoch,test_model.best_test_loss,test_model.best_test_loss_epoch]
        record_data(experiment_path, headers=heads, data=data)
        record_data(args_path, headers=heads, data=data)  # 记录同样参数下训练用于求平均，避免随机性带来的误差

        # 恢复标准输出
        sys.stdout = dual_output.stdout
        sys.stderr = dual_output.stderr
        dual_output.file.close()
        print(f"\n日志已保存至: {log_filename}")

    end_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    time_cost = datetime.strptime(end_time, "%Y%m%d_%H%M%S") - datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
    print(f"本次实验总计耗时{time_cost}")
    # else:
    #     model.test(if_show_png=cfg['是否展示实验结果的图片'])
    # if cfg['是否需要记录此次网络权重']:
    #     torch.save(model.network, path + '\model_pro.pth')

















