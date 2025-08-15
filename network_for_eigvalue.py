# -*- coding: utf-8 -*-

import torch
from network import *
#这里写入一些针对特征值预测的网络

def set_network_trainable(network: nn.Module, unfreeze: bool):
    """
    冻结或解冻整个网络的权重参数。

    参数:
    - network (nn.Module): 要操作的整个神经网络
    - unfreeze (bool): 如果为 True，解冻权重；如果为 False，冻结权重
    """
    for param in network.parameters():
        param.requires_grad = unfreeze
class CustomNet(nn.Module):
    def __init__(self, feature_dim,initial_c=1.0,update_c=0.5,min_c=1/256):
        super().__init__()
        self.feature_dim = feature_dim
        self.omega = nn.Parameter(-torch.ones(feature_dim)/feature_dim)  # [feature_dim]#初始化为-1/feature
        self.b = torch.tensor(1.0)              # 标量偏置
        self.c=initial_c
        self.update_c=update_c
        self.min_c=min_c
        self.train_c=1.0
        self.test_c=100.0
    def forward(self, T, tho):
        """

        :param T:[batch, feature_dim]
        :param tho:[batch, klist,1]
        :return: out_{ij}=b+\sum_{k=1}^{feature_dim}\omega_k sigmoid((tho_{ij}-T_{ik})/self.c))
        """
        # T: [batch, feature_dim], tho: [batch, klist], c: 标量常数
        #
        T_exp = T.unsqueeze(1)         # [batch, 1, feature_dim]
        #tho_exp = tho.unsqueeze(2)     # [batch, klist, 1]
        # diff = (tho_exp - T_exp) / self.c   # [batch, klist, feature_dim]
        diff = (tho- T_exp) / self.c
        sig = torch.sigmoid(diff)
        weighted = sig * self.omega    # 广播: [batch, klist, feature_dim]
        summed = weighted.sum(dim=-1)  # [batch, klist]
        return self.b + summed
    def update(self):
        self.c=max(self.c*self.update_c,self.min_c)

class CustomNet_with_omiga(nn.Module):
    def __init__(self,  feature_dim, initial_c=1.0, learnable_c=False,update_c=2.0,min_c=1/256):
        super().__init__()

        self.feature_dim = feature_dim
        # self.b = torch.tensor(1.0)
        self.b = nn.Parameter(torch.tensor(1.0))
        self.para=nn.Parameter(torch.tensor(1.0))#形式参数，nn.Module至少要有一个参数
        self.update_c=update_c
        self.min_c=torch.tensor(min_c)
        self.c=initial_c
        #self.c=nn.Parameter(torch.tensor(initial_c))
        # if learnable_c==False:
        #     set_network_trainable(self, unfreeze=False)#冻结b与c，不进行梯度更新
        # if learnable_c:
        #     self.c = nn.Parameter(torch.tensor(initial_c))
        # else:
        #     self.register_buffer('c', torch.tensor(initial_c))

    def forward(self, T,omega, tho):
        # x: 输入用于生成 T 和 omega
        # tho: 外部输入 [batch, klist, 1]

        # T = self.T_net(x)  # [batch, feature_dim]
        # omega = self.omega_net(x)  # [batch, feature_dim]

        T_exp = T.unsqueeze(1)  # [batch, 1, feature_dim]
        omega_exp = omega.unsqueeze(1)  # [batch, 1, feature_dim]
        tho_exp = tho  # [batch, klist, 1]

        diff = (tho_exp - T_exp) / self.c  # [batch, klist, feature_dim]
        sig = torch.sigmoid(diff)  # [batch, klist, feature_dim]
        weighted = sig * omega_exp  # [batch, klist, feature_dim]
        summed = weighted.sum(dim=-1)  # [batch, klist]
        return self.b + summed
    def update(self):
        #self.c=torch.max(self.c*self.update_c,self.min_c)
        self.c=max(self.c*self.update_c,self.min_c)
class Step_function_network(nn.Module):#用sigmoid拟合阶梯函数的网络
    def __init__(self,input_dim,output_dim,weight_function_network,subdomain_information_network,predictor_network,feature_out_network,device=None,
                 weihgt_function_keys=['weight_function_grid'],subdomain_information_keys=['H','h','m_l']):


        super(Step_function_network, self).__init__()
        if device is None:
            self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device=device
        #print(self.device)
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.weight_function_network=weight_function_network

        self.subdomain_information_network=subdomain_information_network
        # self.query_information_network=query_information_network
        # self.attention_network=attention_network
        self.feature_out_network=feature_out_network#用于整合信息，输出特征值的最终分割值
        self.predictor_network=predictor_network
        # print("----------------测试冻结omega与b----------------------")
        # set_network_trainable(self.predictor_network,unfreeze=False)

        self.weight_function_keys=weihgt_function_keys
        # self.physics_information_keys=physics_information_keys
        self.subdomain_information_keys=subdomain_information_keys
    def forward(self, subdomain_data,global_data,tho_batch):
        weight_function=subdomain_data[self.weight_function_keys[0]]# (batch_size, H, W)
        weight_function=weight_function.unsqueeze(1)# 增加chennel维度(batch_size,1, H, W)
        # test_process=True#测试用
        # if len(weight_function.shape)==1:
        #     weight_function=weight_function.unsqueeze(1)
        #print("测试节点 weight_function的维度为",weight_function.shape)
        batch_size=tho_batch.size(0)
        #physics_information=torch.stack([global_data[name] for name in self.physics_information_keys],dim=-1)
        #print("测试节点：",[global_data[name] for name in self.physics_information_keys])
        # physics_information=torch.cat([global_data[name].unsqueeze(0) if global_data[name].dim() == 0 else global_data[name]
        #     for name in self.physics_information_keys],dim=0)#形状为[input_dim]
        # physics_information=physics_information.unsqueeze(dim=0)#形状为[1,input_dim]
        # physics_information=physics_information.expand(batch_size,-1)#形状为[batch_size,input_dim]
        #print("测试节点 physics_information的维度为：",physics_information.shape)
        # geometrial_information=torch.cat([global_data[name].unsqueeze(0) if global_data[name].dim() == 0 else global_data[name]
        #     for name in self.geometrial_information_keys],dim=0).unsqueeze(dim=0)
        sub_information = torch.cat(
            [
                global_data[name].unsqueeze(0) if global_data[name].dim() == 0 else global_data[name]
                for name in self.subdomain_information_keys
                if name in global_data  # 跳过不存在的键
            ],
            dim=0
        ).unsqueeze(dim=0)
        sub_information=sub_information.expand(batch_size,-1)
        # if any(key in subdomain_data for key in self.geometrial_information_keys):
        #     geometrial_information_subdata=torch.cat(
        #         [
        #             subdomain_data[name].unsqueeze(0) if subdomain_data[name].dim() == 0 else subdomain_data[name]
        #             for name in self.geometrial_information_keys
        #             if name in subdomain_data  # 跳过不存在的键
        #         ],
        #         dim=-1
        #     ).unsqueeze(dim=1)
        #     print("测试节点 geometrial_information_subdata形状：",geometrial_information_subdata.shape)
        #     print("测试节点 geometrial_information形状: ",geometrial_information.shape )
        #     geometrial_information=torch.cat([geometrial_information_subdata,geometrial_information],dim=-1)
        if any(key in subdomain_data for key in self.subdomain_information_keys):
            sub_information_subdata=torch.cat(
                [
                    subdomain_data[name].unsqueeze(1)
                    for name in self.subdomain_information_keys
                    if name in subdomain_data  # 跳过不存在的键
                ],
                dim=-1
            )
            # print("测试节点 geometrial_information_subdata形状：",geometrial_information_subdata.shape)
            # print("测试节点 geometrial_information形状: ",geometrial_information.shape )
            sub_information=torch.cat([sub_information_subdata,sub_information],dim=-1)
            #print("测试节点，sub_information最终的shape为：",sub_information.shape)

        # weight_function=weight_function
        # query_information=query_information
        # geometrial_information=geometrial_information
        F_weight=self.weight_function_network(weight_function)
        # print(f"weight特征向量的形状为{F_weight.shape}")
        # F_physics=self.physics_information_network(physics_information)
        # print(f"physics特征向量的形状为{F_physics.shape}")
        F_sub_information=self.subdomain_information_network(sub_information)
        #print(f"weight特征向量的形状为{F_weight.shape}")
        #print(f"subdoamin_information特征向量的形状为{F_geometrial.shape}")
        F_subdomain=torch.cat([F_weight,F_sub_information],dim=1)
        #print(f"拼接的向量形状为：{F_subdomain.shape}")
        # F_subdomain = F_subdomain.unsqueeze(1)  # 变为 (batch_size, 1, d_model)
        # #print(f"拼接的向量形状为：{F_subdomain.shape}")
        # F_query=self.query_information_network(query_information)
        # print(f"query的特征向量的形状wield{F_query.shape}")
        F_feature=self.feature_out_network(F_subdomain)
        # print(f"predictor特征向量形状为{F_feature.shape}")
        # print(f"测试节点 tho_batch's shape is ",tho_batch.shape)
        output=self.predictor_network(F_feature,tho_batch)
        # print(f"output的形状为{output.shape}")
        # output=F.softmax(output,dim=-1)
        # print(f"经过softmax之后的形状{output.shape}")
        # output=torch.argmax(output,dim=-1)
        # print(f"最终输出的预测类别的形状为{output.shape}")
        return output
    def update(self,update=2.0):
        try:
            self.predictor_network.update()
        except AttributeError:
            print("网络更新异常")

    def alternating_train(self, epochs):#暂时用于站位与Step_function_network_with_omegann一致
        pass

class Step_function_network_with_omegann(nn.Module):#用sigmoid拟合阶梯函数的网络,相比与Step_function_network，权重Omegia由一个网络作为输出产生
    def __init__(self,input_dim,output_dim,weight_function_network,subdomain_information_network,predictor_network,feature_out_network,omega_network,device=None,
                 weihgt_function_keys=['weight_function_grid'],subdomain_information_keys=['H','h','m_l'],omega_information_keys=['H','h','m_l'],if_train_independ=False,alternating_epoch=-1):
        """

        :param input_dim: 输入维度
        :param output_dim: 输出维度
        :param weight_function_network:处理权重信息的网络
        :param subdomain_information_network: 处理子区域信息的网络
        :param predictor_network: 用于做最后预测的网络
        :param feature_out_network:将子区域信息特征与权重信息特征用于得到
        :param omega_network:
        :param device:运行设备
        :param weihgt_function_keys:权重信息的键的列表
        :param subdomain_information_keys:子区域信息的键的列表
        :param omega_information_keys:\omega网络输入信息的键的列表
        :param if_train_independ:是否采用交替单独训练omega_network的策略(训练omega_network参数时其余网络参数冻结，反之omega_network参数冻结，交替进行)
        :param alternating_epoch:交替轮次
        """


        super(Step_function_network_with_omegann, self).__init__()
        if device is None:
            self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device=device
        #print(self.device)
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.weight_function_network=weight_function_network

        self.subdomain_information_network=subdomain_information_network
        # self.query_information_network=query_information_network
        # self.attention_network=attention_network
        self.feature_out_network=feature_out_network#用于整合信息，输出特征值的最终分割值
        self.predictor_network=predictor_network
        # print("----------------测试冻结omega与b----------------------")
        # set_network_trainable(self.predictor_network,unfreeze=False)

        self.weight_function_keys=weihgt_function_keys
        # self.physics_information_keys=physics_information_keys
        self.subdomain_information_keys=subdomain_information_keys
        self.omega_network=omega_network
        self.omega_information_keys = omega_information_keys
        self.alternating_epoch=alternating_epoch
        self.if_train_independ=if_train_independ

    def forward(self, subdomain_data,global_data,tho_batch):
        weight_function=subdomain_data[self.weight_function_keys[0]]# (batch_size, H, W)
        weight_function=weight_function.unsqueeze(1)# 增加chennel维度(batch_size,1, H, W)
        # test_process=True#测试用
        # if len(weight_function.shape)==1:
        #     weight_function=weight_function.unsqueeze(1)
        #print("测试节点 weight_function的维度为",weight_function.shape)
        batch_size=tho_batch.size(0)
        omega_information = torch.cat(
            [
                global_data[name].unsqueeze(0) if global_data[name].dim() == 0 else global_data[name]
                for name in self.omega_information_keys
                if name in global_data  # 跳过不存在的键
            ],
            dim=0
        ).unsqueeze(dim=0)
        omega_information=omega_information.expand(batch_size,-1)

        if any(key in subdomain_data for key in self.omega_information_keys):
            omega_information_subdata=torch.cat(
                [
                    subdomain_data[name].unsqueeze(1)
                    for name in self.omega_information_keys
                    if name in subdomain_data  # 跳过不存在的键
                ],
                dim=-1
            )
            # print("测试节点 geometrial_information_subdata形状：",geometrial_information_subdata.shape)
            # print("测试节点 geometrial_information形状: ",geometrial_information.shape )
            omega_information=torch.cat([omega_information_subdata,omega_information],dim=-1)
        #-----------------------------------------------分割线---------------------------------------
        sub_information = torch.cat(
            [
                global_data[name].unsqueeze(0) if global_data[name].dim() == 0 else global_data[name]
                for name in self.subdomain_information_keys
                if name in global_data  # 跳过不存在的键
            ],
            dim=0
        ).unsqueeze(dim=0)
        sub_information=sub_information.expand(batch_size,-1)

        if any(key in subdomain_data for key in self.subdomain_information_keys):
            sub_information_subdata=torch.cat(
                [
                    subdomain_data[name].unsqueeze(1)
                    for name in self.subdomain_information_keys
                    if name in subdomain_data  # 跳过不存在的键
                ],
                dim=-1
            )
            # print("测试节点 geometrial_information_subdata形状：",geometrial_information_subdata.shape)
            # print("测试节点 geometrial_information形状: ",geometrial_information.shape )
            sub_information=torch.cat([sub_information_subdata,sub_information],dim=-1)
            #print("测试节点，sub_information最终的shape为：",sub_information.shape)

        # weight_function=weight_function
        # query_information=query_information
        # geometrial_information=geometrial_information
        F_weight=self.weight_function_network(weight_function)
        # print(f"weight特征向量的形状为{F_weight.shape}")
        # F_physics=self.physics_information_network(physics_information)
        # print(f"physics特征向量的形状为{F_physics.shape}")
        F_sub_information=self.subdomain_information_network(sub_information)
        #print(f"weight特征向量的形状为{F_weight.shape}")
        #print(f"subdoamin_information特征向量的形状为{F_geometrial.shape}")
        F_subdomain=torch.cat([F_weight,F_sub_information],dim=1)
        #print(f"拼接的向量形状为：{F_subdomain.shape}")
        # F_subdomain = F_subdomain.unsqueeze(1)  # 变为 (batch_size, 1, d_model)
        # #print(f"拼接的向量形状为：{F_subdomain.shape}")
        # F_query=self.query_information_network(query_information)
        # print(f"query的特征向量的形状wield{F_query.shape}")
        F_feature=self.feature_out_network(F_subdomain)
        F_omega=self.omega_network(omega_information)
        F_omega=-F.softmax(F_omega,dim=-1)#确保omega满足约束，权重为负
        # print(f"predictor特征向量形状为{F_feature.shape}")
        # print(f"测试节点 tho_batch's shape is ",tho_batch.shape)
        output=self.predictor_network(F_feature,F_omega,tho_batch)
        # print(f"output的形状为{output.shape}")
        # output=F.softmax(output,dim=-1)
        # print(f"经过softmax之后的形状{output.shape}")
        # output=torch.argmax(output,dim=-1)
        # print(f"最终输出的预测类别的形状为{output.shape}")
        return output
    def update(self,update=1.5):
        try:
            self.predictor_network.update()
        except AttributeError:
            print("网络更新异常")

    def alternating_train(self,epochs):
        if self.if_train_independ==True:
            pass
        else:
            if epochs%self.alternating_epoch!=0:
                pass
            elif int(epochs/self.alternating_epoch)%2==0:#冻结omega_network的权重
                omega_network_unfreeze=False
                other_network_unfreeze=True
                set_network_trainable(self.omega_network,unfreeze=omega_network_unfreeze)
                set_network_trainable(self.predictor_network,unfreeze=other_network_unfreeze)
                set_network_trainable(self.weight_function_network,unfreeze=other_network_unfreeze)
                set_network_trainable(self.feature_out_network,unfreeze=other_network_unfreeze)
                set_network_trainable(self.subdomain_information_network,unfreeze=other_network_unfreeze)
            elif int(epochs/self.alternating_epoch)%2==1:
                omega_network_unfreeze=True
                other_network_unfreeze=False
                set_network_trainable(self.omega_network,unfreeze=omega_network_unfreeze)
                set_network_trainable(self.predictor_network,unfreeze=other_network_unfreeze)
                set_network_trainable(self.weight_function_network,unfreeze=other_network_unfreeze)
                set_network_trainable(self.feature_out_network,unfreeze=other_network_unfreeze)
                set_network_trainable(self.subdomain_information_network,unfreeze=other_network_unfreeze)

if __name__=="__main__":


    input_dim = 20  # 全局输入特征维度
    d_model = 8  # 主要特征维度
    hidden_dim = 32  # 隐藏层维度
    output_dim = 20  # 预测输出维度
    batch_size = 8  # 2 组数据
    k_list = 5  # 每组数据的 tho 数量不同
    # 创建模型
    #attention_network=AttentionModule(3*d_model)
    # attention_network=nn.MultiheadAttention(embed_dim=3*d_model, num_heads=1, batch_first=True)
    #weight_network=MLP(input_dim=1,output_dim=d_model,layer_number=2,layer_size=[16,16])
    weight_network=AdaptiveCNN(input_channels=1,output_size=d_model)
    #query_network=MLP(input_dim=1,output_dim=d_model,layer_number=2,layer_size=[16,16])
    # query_network=ThoFeatureExtractor(d_model=3*d_model,input_dim=2)
    sub_network=MLP(input_dim=3,output_dim=d_model,layer_number=2,layer_size=[16,16])
    feature_out_network=MLP(input_dim=2*d_model,output_dim=output_dim,layer_number=2,layer_size=[16,16])
    preditor_network=CustomNet(feature_dim=output_dim)
    #physicis_network=MLP(input_dim=2,output_dim=d_model,layer_number=2,layer_size=[16,16])
    model=Step_function_network(input_dim=input_dim,output_dim=output_dim,weight_function_network=weight_network
                                            ,feature_out_network=feature_out_network,predictor_network=preditor_network,device=torch.device('cpu'),weihgt_function_keys=['weight_function_grid'],subdomain_information_keys=["h","H","m_l"],subdomain_information_network=sub_network)
    tho_inputs=torch.randn(batch_size,k_list,1)
    #
    dataset = MultiHDF5Dataset_for_eigenvale_number(r"D:\pycharm_project\NN_for_DDM\data\low_frequency_interatation",
                                                    "data", global_data_keys=['H', 'h','tho','kappa','sigma'],
                                                    subdomain_data_keys=['number_of_eigval_in_found', 'blk_size',
                                                                         'm_l','weight_function_grid'],batch_size=batch_size)

    hdf5data=dataset[1]
    dataloader_hdf5data = hdf5data.get_batches()
    criterion = nn.CrossEntropyLoss(reduction='none')
    mse=nn.MSELoss()
    globaldata=hdf5data.get_globaldata()
    for subdomain_data, caogao_data in dataloader_hdf5data:
        caogao_data = list(caogao_data)
        batch_size = len(caogao_data)
        tho_inputs = torch.randn(batch_size, k_list,1)
        # weight_input = torch.randn(batch_size, input_dim)
        # geometrial_input = torch.randn(batch_size, input_dim)
        # physicis_input = torch.randn(batch_size, input_dim)
        print("测试节点 tho_input的维度为：",tho_inputs.shape)
        # tho_inputs=torch.ones_like(tho_inputs)
        labels = hdf5data.get_labels(caogao_data, tho_inputs)
        # print(get_discrate_labels(labels,subdomain_data['m_l'],5))
        discrate_labels = get_discrete_labels(labels=labels, ml=subdomain_data['m_l'], feature_dim=output_dim)
        #print("测试节点 discrete_labels's shape is ",discrate_labels.s)
        output = model(subdomain_data,globaldata,tho_inputs)
        print("得到的标签的维度为:", discrate_labels.shape)
        print("测试节点：最终得到的output:",output)
        print("测试节点：output在重新协调前形状为:",output.shape)
        # output=output.permute(0,2,1)
        # print("测试节点：output在重新协调后形状为:",output.shape)
        loss=mse(output,discrate_labels.to(dtype=torch.float32))
        loss=loss.sum()
        print(loss.shape)
        loss.backward()



        # print("测试节点 labels形状", labels.shape)
        # print("测试节点 离散label形状", discrate_labels.shape)
        # loss_preb=torch.softmax(output,dim=2)
        # print("测试节点 loss_preb形状为",loss_preb.shape)
        # loss=-(discrate_labels*loss_preb).sum(dim=2)
        # print("测试节点 loss形状为:",loss.shape)
        # losssum=loss.sum()
        # print("测试节点 losssum形状为:",losssum.shape)
        # print(losssum)
        # losssum.backward()






















