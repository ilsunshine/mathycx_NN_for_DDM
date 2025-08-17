#网络命名规则为架构类型，主要结构+整体编号，例如PINNs_FC_1表示是第一个构建的网络结构，其结构为主要为FC
import torch
from torch.autograd import Function
from act_fun import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
# import torch.nn.functional as F
import math
import torch.nn.functional as F
from DataSet import *
from typing import Dict, Tuple, Optional
def get_activation_function(name, **kwargs):
    #通过关键字返回激活函数类用于构建网络
    activation_functions = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "leakyrelu":nn.LeakyReLU,
        "leakyrelu_power": lambda: Leaky_ReLUPower(**kwargs),
        "relu_power": lambda: ReLUPower(**kwargs),  # 动态传递参数
        "crelu_power": lambda: cReLuPower(**kwargs)
    }
    return activation_functions[name]()

def init_last_linear_zero_weight_bias_one(model: nn.Module):#将网络最后一层的nn.Linear权重初始化为0，偏置初始化为1
    last_linear = None
    # 查找最后一个 nn.Linear 层
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear):
            last_linear = module
            break
    if last_linear is not None:
        nn.init.constant_(last_linear.weight, 0.0)
        if last_linear.bias is not None:
            nn.init.constant_(last_linear.bias, 1.0)
        print(f"Initialized: {last_linear}")
    else:
        print("No nn.Linear layer found in model.")
# class AdaptivePoolEncoder(nn.Module):#带自适应池化的编码器
#     def __init__(self, output_dim=256):
#         super().__init__()
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(1, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#         )
#         self.pool = nn.AdaptiveMaxPool1d(output_dim)  # 关键修改点
#
#     def forward(self, x):
#         """
#         修复维度问题后的前向传播
#         输入x: [batch_size, seq_len]
#         输出: [batch_size, output_dim*256]
#         """
#         # 步骤1：维度调整
#         x = x.unsqueeze(-1)  # [batch, seq_len, 1]
#
#         # 步骤2：特征提取
#         features = self.feature_extractor(x)  # [batch, seq_len, 256]
#
#         # 步骤3：维度转置
#         features = features.permute(0, 2, 1)  # [batch, 256, seq_len]
#
#         # 步骤4：自适应池化
#         pooled = self.pool(features)  # [batch, 256, output_dim]
#
#         # 步骤5：安全重塑维度（修复错误的核心）
#         return pooled.reshape(pooled.size(0), -1)  # 使用reshape替代view
class MLP(nn.Module):#构建MLP
    def __init__(self,input_dim,output_dim,layer_number,layer_size=[],act_fun="relu",pow_k=1):
        super(MLP, self).__init__()
        activation_fn = get_activation_function(act_fun, k=pow_k)
        layers = []
        layers.append(nn.Linear(input_dim, layer_size[0], bias=True))
        layers.append(activation_fn)
        for i in range(1, layer_number):
            layers.append(nn.Linear(layer_size[i - 1], layer_size[i], bias=True))
            layers.append(activation_fn)

            # 输出层
        layers.append(nn.Linear(layer_size[-1], output_dim))
        #self.layers = layers
        # 将层组合成一个 nn.Sequential 模块
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # for layer in self.layers:
        #     print(layer.__class__.__name__)
        #     x=layer(x)
        # return x
        return self.model(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features,if_batchnorm=True):
        """
        参考ResNet的Block，将CNN改成FC
        :param in_features:
        :param out_features:
        :param if_batchnorm: 是否采用batch_norm技术
        """
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)

        self.fc2 = nn.Linear(out_features, out_features)
        self.if_batchnorm=if_batchnorm
        if if_batchnorm:
            self.bn1 = nn.BatchNorm1d(out_features)
            self.bn2 = nn.BatchNorm1d(out_features)

        # 如果输入输出维度不一致，用downsample调整
        self.downsample = None
        if in_features != out_features:
            self.downsample = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        if self.if_batchnorm:
            out = self.bn1(out)
        out = F.relu(out)

        out = self.fc2(out)
        if self.if_batchnorm:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = F.relu(out)
        return out


class FCResNet_Block(nn.Module):
    def __init__(self, input_dim, output_dim, layers=[2,2,2,2],layer_size=[64,128,256,512],if_batchnorm=True):
        """
        input_dim: 输入特征维度
        num_classes: 输出类别数
        layers: 每层堆叠的 ResidualBlock 数量, 类似 ResNet18 的 [2,2,2,2]
        layer_size:每层layer的输出维度
        """
        super(FCResNet_Block, self).__init__()

        # 起始投影层，统一映射到layer_size[0]
        self.if_batchnorm=if_batchnorm
        if self.if_batchnorm:
            self.fc_in = nn.Sequential(
                nn.Linear(input_dim, layer_size[0]),
                nn.BatchNorm1d(layer_size[0]),
                nn.ReLU(inplace=True)
            )
        else:
            self.fc_in = nn.Sequential(
                nn.Linear(input_dim, layer_size[0]),
                nn.ReLU(inplace=True)
            )

        # 残差层堆叠
        self.layer1 = self._make_layer(layer_size[0], layer_size[0], layers[0])
        self.layer2 = self._make_layer(layer_size[0],layer_size[1], layers[1])
        self.layer3 = self._make_layer(layer_size[1],layer_size[2], layers[2])
        self.layer4 = self._make_layer(layer_size[2],layer_size[3], layers[3])

        # 输出层
        self.fc_out = nn.Linear(layer_size[3], output_dim)

    def _make_layer(self, in_features, out_features, blocks):
        layers = []
        layers.append(ResidualBlock(in_features, out_features,if_batchnorm=self.if_batchnorm))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_features, out_features,if_batchnorm=self.if_batchnorm))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.fc_in(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.fc_out(out)
        return out
class CNN_with_adaptive_pool(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_feature_dim,adaptive_mode='max',cnn_layer=2):
        super(CNN_with_adaptive_pool, self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.hidden_feature_dim=hidden_feature_dim
        self.adaptive_mode=adaptive_mode
        # if adaptive_mode=='max':
        #     self.adaptive_pool=F.adaptive_max_pool2d()
class ThoFeatureExtractor(nn.Module):
    """ 从输入 \tho 生成特征向量 """

    def __init__(self, d_model,input_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, tho):
        #return self.mlp(tho.unsqueeze(-1))  # (k, d_model)
        return self.mlp(tho)  # (k, d_model)


class AttentionModule(nn.Module):
    """ 计算 \tho 特征与全局特征的注意力 """

    def __init__(self, d_model):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

    def forward(self, F_tho, F_final):
        #print("开始测试计算")
        Q = self.W_Q(F_tho)  # (k, d_model)
        #print(f"Q的形状为{Q.shape}")
        K = self.W_K(F_final)  # (1, d_model)
        #print(f"K的形状为{K.shape}")
        V = self.W_V(F_final)  # (1, d_model)
        #print(f"V的形状为{V.shape}")
        attn_weights = torch.softmax(Q @ K.T / (d_model ** 0.5), dim=0)  # (k, 1)
        #print(f"注意力权重形状：{attn_weights.shape}")
        return attn_weights * V  # (k, d_model)


class PredictionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)  # (k, output_dim)

class AdaptivePoolEncoder(nn.Module):
    def __init__(self, feature_dim=128,output_dim=256):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )
        if output_dim%feature_dim!=0:
            raise ValueError("feature_dim应当为output_dim的因数")
        if feature_dim<=0:
            raise ValueError("feature_dim应当为正整数")
        pool_dim=int(output_dim/feature_dim)
        #print("测试节点 pool_dim",pool_dim)
        self.pool = nn.AdaptiveMaxPool1d(pool_dim)  # 关键修改点

    def forward(self, x):
        """
        修复维度问题后的前向传播
        输入x: [batch_size, seq_len]
        输出: [batch_size, output_dim*256]
        """
        # 步骤1：维度调整
        # print("测试节点 x的维度为",x.shape)
        x = x.unsqueeze(-1)  # [batch, seq_len, 1]

        # 步骤2：特征提取
        features = self.feature_extractor(x)  # [batch, seq_len, 256]

        # 步骤3：维度转置
        # print("测试节点 features的维度为",features.shape)
        features = features.permute(0, 2, 1)  # [batch, 256, seq_len]

        # 步骤4：自适应池化
        pooled = self.pool(features)  # [batch, 256, output_dim]
        # print("测试节点 经过自适应池化之后的维度为",pooled.shape)
        # 步骤5：安全重塑维度（修复错误的核心）
        return pooled.reshape(pooled.size(0), -1)  # 使用reshape替代view
# class NeuralNetwork(nn.Module):
#     def __init__(self, input_dim, d_model, hidden_dim, output_dim):
#         super().__init__()
#         self.global_extractor = nn.Linear(input_dim, d_model)  # 全局特征提取
#         self.tho_extractor = ThoFeatureExtractor(d_model)  # \tho 特征提取
#         self.attention = AttentionModule(d_model)  # 注意力机制
#         self.predictor = PredictionNetwork(d_model, hidden_dim, output_dim)  # 最终预测
#
#     def forward(self, global_input, tho_values):
#         F_final = self.global_extractor(global_input).unsqueeze(0)  # (1, d_model)
#         print(f"全局特征的形状：{F_final.shape}")
#         F_tho = self.tho_extractor(tho_values)  # (k, d_model)
#         print(f"查询特征的形状：{F_tho.shape}")
#         F_output = self.attention(F_tho, F_final)  # (k, d_model)
#         print(f"经过注意力机制后的形状{F_output.shape}")
#         output=self.predictor(F_output)
#         print(f"最终输出的形状:{output.shape}")
#         return  output# (k, output_dim)

#对batch_size进行处理的
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, d_model, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        # 处理全局输入的 MLP
        self.global_fc = nn.Linear(input_dim, d_model)
        # 处理 tho 输入的 MLP
        self.tho_fc = nn.Linear(1, d_model)
        # 注意力机制
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, batch_first=True)
        # 最终预测的 MLP
        self.final_fc = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, global_input, tho_input):
        """
        global_input: (batch_size, input_dim)
        tho_input: (batch_size, k, 1)  # k 代表 tho 的数量，每个样本 k 一样
        """
        batch_size, k, _ = tho_input.shape  # 获取 batch_size 和 tho 数量

        # 计算全局特征 F_final
        F_final = self.global_fc(global_input)  # (batch_size, d_model)
        F_final = F_final.unsqueeze(1)  # 变为 (batch_size, 1, d_model)

        # 计算 tho 特征 F_tho
        F_tho = self.tho_fc(tho_input)  # (batch_size, k, d_model)

        # 注意力机制 (Query = F_tho, Key = F_final, Value = F_final)
        F_output, _ = self.attn(F_tho, F_final, F_final)  # (batch_size, k, d_model)

        # 通过 MLP 进行最终预测
        output = self.final_fc(F_output)  # (batch_size, k, output_dim)
        print(f"全局特征的形状：{F_final.shape}")

        print(f"查询特征的形状：{F_tho.shape}")

        print(f"经过注意力机制后的形状{F_output.shape}")

        print(f"最终输出的形状:{output.shape}")

        return output
class Feature_Number_Prediction_Network(nn.Module):
    def __init__(self,input_dim,output_dim,weight_function_network,physics_information_network,geometrical_information_network,query_information_network,attention_network,predictor_network,device=None):
        super(Feature_Number_Prediction_Network, self).__init__()
        if device is None:
            self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device=device
        print(self.device)
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.weight_function_network=weight_function_network.to(self.device)
        self.physics_information_network=physics_information_network.to(self.device)
        self.geometrial_information_network=geometrical_information_network.to(self.device)
        self.query_information_network=query_information_network.to(self.device)
        self.attention_network=attention_network.to(self.device)
        self.predictor_network=predictor_network.to(self.device)
    def forward(self, weight_function, physics_information,geometrial_information,query_information):
        F_weight=self.weight_function_network(weight_function)
        F_physics=self.physics_information_network(physics_information)
        F_geometrial=self.geometrial_information_network(geometrial_information)
        print(f"weight特征向量的形状为{F_weight.shape}")
        print(f"physics特征向量的形状为{F_physics.shape}")
        print(f"geometroal特征向量的形状为{F_geometrial.shape}")
        F_subdomain=torch.cat([F_weight,F_physics,F_geometrial],dim=1)
        print(f"拼接的向量形状为：{F_subdomain.shape}")
        F_subdomain = F_subdomain.unsqueeze(1)  # 变为 (batch_size, 1, d_model)
        print(f"拼接的向量形状为：{F_subdomain.shape}")
        F_query=self.query_information_network(query_information)
        print(f"query的特征向量的形状wield{F_query.shape}")
        F_preditor,_=self.attention_network(F_query,F_subdomain,F_subdomain)
        print(f"predictor特征向量形状为{F_preditor.shape}")
        output=self.predictor_network(F_preditor)
        print(f"output的形状为{output.shape}")
        output=F.softmax(output,dim=-1)
        print(f"经过softmax之后的形状{output.shape}")
        # output=torch.argmax(output,dim=-1)
        # print(f"最终输出的预测类别的形状为{output.shape}")

        return output
class Feature_Number_Prediction_Network_with_subdomain_data(nn.Module):
    def __init__(self,input_dim,output_dim,weight_function_network,physics_information_network,geometrical_information_network,query_information_network,attention_network,predictor_network,device=None,
                 weihgt_function_keys=['weight_function'],physics_information_keys=["kappa","sigma"],geometrial_information_keys=['H','h']):
        #相比原有的多一步将数据进行拆分与组装的过程

        super(Feature_Number_Prediction_Network_with_subdomain_data, self).__init__()
        if device is None:
            self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device=device
        #print(self.device)
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.weight_function_network=weight_function_network
        self.physics_information_network=physics_information_network
        self.geometrial_information_network=geometrical_information_network
        self.query_information_network=query_information_network
        self.attention_network=attention_network
        self.predictor_network=predictor_network


        self.weight_function_keys=weihgt_function_keys
        self.physics_information_keys=physics_information_keys
        self.geometrial_information_keys=geometrial_information_keys
    def forward(self, subdomain_data,global_data,tho_batch):
        weight_function=subdomain_data[self.weight_function_keys[0]]

        # test_process=True#测试用
        if len(weight_function.shape)==1:
            weight_function=weight_function.unsqueeze(1)
        #print("测试节点 weight_function的维度为",weight_function.shape)
        batch_size=tho_batch.size(0)
        #physics_information=torch.stack([global_data[name] for name in self.physics_information_keys],dim=-1)
        #print("测试节点：",[global_data[name] for name in self.physics_information_keys])
        physics_information=torch.cat([global_data[name].unsqueeze(0) if global_data[name].dim() == 0 else global_data[name]
            for name in self.physics_information_keys],dim=0)#形状为[input_dim]
        physics_information=physics_information.unsqueeze(dim=0)#形状为[1,input_dim]
        physics_information=physics_information.expand(batch_size,-1)#形状为[batch_size,input_dim]
        #print("测试节点 physics_information的维度为：",physics_information.shape)
        # geometrial_information=torch.cat([global_data[name].unsqueeze(0) if global_data[name].dim() == 0 else global_data[name]
        #     for name in self.geometrial_information_keys],dim=0).unsqueeze(dim=0)
        geometrial_information = torch.cat(
            [
                global_data[name].unsqueeze(0) if global_data[name].dim() == 0 else global_data[name]
                for name in self.geometrial_information_keys
                if name in global_data  # 跳过不存在的键
            ],
            dim=0
        ).unsqueeze(dim=0)
        geometrial_information=geometrial_information.expand(batch_size,-1)
        if any(key in subdomain_data for key in self.geometrial_information_keys):
            geometrial_information_subdata=torch.cat(
                [
                    subdomain_data[name].unsqueeze(0) if subdomain_data[name].dim() == 0 else subdomain_data[name]
                    for name in self.geometrial_information_keys
                    if name in subdomain_data  # 跳过不存在的键
                ],
                dim=-1
            ).unsqueeze(dim=1)
            geometrial_information=torch.cat([geometrial_information_subdata,geometrial_information],dim=-1)

        # print("测试节点 geometrail_information维度为",geometrial_information.shape)
        # print("测试节点 geometrail_information_subdata维度为", geometrial_information_subdata.shape)
        m_l=subdomain_data['m_l']
        m_l=m_l.unsqueeze(1).unsqueeze(2)
        k_list=tho_batch.size(1)
        m_l=m_l.expand(-1,k_list,-1)
        m_l=m_l
        query_information=torch.cat([m_l,tho_batch],dim=-1)
        # weight_function=weight_function
        # query_information=query_information
        # geometrial_information=geometrial_information
        F_weight=self.weight_function_network(weight_function)
        F_physics=self.physics_information_network(physics_information)
        F_geometrial=self.geometrial_information_network(geometrial_information)
        #print(f"weight特征向量的形状为{F_weight.shape}")
        #print(f"physics特征向量的形状为{F_physics.shape}")
        #print(f"geometroal特征向量的形状为{F_geometrial.shape}")
        F_subdomain=torch.cat([F_weight,F_physics,F_geometrial],dim=1)
        #print(f"拼接的向量形状为：{F_subdomain.shape}")
        F_subdomain = F_subdomain.unsqueeze(1)  # 变为 (batch_size, 1, d_model)
        #print(f"拼接的向量形状为：{F_subdomain.shape}")
        F_query=self.query_information_network(query_information)
        #print(f"query的特征向量的形状wield{F_query.shape}")
        F_preditor,_=self.attention_network(F_query,F_subdomain,F_subdomain)
        #print(f"predictor特征向量形状为{F_preditor.shape}")
        output=self.predictor_network(F_preditor)
        # print(f"output的形状为{output.shape}")
        output=F.softmax(output,dim=-1)
        #output = F.tanh(output)/2+1
        #print(f"经过softmax之后的形状{output.shape}")
        # output=torch.argmax(output,dim=-1)
        # print(f"最终输出的预测类别的形状为{output.shape}")

        return output

class Feature_Number_Prediction_Network_with_subdomain_data_with_weight_function_grid(nn.Module):#对含有weight_function经过重排列为矩阵与延拓后的数据进行处理，用自适应CNN进行处理，其余不变
    def __init__(self,input_dim,output_dim,weight_function_network,physics_information_network,geometrical_information_network,query_information_network,attention_network,predictor_network,device=None,
                 weihgt_function_keys=['weight_function'],physics_information_keys=["kappa","sigma"],geometrial_information_keys=['H','h']):


        super(Feature_Number_Prediction_Network_with_subdomain_data_with_weight_function_grid, self).__init__()
        if device is None:
            self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device=device
        #print(self.device)
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.weight_function_network=weight_function_network
        self.physics_information_network=physics_information_network
        self.geometrial_information_network=geometrical_information_network
        self.query_information_network=query_information_network
        self.attention_network=attention_network
        self.predictor_network=predictor_network


        self.weight_function_keys=weihgt_function_keys
        self.physics_information_keys=physics_information_keys
        self.geometrial_information_keys=geometrial_information_keys
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
        physics_information=torch.cat([global_data[name].unsqueeze(0) if global_data[name].dim() == 0 else global_data[name]
            for name in self.physics_information_keys],dim=0)#形状为[input_dim]
        physics_information=physics_information.unsqueeze(dim=0)#形状为[1,input_dim]
        physics_information=physics_information.expand(batch_size,-1)#形状为[batch_size,input_dim]
        #print("测试节点 physics_information的维度为：",physics_information.shape)
        # geometrial_information=torch.cat([global_data[name].unsqueeze(0) if global_data[name].dim() == 0 else global_data[name]
        #     for name in self.geometrial_information_keys],dim=0).unsqueeze(dim=0)
        geometrial_information = torch.cat(
            [
                global_data[name].unsqueeze(0) if global_data[name].dim() == 0 else global_data[name]
                for name in self.geometrial_information_keys
                if name in global_data  # 跳过不存在的键
            ],
            dim=0
        ).unsqueeze(dim=0)
        geometrial_information=geometrial_information.expand(batch_size,-1)
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
        if any(key in subdomain_data for key in self.geometrial_information_keys):
            geometrial_information_subdata=torch.cat(
                [
                    subdomain_data[name].unsqueeze(1)
                    for name in self.geometrial_information_keys
                    if name in subdomain_data  # 跳过不存在的键
                ],
                dim=-1
            )
            # print("测试节点 geometrial_information_subdata形状：",geometrial_information_subdata.shape)
            # print("测试节点 geometrial_information形状: ",geometrial_information.shape )
            geometrial_information=torch.cat([geometrial_information_subdata,geometrial_information],dim=-1)
        # print("测试节点 geometrail_information维度为",geometrial_information.shape)
        # print("测试节点 geometrail_information_subdata维度为", geometrial_information_subdata.shape)
        m_l=subdomain_data['m_l']
        m_l=m_l.unsqueeze(1).unsqueeze(2)
        k_list=tho_batch.size(1)
        m_l=m_l.expand(-1,k_list,-1)
        m_l=m_l
        query_information=torch.cat([m_l,tho_batch],dim=-1)

        # weight_function=weight_function
        # query_information=query_information
        # geometrial_information=geometrial_information
        F_weight=self.weight_function_network(weight_function)
        # print(f"weight特征向量的形状为{F_weight.shape}")
        F_physics=self.physics_information_network(physics_information)
        # print(f"physics特征向量的形状为{F_physics.shape}")
        F_geometrial=self.geometrial_information_network(geometrial_information)
        # print(f"weight特征向量的形状为{F_weight.shape}")
        # print(f"physics特征向量的形状为{F_physics.shape}")
        # print(f"geometroal特征向量的形状为{F_geometrial.shape}")
        F_subdomain=torch.cat([F_weight,F_physics,F_geometrial],dim=1)
        #print(f"拼接的向量形状为：{F_subdomain.shape}")
        F_subdomain = F_subdomain.unsqueeze(1)  # 变为 (batch_size, 1, d_model)
        #print(f"拼接的向量形状为：{F_subdomain.shape}")
        F_query=self.query_information_network(query_information)
        #print(f"query的特征向量的形状wield{F_query.shape}")
        F_preditor,_=self.attention_network(F_query,F_subdomain,F_subdomain)
        # print(f"predictor特征向量形状为{F_preditor.shape}")
        output=self.predictor_network(F_preditor)
        # print(f"output的形状为{output.shape}")
        output=F.softmax(output,dim=-1)
        #print(f"经过softmax之后的形状{output.shape}")
        # output=torch.argmax(output,dim=-1)
        # print(f"最终输出的预测类别的形状为{output.shape}")

        return output
class AdaptiveCNN(nn.Module):#带自适应的网格
    def __init__(self,
                 input_channels=1,
                 output_size=32,  # 最终输出的向量维度
                 target_size=(64, 64),# 自适应卷积的中间统一尺寸
                 initial_cnn_arg={"layer_number":4,"kernel_size":[3,3,3,3],"channel_size":[16,32,64,64],"padding_size":[1,1,1,1],"if_use_batchNorm":[True,True,True,True],"act_fun":"relu","if_pooling":[True,True,False,False], "pooling_model":"max","pooling_size":[2,2,2,2],},#前处理的CNN的相关参数设置
                 final_cnn_arg={"layer_number": 3, "kernel_size": [3, 3, 3,], "channel_size": [64,64,32],"padding_size": [1, 1, 1], "if_use_batchNorm": [True, True, True],"act_fun": "relu", "if_pooling": [True, True, False], "pooling_model": "max","pooling_size": [2, 2, 2, 2], },
                 adaptive_layer_arg={ "kernel_size":[3],"channel_size":[64],"padding_size":[1],"if_use_batchNorm":[True],"act_fun":"relu",},
                 global_pool_size=1
                 ):
        super(AdaptiveCNN, self).__init__()
        self.target_size = target_size
        initial_cnn_act=get_activation_function(initial_cnn_arg["act_fun"])
        layers=[nn.Conv2d(input_channels, initial_cnn_arg["channel_size"][0], kernel_size=initial_cnn_arg["kernel_size"][0], padding=initial_cnn_arg["padding_size"][0])]
        if initial_cnn_arg["if_use_batchNorm"][0]:
            layers.append(nn.BatchNorm2d(initial_cnn_arg["channel_size"][0]))
        layers.append(initial_cnn_act)
        if initial_cnn_arg["if_pooling"][0]:
            if initial_cnn_arg["pooling_model"]=="max":
                layers.append(nn.MaxPool2d(initial_cnn_arg["pooling_size"][0]))
            elif initial_cnn_arg["pooling_model"]=="avg":
                layers.append(nn.AvgPool2d(initial_cnn_arg["pooling_size"][0]))
        for i in range(initial_cnn_arg["layer_number"]-1):

            layers.append(nn.Conv2d(initial_cnn_arg["channel_size"][i], initial_cnn_arg["channel_size"][i+1],
                                kernel_size=initial_cnn_arg["kernel_size"][i+1], padding=initial_cnn_arg["padding_size"][i+1]))
            if initial_cnn_arg["if_use_batchNorm"][i+1]:
                layers.append(nn.BatchNorm2d(initial_cnn_arg["channel_size"][i+1]))
            layers.append(initial_cnn_act)
            if initial_cnn_arg["if_pooling"][i+1]:
                if initial_cnn_arg["pooling_model"] == "max":
                    layers.append(nn.MaxPool2d(initial_cnn_arg["pooling_size"][i+1]))
                elif initial_cnn_arg["pooling_model"] == "avg":
                    layers.append(nn.AvgPool2d(initial_cnn_arg["pooling_size"][i+1]))
        self.initial_cnn = nn.Sequential(*layers)
        del layers
        #第一部分：初始CNN处理
        # self.initial_cnn = nn.Sequential(
        #     nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #
        #     nn.Conv2d(16, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )

        # 第二部分：自适应调整尺寸的CNN
        adaptive_layer_act=get_activation_function(adaptive_layer_arg["act_fun"])
        layers=[ nn.AdaptiveAvgPool2d(target_size)]
        layers.append(nn.Conv2d(initial_cnn_arg["channel_size"][-1], adaptive_layer_arg["channel_size"][0], kernel_size=adaptive_layer_arg["kernel_size"][0], padding=adaptive_layer_arg["padding_size"][0]))
        if adaptive_layer_arg["if_use_batchNorm"][0]:
            layers.append(nn.BatchNorm2d(adaptive_layer_arg["channel_size"][0]))
        layers.append(adaptive_layer_act)
        self.adaptive_layer = nn.Sequential(*layers)
        del layers
        # 第三部分：最终CNN处理
        # self.final_cnn = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #
        #     nn.Conv2d(64, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU()
        # )

        final_cnn_act = get_activation_function(final_cnn_arg["act_fun"])
        layers = [
            nn.Conv2d(adaptive_layer_arg["channel_size"][0], final_cnn_arg["channel_size"][0], kernel_size=final_cnn_arg["kernel_size"][0],
                      padding=final_cnn_arg["padding_size"][0])]
        if final_cnn_arg["if_use_batchNorm"][0]:
            layers.append(nn.BatchNorm2d(final_cnn_arg["channel_size"][0]))
        layers.append(initial_cnn_act)
        if final_cnn_arg["if_pooling"][0]:
            if final_cnn_arg["pooling_model"] == "max":
                layers.append(nn.MaxPool2d(final_cnn_arg["pooling_size"][0]))
            elif final_cnn_arg["pooling_model"] == "avg":
                layers.append(nn.AvgPool2d(final_cnn_arg["pooling_size"][0]))
        for i in range(final_cnn_arg["layer_number"] - 1):

            layers.append(nn.Conv2d(final_cnn_arg["channel_size"][i], final_cnn_arg["channel_size"][i + 1],
                                    kernel_size=final_cnn_arg["kernel_size"][i + 1],
                                    padding=final_cnn_arg["padding_size"][i + 1]))
            if final_cnn_arg["if_use_batchNorm"][i + 1]:
                layers.append(nn.BatchNorm2d(final_cnn_arg["channel_size"][i + 1]))
            layers.append(initial_cnn_act)
            if final_cnn_arg["if_pooling"][i + 1]:
                if final_cnn_arg["pooling_model"] == "max":
                    layers.append(nn.MaxPool2d(final_cnn_arg["pooling_size"][i + 1]))
                elif final_cnn_arg["pooling_model"] == "avg":
                    layers.append(nn.AvgPool2d(final_cnn_arg["pooling_size"][i + 1]))
        self.final_cnn = nn.Sequential(*layers)
        del layers
        # 自适应全局池化+展平
        self.global_pool = nn.AdaptiveAvgPool2d(global_pool_size)
        self.flatten = nn.Flatten()

        # 最终输出调整层
        # print("测试节点 final_cnn_arg[kernel_size][-1]",
        #        final_cnn_arg["kernel_size"][-1])
        # print("测试节点 global_pool_size*final_cnn_arg[kernel_size][-1]",global_pool_size*final_cnn_arg["kernel_size"][-1])
        self.output_layer = nn.Linear(global_pool_size**2*final_cnn_arg["channel_size"][-1], output_size)

    def forward(self, x):
        # 初始CNN处理
        x = self.initial_cnn(x)

        # 自适应调整到统一尺寸
        x = self.adaptive_layer(x)

        # 最终CNN处理
        x = self.final_cnn(x)

        # 全局池化+展平
        x = self.global_pool(x)
        x = self.flatten(x)
        #print("测试节点 x形状",x.shape)
        # 输出向量
        x = self.output_layer(x)
        return x





def load_partial_weights(
        model: torch.nn.Module,
        weight_path: str,
        key_mapping: Optional[Dict[str, str]] = None,
        verbose: bool = True
) -> Tuple[torch.nn.Module, Dict[str, list]]:
    """
    选择性加载模型权重（允许部分匹配）

    参数:
        model: 要加载权重的模型（nn.Module）
        weight_path: 权重文件路径（.pth文件）
        key_mapping: 键名映射字典（可选），格式为 {"旧键名": "新键名"}
        verbose: 是否打印加载信息

    返回:
        (model, mismatch_info): 加载后的模型和缺失/冗余键信息
        mismatch_info格式: {
            "missing": List[str],  # 当前模型有但权重文件中没有的键
            "unexpected": List[str]  # 权重文件有但当前模型没有的键
        }
    """
    # 加载保存的权重
    saved_state_dict = torch.load(weight_path)
    if not isinstance(saved_state_dict, dict):
        raise ValueError("权重文件应包含state_dict（字典格式）")

    # 初始化键名映射
    key_mapping = key_mapping or {}
    model_state_dict = model.state_dict()

    # 转换键名（如果提供映射）
    mapped_saved_dict = {}
    for old_key, new_key in key_mapping.items():
        if old_key in saved_state_dict:
            mapped_saved_dict[new_key] = saved_state_dict.pop(old_key)
    saved_state_dict.update(mapped_saved_dict)

    # 筛选匹配的权重（键名和形状都匹配）
    matched_weights = {}
    for k in model_state_dict:
        if k in saved_state_dict and saved_state_dict[k].shape == model_state_dict[k].shape:
            matched_weights[k] = saved_state_dict[k]

    # 更新模型权重
    model_state_dict.update(matched_weights)
    model.load_state_dict(model_state_dict, strict=False)

    # 记录不匹配的键
    mismatch_info = {
        "missing": [k for k in model_state_dict if k not in matched_weights],
        "unexpected": [k for k in saved_state_dict if k not in model_state_dict]
    }

    # 打印信息（可选）
    if verbose:
        print(f"成功加载 {len(matched_weights)}/{len(model_state_dict)} 层权重")
        if mismatch_info["missing"]:
            print(f"缺失键（当前模型有但未加载）: {mismatch_info['missing']}")
        if mismatch_info["unexpected"]:
            print(f"冗余键（权重文件有但未使用）: {mismatch_info['unexpected']}")

    return model, mismatch_info
if __name__=="__main__":
    # 超参数
    # input_dim = 20  # 全局输入特征维度
    # d_model = 16  # 主要特征维度
    # hidden_dim = 32  # 隐藏层维度
    # output_dim = 20  # 预测输出维度
    # batch_size = 8  # 2 组数据
    # k_list = 5  # 每组数据的 tho 数量不同
    # # 创建模型
    # #attention_network=AttentionModule(3*d_model)
    # attention_network=nn.MultiheadAttention(embed_dim=3*d_model, num_heads=1, batch_first=True)
    # weight_network=MLP(input_dim=input_dim,output_dim=d_model,layer_number=2,layer_size=[16,16])
    # #query_network=MLP(input_dim=1,output_dim=d_model,layer_number=2,layer_size=[16,16])
    # query_network=ThoFeatureExtractor(d_model=3*d_model)
    # geometrial_network=MLP(input_dim=input_dim,output_dim=d_model,layer_number=2,layer_size=[16,16])
    # preditor_network=MLP(input_dim=3*d_model,output_dim=output_dim,layer_number=2,layer_size=[16,16])
    #
    # physicis_network=MLP(input_dim=input_dim,output_dim=d_model,layer_number=2,layer_size=[16,16])
    # model=Feature_Number_Prediction_Network(input_dim=input_dim,output_dim=output_dim,weight_function_network=weight_network
    #                                         ,physics_information_network=physicis_network,geometrical_information_network=geometrial_network,
    #                                         query_information_network=query_network,attention_network=attention_network,predictor_network=preditor_network,device=torch.device('cpu'))
    # tho_inputs=torch.randn(batch_size,k_list,1)
    # weight_input=torch.randn(batch_size, input_dim)
    # geometrial_input=torch.randn(batch_size, input_dim)
    # physicis_input=torch.randn(batch_size, input_dim)
    #
    # output=model(weight_function=weight_input, physics_information=weight_input,geometrial_information=geometrial_input,query_information=tho_inputs)
    # dataset = MultiHDF5Dataset_for_eigenvale_number(r"D:\pycharm_project\NN_for_DDM\data\low_frequency_interatation",
    #                                                 "data", global_data_keys=['H', 'tho'],
    #                                                 subdomain_data_keys=['number_of_eigval_in_found', 'blk_size',
    #                                                                      'm_l'],batch_size=batch_size)
    #
    # hdf5data=dataset[1]
    # dataloader_hdf5data = hdf5data.get_batches()
    # criterion = nn.CrossEntropyLoss(reduction='none')
    # for subdomain_data, caogao_data in dataloader_hdf5data:
    #     caogao_data = list(caogao_data)
    #     batch_size = len(caogao_data)
    #     tho_inputs = torch.randn(batch_size, k_list,1)
    #     weight_input = torch.randn(batch_size, input_dim)
    #     geometrial_input = torch.randn(batch_size, input_dim)
    #     physicis_input = torch.randn(batch_size, input_dim)
    #     print("测试节点 tho_input的维度为：",tho_inputs.shape)
    #     # tho_inputs=torch.ones_like(tho_inputs)
    #     labels = hdf5data.get_labels(caogao_data, tho_inputs)
    #     # print(get_discrate_labels(labels,subdomain_data['m_l'],5))
    #     discrate_labels = get_discrete_labels(labels=labels, ml=subdomain_data['m_l'], feature_dim=output_dim)
    #
    #     output = model(weight_function=weight_input, physics_information=weight_input,
    #                    geometrial_information=geometrial_input, query_information=tho_inputs)
    #     print("得到的标签的维度为:", discrate_labels.shape)
    #     print("测试节点：output在重新协调前形状为:",output.shape)
    #     output=output.permute(0,2,1)
    #     print("测试节点：output在重新协调后形状为:",output.shape)
    #     loss=criterion(output,discrate_labels)
    #     loss=loss.sum()
    #     print(loss.shape)
    #     loss.backward()
    #     # print("测试节点 labels形状", labels.shape)
    #     # print("测试节点 离散label形状", discrate_labels.shape)
    #     # loss_preb=torch.softmax(output,dim=2)
    #     # print("测试节点 loss_preb形状为",loss_preb.shape)
    #     # loss=-(discrate_labels*loss_preb).sum(dim=2)
    #     # print("测试节点 loss形状为:",loss.shape)
    #     # losssum=loss.sum()
    #     # print("测试节点 losssum形状为:",losssum.shape)
    #     # print(losssum)
    #     # losssum.backward()


    # 生成测试数据

    # model = NeuralNetwork(input_dim, d_model, hidden_dim, output_dim)
    #
    # tho_inputs = torch.randn(batch_size, k_list, 1)
    # global_inputs = torch.randn(batch_size, input_dim)  # (batch_size, input_dim)
    # output = model(global_inputs, tho_inputs)

    #output=model(weight_function=weight_input, physics_information=weight_input,geometrial_information=geometrial_input,query_information=tho_inputs)
    # 运行测试
    # for i in range(batch_size):
    #     print(f"\n===== 测试样本 {i + 1} =====")
    #     print(f"全局输入 (global_input) 形状: {global_inputs[i].shape}")  # (input_dim,)
    #     output=model(weight_function=weight_input[i], physics_information=weight_input[i],geometrial_information=geometrial_input[i],query_information=tho_inputs[i])
    #     # 运行网络


        # 打印形状
        # print(f"\tho 输入形状: {tho_inputs[i].shape}")  # (k, 1)
        # print(f"\tho 提取的特征 (F_tho) 形状: ({tho_inputs[i].shape[0]}, {d_model})")  # (k, d_model)
        # print(f"全局特征 (F_final) 形状: (1, {d_model})")  # (1, d_model)
        # print(f"注意力输出 (F_output) 形状: ({tho_inputs[i].shape[0]}, {d_model})")  # (k, d_model)
        # print(f"最终预测 (output) 形状: {output.shape}")  # (k, output_dim)



    #测试网络以及数据加载的相容性

    input_dim = 20  # 全局输入特征维度
    d_model = 8  # 主要特征维度
    hidden_dim = 32  # 隐藏层维度
    output_dim = 20  # 预测输出维度
    batch_size = 8  # 2 组数据
    k_list = 5  # 每组数据的 tho 数量不同
    # 创建模型
    #attention_network=AttentionModule(3*d_model)
    attention_network=nn.MultiheadAttention(embed_dim=3*d_model, num_heads=1, batch_first=True)
    #weight_network=MLP(input_dim=1,output_dim=d_model,layer_number=2,layer_size=[16,16])
    weight_network=AdaptiveCNN(input_channels=1,output_size=d_model)
    #query_network=MLP(input_dim=1,output_dim=d_model,layer_number=2,layer_size=[16,16])
    query_network=ThoFeatureExtractor(d_model=3*d_model,input_dim=2)
    geometrial_network=MLP(input_dim=2,output_dim=d_model,layer_number=2,layer_size=[16,16])
    preditor_network=MLP(input_dim=3*d_model,output_dim=output_dim,layer_number=2,layer_size=[16,16])

    physicis_network=MLP(input_dim=2,output_dim=d_model,layer_number=2,layer_size=[16,16])
    model=Feature_Number_Prediction_Network_with_subdomain_data_with_weight_function_grid(input_dim=input_dim,output_dim=output_dim,weight_function_network=weight_network
                                            ,physics_information_network=physicis_network,geometrical_information_network=geometrial_network,
                                            query_information_network=query_network,attention_network=attention_network,predictor_network=preditor_network,device=torch.device('cpu'),weihgt_function_keys=['weight_function_grid'],geometrial_information_keys=['H','H'])
    tho_inputs=torch.randn(batch_size,k_list,1)
    #
    dataset = MultiHDF5Dataset_for_eigenvale_number(r"D:\pycharm_project\NN_for_DDM\data\low_frequency_interatation",
                                                    "data", global_data_keys=['H', 'h','tho','kappa','sigma'],
                                                    subdomain_data_keys=['number_of_eigval_in_found', 'blk_size',
                                                                         'm_l','weight_function_grid'],batch_size=batch_size)

    hdf5data=dataset[1]
    dataloader_hdf5data = hdf5data.get_batches()
    criterion = nn.CrossEntropyLoss(reduction='none')
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

        output = model(subdomain_data,globaldata,tho_inputs)
        print("得到的标签的维度为:", discrate_labels.shape)
        print("测试节点：output在重新协调前形状为:",output.shape)
        output=output.permute(0,2,1)
        print("测试节点：output在重新协调后形状为:",output.shape)
        loss=criterion(output,discrate_labels)
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












