
#构建网络训练所需要的数据集(DataSet类)以及相应的DataLoader
import os
import re
import h5py
import torch
from torch.utils.data import Dataset ,DataLoader
import numpy as np
from torch import nn as nn
# from model import MetricAccumulator
import time,random
from network import *
from model import MetricAccumulator
from datetime import datetime
import warnings
import copy
class HDF5Dataset(Dataset):
    def __init__(self, file_path, global_data_keys=None, subdomain_data_keys=None, if_big_data=True,batch_size=512,shuffle=False,device=None):
        """
        Args:
            file_path (str): HDF5 文件路径
            global_data_keys (list, optional): 需要读取的 global 数据类型列表，None 代表读取全部。
            subdomain_data_keys (list, optional): 需要读取的 subdomain 数据类型列表，None 代表读取全部。
            if_big_data (bool): 是否为大数据集，决定数据加载模式。
        """
        self.file_path = file_path
        self.global_data_keys = global_data_keys
        self.subdomain_data_keys = subdomain_data_keys
        self.if_big_data = if_big_data
        self.batch_size=batch_size
        self.shuffle=shuffle
        if device is None:
            self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device=device



        # 预加载 global 数据
        with h5py.File(self.file_path, 'r') as f:
            self.global_data = self._load_group(f['GlobalData'], self.global_data_keys)
            for key in self.global_data:
                self.global_data[key]=self.global_data[key].to(self.device)
            self.subdomain_names = list(f['sub_domain'].keys())  # 获取所有 subdomain_i 名称

            # 小数据集时，预加载所有 subdomain 数据
            if not self.if_big_data:
                self.subdomain_data = {
                    name: self._load_group(f[f'sub_domain/{name}'], self.subdomain_data_keys)
                    for name in self.subdomain_names
                }

    def _load_group(self, group, data_keys):
        """从 HDF5 组中加载数据，并自动匹配类型
        group: 需要加载的群组
        data_keys:需要加载的数据类型，为None时表示全部加载，若data_keys中
        """
        data_dict = {}
        for key in group.keys():
            if data_keys is None or key in data_keys:
            #     data = group[key][()]
            #     if isinstance(data, (list, tuple)) or data.ndim > 0:  # 如果是数组
            #         data_dict[key] = torch.tensor(data, dtype=torch.float32)
            #     else:  # 标量保持原格式
            #         data_dict[key] = data
                data = group[key][()]
                data_dict[key] = torch.tensor(data, dtype=torch.float32)#统一数据格式方便后面进行拼接与加载


            # 加载原始数据并保持numpy类型
            #     raw_data = group[key][()]
            #
            # # 类型转换逻辑
            #     if isinstance(raw_data, np.ndarray):
            #     # 自动匹配数据类型（保持原始精度）
            #         dtype_map = {
            #             np.float16: torch.float16,
            #             np.float32: torch.float32,
            #             np.float64: torch.float64,
            #             np.int32: torch.int32,
            #             np.int64: torch.int64
            #             }
            #         torch_dtype = dtype_map.get(raw_data.dtype.type, torch.float32)
            #
            #     # 转换为张量并分配设备
            #         data_dict[key] = torch.tensor(raw_data, dtype=torch_dtype).to(self.device)
            #     else:  # 处理标量和非数组数据
            #     # 标量转换为0维张量
            #         data_dict[key] = torch.tensor(raw_data).to(self.device)
            elif key in data_keys:
                warnings.warn(f'key {key} is not in data_keys{data_keys}', DeprecationWarning)

        return data_dict
    def caogao_data(self):#测试与debug用
        num_dims = torch.randint(2, 4, (1,)).item()  # 生成 4, 5 或 6

        # 随机生成每个维度的大小（范围 2 到 10）
        shape = torch.randint(2, 3, (num_dims,)).tolist()

        return torch.randn(*shape)
    def get_globaldata(self):
        """返回 global 数据"""
        return self.global_data

    def __len__(self):
        return len(self.subdomain_names)

    def __getitem__(self, idx):
        """获取 subdomain_i 的数据"""
        subdomain_name = self.subdomain_names[idx]

        if self.if_big_data:
            with h5py.File(self.file_path, 'r') as f:
                subdomain_data = self._load_group(f[f'sub_domain/{subdomain_name}'], self.subdomain_data_keys)
        else:
            subdomain_data = self.subdomain_data[subdomain_name]
        # if True:#测试用
        #     # return subdomain_data,self.caogao_data()
        #     return subdomain_data,subdomain_name
        # else:
        #     None
        return subdomain_data,subdomain_name  # 只返回 subdomain_i 数据
    def get_batches(self,if_shuffle=None):
        """
        创建并返回一个DataLoader对象，用于分批加载数据。
        """
        if if_shuffle is None:
            return DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle)
        else:
            return DataLoader(self, batch_size=self.batch_size, shuffle=if_shuffle)

class MultiHDF5Dataset(Dataset):
    def __init__(self, folder_path, file_prefix, global_data_keys=None, subdomain_data_keys=None, if_big_data=True):
        """
        Args:
            folder_path (str): 存储 HDF5 文件的文件夹路径。
            file_prefix (str): HDF5 文件名前缀，例如 'data' 匹配 'dataxxx.h5'。
            global_data_keys (list, optional): 需要读取的 global 数据类型列表，None 代表读取全部。
            subdomain_data_keys (list, optional): 需要读取的 subdomain 数据类型列表，None 代表读取全部。
            if_big_data (bool): 是否为大数据集，决定数据加载模式。
        """
        self.folder_path = folder_path
        self.file_prefix = file_prefix
        self.global_data_keys = global_data_keys
        self.subdomain_data_keys = subdomain_data_keys
        self.if_big_data = if_big_data

        # 获取符合命名规则的 HDF5 文件
        pattern = re.compile(f"^{file_prefix}(\\d+).h5$")
        self.subdir = sorted(
            [f for f in os.listdir(folder_path) if pattern.match(f)],
            key=lambda x: int(pattern.match(x).group(1))  # 按编号排序
        )
        self.subdir = [os.path.join(folder_path, f) for f in self.subdir]  # 转为完整路径

        if not self.subdir:
            raise ValueError(f"No files matching {file_prefix}XXX.h5 found in {folder_path}")

    def __len__(self):
        return len(self.subdir)

    def __getitem__(self, idx):
        """加载指定 HDF5 文件的数据"""
        file_path = self.subdir[idx]
        return HDF5Dataset(file_path, self.global_data_keys, self.subdomain_data_keys, self.if_big_data)
        #return HDF5Dataset_for_eigenvale_number(file_path, self.global_data_keys, self.subdomain_data_keys, self.if_big_data)
    # def get_batches(self):
    #     """
    #     创建并返回一个DataLoader对象，用于分批加载数据。
    #     """
    #     return DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle)
    
class MultiHDF5Dataset_for_eigenvale_number(Dataset):
    def __init__(self, folder_path, file_prefix, global_data_keys=None, subdomain_data_keys=None, if_big_data=False,batch_size=128):
        """
        Args:
            folder_path (str): 存储 HDF5 文件的文件夹路径。
            file_prefix (str): HDF5 文件名前缀，例如 'data' 匹配 'dataxxx.h5'。
            global_data_keys (list, optional): 需要读取的 global 数据类型列表，None 代表读取全部。
            subdomain_data_keys (list, optional): 需要读取的 subdomain 数据类型列表，None 代表读取全部。
            if_big_data (bool): 是否为大数据集，决定数据加载模式。
        """
        self.folder_path = folder_path
        self.file_prefix = file_prefix
        self.global_data_keys = global_data_keys
        self.subdomain_data_keys = subdomain_data_keys
        self.if_big_data = if_big_data
        self.batch_in_each_data=batch_size

        # 获取符合命名规则的 HDF5 文件
        pattern = re.compile(f"^{file_prefix}(\\d+).h5$")
        self.subdir = sorted(
            [f for f in os.listdir(folder_path) if pattern.match(f)],
            key=lambda x: int(pattern.match(x).group(1))  # 按编号排序
        )
        self.subdir = [os.path.join(folder_path, f) for f in self.subdir]  # 转为完整路径

        if not self.subdir:
            raise ValueError(f"No files matching {file_prefix}XXX.h5 found in {folder_path}")


        #记录所有文件中一共划分了多少个子区域
        self.number_of_domain=0
        for i in range(self.__len__()):
            self.number_of_domain+=self.__getitem__(i).__len__()
        print(f"测试节点：测试self.number_of_domain is {self.number_of_domain}")
    def __len__(self):
        return len(self.subdir)

    def __getitem__(self, idx):
        """加载指定 HDF5 文件的数据"""
        file_path = self.subdir[idx]
        #return HDF5Dataset(file_path, self.global_data_keys, self.subdomain_data_keys, self.if_big_data)
        return HDF5Dataset_for_eigenvale_number(file_path, self.global_data_keys, self.subdomain_data_keys, self.if_big_data,batch_size=self.batch_in_each_data)
class HDF5Dataset_for_eigenvale_number(HDF5Dataset):
    def __init__(self, file_path, global_data_keys=None, subdomain_data_keys=None, if_big_data=True,batch_size=128,shuffle=False,different_shape_key='eigval'):
        super(HDF5Dataset_for_eigenvale_number, self).__init__( file_path, global_data_keys, subdomain_data_keys, if_big_data,batch_size,shuffle)
        self.different_shape_keys=different_shape_key
        self.subdomain_eigval_keys=['eigval']
        self.subdomain_eigval={}#存放不同区域的特征值
        with h5py.File(self.file_path, 'r') as f:

            if not self.if_big_data:
                self.subdomain_eigval = {
                    name: self._load_group(f[f'sub_domain/{name}'], self.subdomain_eigval_keys)
                    for name in self.subdomain_names
                }
    def get_labels(self,subdomain_names,tho,if_need_grad=False,temperature=300):
        """
        :param subdomain_names: 长度为batch的list代表subdomain的names
        :param tho: 形状为（batch,k_list)的tensor向量
        :return: 形状为(batch,klist)的tensor向量，返回大于\tho的特征值个数,特征值的数据是一个batch的字典，键为subdomain的names，对应的值为一维的tensor升序排列的向量
        """
        output=torch.zeros_like(tho)


        if self.if_big_data:
            with h5py.File(self.file_path, 'r') as f:
                eigvals= {
                    name: self._load_group(f[f'sub_domain/{name}'], self.subdomain_eigval_keys)
                    for name in subdomain_names
                }
        else:
            eigvals={name:self.subdomain_eigval[name] for name in subdomain_names}
        for i,name in enumerate(subdomain_names):
            # print("测试节点eigvals[names]:",eigvals[name])
            eigval=eigvals[name]['eigval'].to(tho.device)
            #print('测试节点eigval:',eigval)

            throsholds=tho[i]
            #print("test point throsholds's if_grad is ",throsholds.requires_grad)
            # print("test point throsholds' shape is ",throsholds.shape)
            # print("test point evigal's shape is ",eigval.shape)
            # if tho.dim==3:
            #     throsholds=throsholds.unsq
            #indices=torch.searchsorted(eigval,throsholds,side='right')

            if if_need_grad:
                throsholds=throsholds.squeeze(-1)
                counts=smooth_searchsorted(eigval,throsholds,temperature=temperature)#采用近似可导值逼近torch.searchsorted,temperature越高逼近效果越好，但可导性变差
            else:
                indices = torch.searchsorted(eigval, throsholds, right=True)
                counts = len(eigval) - indices
            #print("test point indices's if_grad is ", indices.requires_grad)
            # if i==0:
            #     print('测试节点 eigval的形状为：',eigval.shape)
            #     print("测试节点 indices的形状为：",indices.shape)
            #     print("测试节点 throsholds的形状为",throsholds.shape)
            # print("test point counts' shape is ",counts.shape)
            output[i]=counts
            #print("测试节点 counts",counts)

            #进行测试输出

        # print('测试节点 输入tho的形状为：', tho.shape)
        #print("测试节点 output:",output)
        if if_need_grad:
            return output.squeeze(-1)
        return output.squeeze(-1).to(torch.long)


def smooth_searchsorted(eigval, thresholds, temperature=100.0):# 可导近似的 searchsorted
    eigval = eigval.unsqueeze(0)          # (1, n)
    thresholds = thresholds.unsqueeze(1)  # (m, 1)
    comparison = eigval - thresholds      # (m, n)
    # print("test point comparison's shape is ",comparison.shape)
    soft_step = torch.sigmoid(temperature * comparison)  # (m, n)
    soft_indices = soft_step.sum(dim=1)   # (m,)
    # print("test point soft_indices' shape is ",soft_indices.shape)
    return soft_indices.unsqueeze(1)
def get_discrete_labels_one_hot(labels, ml, feature_dim):
    # 归一化处理，labels的形状为[batch, k]，ml的形状为[batch] -> [batch, 1]进行广播
    #更新：不进行归一化处理
    #normalized = labels / ml.unsqueeze(-1)
    normalized =labels
    # 生成中点坐标，形状为[feature_dim]
    mid_points = (torch.arange(feature_dim, dtype=torch.float32, device=labels.device) + 0.5) / feature_dim
    mid_points = mid_points.to(normalized.dtype)  # 保持数据类型一致

    # 计算每个归一化值到所有中点的绝对距离，形状为[batch, k, feature_dim]
    distances = torch.abs(normalized.unsqueeze(-1) - mid_points)

    # 找到最近中点的索引，形状为[batch, k]
    indices = torch.argmin(distances, dim=-1)
    # print('测试节点 indices is ',indices)
    # 转换为one-hot编码，形状为[batch, k, feature_dim]
    one_hot = torch.nn.functional.one_hot(indices, num_classes=feature_dim).float()

    return one_hot

class Tho_generator_uniform:
    def __init__(self,beta,klist):
        self.beta=beta
        self.klist=klist
    def gettho(self,batch_size,kappa):
        mean=kappa**self.beta
        device=kappa.device
        low=0.5*mean
        up=1.5*mean
        shape=(batch_size,self.klist,1)
        # print(torch.randn(shape).device)
        # print(low.device)
        output= (up - low) * torch.rand(shape).to(device) + low
        return output
class Tho_generator_uniform_xiugai:
    def __init__(self,beta,klist,low_rate=0.75,up_rate=2.0,mean_rate=0.25):
        self.beta=beta
        self.klist=klist
        self.low_rate=low_rate
        self.up_rate=up_rate
        self.mean_rate=mean_rate
    def gettho(self,batch_size,kappa):
        mean=kappa**self.beta*self.mean_rate
        device=kappa.device
        low=self.low_rate*mean
        up=min(self.up_rate*mean,1.0)
        shape=(batch_size,self.klist,1)
        # print(torch.randn(shape).device)
        # print(low.device)
        output= (up - low) * torch.rand(shape).to(device) + low
        return output
    def update(self,class_number):
        #通过class_number进行更新，尽可能使得每类的均衡
        mean=class_number.mean()
        var=class_number.var()
        N_class=len(class_number)

class Tho_generator_uniform_0_1:
    def __init__(self,beta,klist):
        self.beta=beta
        self.klist=klist
    def gettho(self,batch_size,kappa):
        # mean=kappa**self.beta*0.25
        device=kappa.device
        # low=0.
        # up=2*mean
        shape=(batch_size,self.klist,1)
        # print(torch.randn(shape).device)
        # print(low.device)
        output= torch.rand(shape).to(device)
        return output

class Tho_generator_network(nn.Module):#通过训练神经网络用用于学习合适的概率分布生成均匀的类别分布,目前是废案，在得到标签时会出现不可导的离散操作，即使想办法绕过后仍然无法进行训练
    def __init__(self,k_list,network,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),input_dim=32,class_number=20,model_pth=None):
        super(Tho_generator_network, self).__init__()
        self.network=network.to(device)
        if model_pth is not None:
            self.network.load_state_dict(torch.load(model_pth))
        self.best_dict=copy.deepcopy(self.network.state_dict())# 深拷贝当前网络权重（完全独立的内存副本）
        self.device=device
        self.class_number=class_number
        self.k_list=k_list
        self.input_dim=input_dim
        self.optim=torch.optim.Adam(params=self.network.parameters(),lr=0.001)
        self.sigmoid=nn.Sigmoid()
    def forward(self,intput):
        return self.sigmoid(self.network(intput))

    def gettho(self,batch_size,kappa=None):
        shape = (batch_size*self.k_list, self.input_dim)
        rand_input=torch.rand(shape).to(self.device)
        # print("test point rand_input's shape is ",rand_input.shape)
        # output=self.forward(rand_input)
        return self.forward(rand_input).view(batch_size,self.k_list,1)
    #def train(self: T, mode: bool = True) -> T:
    #
    def train_for_data(self,multihdfdataset,loss=None,epochs=40):#训练函数train,nn.Module类已经定义了train函数，为避免重构，命名为train_for_data
        self.network.train()
        best_kl=np.inf
        for epoch in range(epochs):
            loss_every_epoch = 0.0
            start_time_each_epoch = time.time()
            print("测试节点 目前的epoch", epoch)
            #test_metris = MetricAccumulator(self.class_number)
            indicies = list(range(multihdfdataset.__len__()))
            random.shuffle(indicies)
            criterion = nn.MSELoss()
            test_kl=0.0
            test_metris=MetricAccumulator(num_classes=self.class_number)
            for j in indicies:
                hdf5dataset=multihdfdataset[j]
                dataloader = hdf5dataset.get_batches(if_shuffle=True)
                # global_dataset = hdf5dataset.get_globaldata()
                # for keys in global_dataset:
                #     global_dataset[keys] = global_dataset[keys].to(self.device)
                # kappa = global_dataset["kappa"]
                self.optim.zero_grad()
                loss = torch.tensor([0.0]).to(self.device)
                for subdomain_data, subdomain_name in dataloader:
                    batch_size = len(subdomain_name)  # 获取实际加载时的batch
                    for keys in subdomain_data:
                        subdomain_data[keys] = subdomain_data[keys].to(self.device)

                    tho_batch = self.gettho(batch_size)
                    #print("test point tho_batch is",tho_batch)
                    # tho_batch=torch.zeros_like(tho_batch)
                    labels = hdf5dataset.get_labels(subdomain_name, tho_batch,if_need_grad=True).to(self.device)
                    labels_ml=labels/subdomain_data['m_l'].unsqueeze(-1)
                    #print("测试节点： labels:",labels)
                    # print("测试节点 labels的形状为",labels.shape)
                    # print("test point labels_ml形状为",labels_ml.shape)
                    loss+=mmd_loss(labels)
                    with torch.no_grad():
                        test_kl+=kl_uniform_loss(labels_ml)
                        dis_labels = get_discrete_labels(labels=labels, ml=subdomain_data['m_l'],feature_dim=self.class_number,if_require_grads=False).to(self.device)
                        # print("test point dislabels is ",dis_labels)
                        # print("test point labels is ",labels)
                        test_metris.update(preds=torch.zeros((dis_labels.shape[0],dis_labels.shape[1],self.class_number)),targets=dis_labels)
                    # outputs = self.network(subdomain_data, global_dataset, tho_batch)
                    # print("test point outputs is",outputs)
                    #
                    # labels_flat = dis_labels.view(-1)  # shape: (batch_size * k_list,)
                    #
                    # # 如果有梯度依赖（即需要反向传播），确保 labels 是 float/int 类型的 index tensor
                    # if labels_flat.dtype not in [torch.int64, torch.int32]:
                    #     labels_flat = labels_flat.int()
                    #
                    # # 统计标签出现频次
                    # counts = torch.bincount(labels_flat, minlength=self.class_number).float()
                    #
                    # # 归一化为概率分布
                    # probs = counts / counts.sum()
                    #
                    # # 构造均匀分布目标
                    # target = torch.full_like(probs, 1.0 / self.class_number)

                    # 避免 log(0)
                    # probs = torch.clamp(probs, min=1e-8)
                    #
                    # # 计算 KL 散度
                    # loss+=F.kl_div(probs.log(), target, reduction='batchmean')
                    #return F.kl_div(probs.log(), target, reduction='batchmean')
                    #probs = generator(batch_size)  # (batch_size, k, 1)

                    # 计算均匀分布损失（两种方式结合）
                    # # 1. 平均分布接近均匀
                    # mean_probs = dis_labels.mean(dim=0).squeeze()
                    # uniform_target = torch.ones(self.k_list) / self.k_list
                    # mse_loss = F.mse_loss(mean_probs, uniform_target)
                    #
                    # # 2. 最大化个体分布熵
                    # entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                    # entropy_loss = -entropy.mean()  # 最大化熵即最小化负熵
                    #
                    # # 组合损失
                    # total_loss = mse_loss + 0.1 * entropy_loss
                    # print("test point dislabels' shape is ",dis_labels.shape)
                    # print("test point dislabels is ",dis_labels)

                    # print("test point dislabels's if_grad is ",dis_labels.requires_grad)
                    # print("test point labels's if_grad is ", labels.requires_grad)
                    # print("test point thobatch's if_grad is ", tho_batch.requires_grad)
                    # expect_label=torch.randint(low=0, high=self.class_number-1, size=dis_labels.shape, dtype=torch.float32)
                    # # dis_labels_onehot=get_discrete_labels_one_hot(labels=labels, ml=subdomain_data['m_l'], feature_dim=self.output_dim).to(self.device)
                    # # print("测试节点 outputs形状",outputs.shape)
                    # # print("测试节点 ")
                    # # test_metris.update(outputs, dis_labels)
                    # # print("测试节点 output形状为",outputs.shape)
                    # # mseloss=self.mseloss(torch.argmax(outputs,dim=-1),dis_labels.to(torch.float32))
                    # # outputs = outputs.permute(0, 2, 1)#用于协调以计算交叉熵损失函数
                    #
                    # # print("测试节点:batch_size,tho_batch.size(1)",batch_size,tho_batch.size(1))
                    # # loss += self.loss(outputs, dis_labels)
                    #loss+=criterion(dis_labels.to(torch.float32),expect_label)
                    loss_every_epoch += loss.item()

                loss.backward()
                self.optim.step()
            print("test point epoch", epoch, "'s loss is ", loss_every_epoch)
            print("test point epoch", epoch, "'s kl is ", test_kl)
            print("test point epoch", epoch, "'s 每类出现的次数为",test_metris.real_class_numbers)
            if test_kl<best_kl:
                best_kl=test_kl
                self.best_dict=copy.deepcopy(self.network.state_dict())
def kl_uniform_loss(y, bins=50):
    hist = torch.histc(y, bins=bins, min=0.0, max=1.0)
    # print("test point hist's requires_grad is ",hist.requires_grad)
    p = hist / torch.sum(hist)
    q = torch.ones_like(p) / bins
    kl = torch.sum(p * torch.log((p + 1e-6) / (q + 1e-6)))
    # print("test point kl's shape is ",kl.shape," and requires_grad is ",kl.requires_grad)
    # print("test point kl is ",kl)
    return kl

def differentiable_kl_uniform(y, bins=50, sigma=0.01):#可微的与均匀分布的KL
    p = soft_histogram(y, bins=bins, sigma=sigma)  # model distribution
    q = torch.ones_like(p) / bins                  # uniform target

    kl = torch.sum(p * torch.log((p + 1e-8) / (q + 1e-8)))
    return kl


def mmd_loss(Y, n_ref=1000):
    ref = torch.rand(n_ref, device=Y.device)
    Y = Y.view(-1, 1)
    ref = ref.view(-1, 1)

    def kernel(x, y, sigma=0.1):
        return torch.exp(-((x - y.T) ** 2) / (2 * sigma ** 2))

    K_yy = kernel(Y, Y).mean()
    K_rr = kernel(ref, ref).mean()
    K_yr = kernel(Y, ref).mean()

    return K_yy + K_rr - 2 * K_yr


def soft_histogram(y, bins=50, sigma=0.01):
    """
    y: Tensor of shape [batch, klist] or [N]
    returns: soft histogram of shape [bins]
    """
    y = y.view(-1, 1)  # flatten to [N, 1]
    bin_edges = torch.linspace(0.0, 1.0, bins, device=y.device).view(1, -1)  # [1, bins]

    # Gaussian kernel centered at bin edges
    weight = torch.exp(-0.5 * ((y - bin_edges) / sigma) ** 2)  # [N, bins]
    weight = weight / (sigma * (2 * torch.pi) ** 0.5)  # optional normalization

    hist = weight.sum(dim=0)
    hist = hist / hist.sum()  # Normalize to form a probability distribution

    return hist  # [bins], differentiable w.r.t y
def get_discrete_labels(labels, ml, feature_dim,if_require_grads=False):
    # 归一化处理（保持原始逻辑）
    #print(ml)
    normalized = labels / ml.unsqueeze(-1)  # [batch, k]
    #print(normalized)
    # 生成中点坐标（保持原始逻辑）
    mid_points = (torch.arange(feature_dim,
                               dtype=torch.float32,
                               device=labels.device) + 0.5) / feature_dim
    mid_points = mid_points.to(normalized.dtype)

    # 计算距离并获取索引（核心修改部分）
    distances = torch.abs(normalized.unsqueeze(-1) - mid_points)  # [batch, k, C]
    #print("test point distances'(in discrete labels) if_grad is ", distances.requires_grad)
    indices = torch.argmin(distances, dim=-1)  # [batch, k]
    if if_require_grads:
        #利用softmax模拟argmin操作,（温度越低越接近离散 argmin）
        temperature=0.0001
        weights = torch.softmax(-distances /temperature, dim=-1)
        # print("test point weights' shape is",weights.shape)
        # print("test point weights is ",weights)
        # 生成索引位置（例如 [0, 1, 2, ..., n-1]）
        positions = torch.arange(distances.size(-1), device=distances.device)
        # 加权求和得到连续索引
        indices_1 = torch.sum(weights * positions, dim=-1)
        # print("test point indices_1",indices_1)
        return indices_1
    # 直接返回类别索引而不是one-hot
    return indices.to(torch.long)  # 确保输出为整数类型

if __name__=="__main__":#用于测试本文件中写的类的代码
    # 示例用法
    # dataset = HDF5Dataset("data0.h5", global_data_keys=['H', 'tho'], subdomain_data_keys=['eigval', 'real_part_eigvec'], if_big_data=False)
    # global_data = dataset.get_globaldata()  # 获取 global 数据
    # subdomain_data = dataset[0]  # 获取第一个 subdomain_i 数据
    #
    # print(global_data.keys())  # 输出 global 数据的键
    # print(subdomain_data.keys())  # 输出 subdomain 数据的键
    dataset = MultiHDF5Dataset_for_eigenvale_number(r"D:\pycharm_project\NN_for_DDM\data\low_frequency_interatation\train_data", "data", global_data_keys=['H', 'tho'],
                               subdomain_data_keys=['number_of_eigval_in_found','blk_size','m_l'])

    # for i in range(dataset.__len__()):
    #     hdf5_dataset = dataset[i]  # 访问第一个 HDF5 文件
    #     global_data = hdf5_dataset.get_globaldata()  # 获取 global 数据
    #     subdomain_data = hdf5_dataset[hdf5_dataset.__len__()-1]  # 获取该文件中的第一个 subdomain 数据
    #     print("sub_domain_0:", subdomain_data)
    #     print(global_data.keys())  # 输出 global 数据键
    #     print(subdomain_data.keys())  # 输出 subdomain 数据键
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hdf5_dataset = dataset[5]
    global_data = hdf5_dataset.get_globaldata()  # 获取 global 数据
    print(f"Golbal数据为{global_data}")
    input_dim=32
    output_dim=1
    layer_number=4
    layer_size=[32,64,128,32]
    epochs=40
    act_function="sigmoid"
    network=MLP(input_dim=input_dim,output_dim=output_dim,layer_number=layer_number,layer_size=layer_size,act_fun=act_function)
    tho_generator=Tho_generator_network(k_list=20,network=network,input_dim=input_dim,class_number=20,model_pth="./tho_generator_model_pth/20250513_114733model_best_kl.pth")
    tho_generator.train_for_data(multihdfdataset=dataset,epochs=epochs)
    if_save=input("是否需要保存网络结构")
    if if_save=='y' or if_save=="yes":
        path_dir="tho_generator_model_pth"
        path_dir=os.path.abspath(path_dir)
        os.makedirs(path_dir,exist_ok=True)
        path_dir_last=os.path.join(path_dir,f"{timestamp}model{epochs}.pth")
        path_dir_best = os.path.join(path_dir, f"{timestamp}model_best_kl.pth")
        torch.save(tho_generator.network.state_dict(),path_dir_last)
        torch.save(tho_generator.best_dict, path_dir_best)
    # for i in range(hdf5_dataset.__len__()):
    #
    #     tho_test=torch.rand((1,3,1))
    #     print("test point tho_test",tho_test)
    #     subdomain_data ,subdomain_name= hdf5_dataset[i]  # 获取该文件中的第一个 subdomain 数据
    #     print(subdomain_name)
    #     print("测试节点 经过soft后 get_labels",hdf5_dataset.get_labels([subdomain_name],tho_test,if_need_grad=True))
    #     print("测试节点 get_labels", hdf5_dataset.get_labels([subdomain_name], tho_test, if_need_grad=False))
    #     #print("test point dis_label",get_discrete_labels(labels=hdf5_dataset.get_labels([subdomain_name],tho_test),ml=subdomain_data["m_l"],feature_dim=20))
    #     print(f"sub_domain_{i}:", subdomain_data)
    #
    #     print(global_data.keys())  # 输出 global 数据键
    #     print(subdomain_data.keys())  # 输出 subdomain 数据键


    # for i in range(1):
    #     print(f"i={i}\n")
    #     hdf5data=dataset[i]
    #     dataloader_hdf5data=hdf5data.get_batches()
    #     for subdomain_data ,caogao_data in dataloader_hdf5data:
    #         print(subdomain_data)
    #         print(f"草稿数据为{caogao_data}")
    #         print(f"草稿数据的数据类型{type(caogao_data)}")
    #         print(f"草稿数据的长度为：{len(caogao_data)}")
    #         print(caogao_data[0])
    #         caogao_data=list(caogao_data)
    #         batch_size=len(caogao_data)
    #         k_list=2
    #         tho_inputs = torch.randn(batch_size, k_list)
    #         #tho_inputs=torch.ones_like(tho_inputs)
    #         labels=hdf5data.get_labels(caogao_data,tho_inputs)
    #         print(labels)
    #         # print(get_discrate_labels(labels,subdomain_data['m_l'],5))
    #         discrate_labels=get_discrete_labels(labels=labels,ml=subdomain_data['m_l'],feature_dim=5)
    #         print("测试节点 labels形状",labels.shape)
    #         print("测试节点 离散label形状",discrate_labels.shape)
    #         print(discrate_labels)
    # dataset = MultiHDF5Dataset_for_eigenvale_number(
    #     r"D:\pycharm_project\NN_for_DDM\data\low_frequency_interatation\train_data", "data",
    #     global_data_keys=['H', 'tho'],
    #     subdomain_data_keys=['m_l'])
    # feature_dim=8
    # output_dim=20
    # layer_number=2
    # layer_size=[16,16]
    # k_list=10
    # tho_network=MLP(input_dim=feature_dim,output_dim=1,layer_number=layer_number,layer_size=layer_size,act_fun="sigmoid")
    # model_tho=Tho_generator_network(k_list=k_list,network=tho_network,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),input_dim=feature_dim,class_number=output_dim)
    # model_tho.train_for_data(multihdfdataset=dataset,loss=None)
