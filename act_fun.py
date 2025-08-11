#构造激活函数
import torch
import torch.nn as nn


class ReLUPower(nn.Module):
    #relu的k次方
    def __init__(self, k=2):
        super(ReLUPower, self).__init__()
        self.k = k  # k 表示 ReLU 的幂次

    def forward(self, x):
        return torch.pow(torch.relu(x),self.k)
class Leaky_ReLUPower(nn.Module):
    def __init__(self, k=2):
        super(Leaky_ReLUPower, self).__init__()
        self.k = k  # k 表示 ReLU 的幂次
        self.act=nn.LeakyReLU()

    def forward(self, x):
        return torch.pow(self.act(x),self.k)
class cReLuPower(nn.Module):
    #复数域的Relu的k次方，分别对实部与虚部进行
    def __init__(self,k=2):
        super(cReLuPower, self).__init__()
        self.k=k
        self.relu=nn.ReLU()
    def forward(self,x):
        return torch.pow(torch.complex(self.relu(torch.real(x)),self.relu(torch.imag(x))),self.k)


