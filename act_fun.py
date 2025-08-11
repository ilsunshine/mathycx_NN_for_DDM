#���켤���
import torch
import torch.nn as nn


class ReLUPower(nn.Module):
    #relu��k�η�
    def __init__(self, k=2):
        super(ReLUPower, self).__init__()
        self.k = k  # k ��ʾ ReLU ���ݴ�

    def forward(self, x):
        return torch.pow(torch.relu(x),self.k)
class Leaky_ReLUPower(nn.Module):
    def __init__(self, k=2):
        super(Leaky_ReLUPower, self).__init__()
        self.k = k  # k ��ʾ ReLU ���ݴ�
        self.act=nn.LeakyReLU()

    def forward(self, x):
        return torch.pow(self.act(x),self.k)
class cReLuPower(nn.Module):
    #�������Relu��k�η����ֱ��ʵ�����鲿����
    def __init__(self,k=2):
        super(cReLuPower, self).__init__()
        self.k=k
        self.relu=nn.ReLU()
    def forward(self,x):
        return torch.pow(torch.complex(self.relu(torch.real(x)),self.relu(torch.imag(x))),self.k)


