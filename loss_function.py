import torch
from torch import nn as nn

class cross_entropy_loss(nn.Module):
    #将outputs调整维度后，调用交叉熵
    def __init__(self,label_smoothing=-1,if_update_weights=False):
        super(cross_entropy_loss, self).__init__()
        self.label_smoothing=label_smoothing
        self.if_update_weights=if_update_weights
        if label_smoothing>0:
            self.loss=nn.CrossEntropyLoss(reduction='none',label_smoothing=label_smoothing)
        else:
            self.loss=nn.CrossEntropyLoss(reduction='none')

    def forward(self,outputs,targets):
        outputs=outputs.permute(0, 2, 1)

        return self.loss(outputs,targets).mean()
    def update_weight(self,weights,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu" )):
        if self.label_smoothing>0:
            self.loss=nn.CrossEntropyLoss(reduction='none',label_smoothing=self.label_smoothing,weight=weights).to(device)
        else:
            self.loss=nn.CrossEntropyLoss(reduction='none',weight=weights).to(device)
class cross_entropu_loss_with_mse(nn.Module):
    #交叉熵与mse加权和
    def __init__(self,label_smoothing=-1,alpha=1.0,if_update_weights=False):
        super(cross_entropu_loss_with_mse, self).__init__()
        self.alpha=alpha
        self.label_smoothing=label_smoothing
        self.if_update_weights=if_update_weights
        if label_smoothing>0:
            self.celoss=nn.CrossEntropyLoss(reduction='none',label_smoothing=label_smoothing)
        else:
            self.celoss=nn.CrossEntropyLoss(reduction='none')
        self.mseloss=nn.MSELoss()
    def forward(self,outputs,targets):
        mse=self.mseloss(torch.argmax(outputs,dim=-1),targets.to(torch.float32))
        outputs=outputs.permute(0, 2, 1)

        return self.celoss(outputs,targets).mean()+self.alpha*mse

    def update_weight(self,weights,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        if self.label_smoothing>0:
            self.celoss=nn.CrossEntropyLoss(reduction='none',label_smoothing=self.label_smoothing,weight=weights).to(device)
        else:
            self.celoss=nn.CrossEntropyLoss(reduction='none',weight=weights).to(device)
class MSE_Loss(nn.Module):
    def __init__(self):
        super(MSE_Loss, self).__init__()
        self.loss=nn.MSELoss()
        self.if_update_weights=False
    def forward(self,outputs,targets):
        return self.loss(outputs.squeeze(-1),targets)



