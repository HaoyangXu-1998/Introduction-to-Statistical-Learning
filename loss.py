import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, classes, alpha=None, gamma=2, size_average=False):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(classes, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.classes = classes
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)
        mask = inputs.data.new(N, C).fill_(0)
        mask = Variable(mask)
        ids = targets.view(-1, 1)
        mask.scatter_(1, ids.long(), 1.)
        # if inputs.is_cuda and not self.alpha.is_cuda:
        #     self.alpha = self.alpha.cuda()
        # alpha = self.alpha[ids.data.view(-1)]
        alpha = self.alpha[ids.view(-1).long()]
        probs = (P * mask).sum(1).view(-1, 1)
        logp = probs.log()
        batchloss = -alpha * (torch.pow(1 - probs, self.gamma)) * logp
        if self.size_average:
            loss = batchloss.mean()
        else:
            loss = batchloss.sum()
        return loss

class ExpandMSELoss(nn.Module):
    def __init__(self):
        super(ExpandMSELoss, self).__init__()
        self.gamma = 4
        self.mse = nn.MSELoss()

    def forward(self, inputs, targets):
        factor = targets ** (1/self.gamma)
        inputs *= factor
        targets *= factor
        return self.mse(inputs,targets)
        
class CombinedLoss(nn.Module):
    def __init__(self, reg_loss, clsf_loss, alpha=10, size_average=True, strategy=None):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.size_average = size_average
        self.reg_loss = reg_loss
        self.clsf_loss  = clsf_loss
        self.strategy = strategy

    def forward(self, clsf, reg, targets, epoch, train_flag=False):
        if train_flag == False:
            pr = torch.argmax(clsf,1).float().mean()
            batch_loss = (1-pr)*self.clsf_loss(clsf, targets[:, 0].long()) + pr*self.alpha * self.reg_loss(reg, targets[:, 1].unsqueeze(1))
            loss_type = "mix"
        else:
            if self._is_clsf_epoch(epoch):
                batch_loss = self.clsf_loss(clsf, targets[:, 0].long())
                loss_type = "clsf"
            elif self._is_reg_epoch(epoch):
                batch_loss = self.reg_loss(reg, targets[:, 1].unsqueeze(1))
                loss_type = "reg"
            elif self._is_mix_epoch(epoch):
                pr = torch.argmax(clsf,1).float().mean()
                batch_loss = (1-pr)*self.clsf_loss(clsf, targets[:, 0].long()) + pr*self.alpha * self.reg_loss(reg, targets[:, 1].unsqueeze(1))
                loss_type = "mix"
            else:
                raise RuntimeError("undefined epoch num!")
        return batch_loss, loss_type
    
    def _is_clsf_epoch(self,epoch):
        if epoch >= self.strategy["clsf_begin_epoch"] and epoch < self.strategy["clsf_end_epoch"]:
            return True
        else:
            return False

    def _is_reg_epoch(self,epoch):
        if epoch >= self.strategy["reg_begin_epoch"] and epoch < self.strategy["reg_end_epoch"]:
            return True
        else:
            return False

    def _is_mix_epoch(self,epoch):
        if epoch >= self.strategy["mix_begin_epoch"] and epoch < self.strategy["mix_end_epoch"]:
            return True
        else:
            return False