import torch.nn as nn
import torch

from . import base
from . import functional as F
from  .base import Activation
import numpy as np
class MOFSmoothLoss(base.Loss):

    def __init__(self, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
    
    def _smooth_l1_loss(self, input, target):
        # type: (Tensor, Tensor) -> Tensor
        t = torch.abs(input - target)
        return torch.where(t < 1, 0.5 * t ** 2, t - 0.5)

    def forward(self, y_pr, y_tr):
        y_pr = self.activation(y_pr)
		
        y_pr = y_pr.view(1,-1)
        y_tr = y_tr.view(1,-1)

        mae = torch.mean(self._smooth_l1_loss(y_pr, y_tr))
	
        y_true = torch.where(y_tr>=0.1,torch.ones(y_tr.size()[0]).cuda(),torch.zeros(y_tr.size()[0]).cuda())
        y_pred = torch.where(y_pr>=0.1,torch.ones(y_pr.size()[0]).cuda(),torch.zeros(y_pr.size()[0]).cuda())
		
        fscore_result = F.f_score(
            y_pred, y_true,
            eps=1e-7,
            beta=1,
            threshold=0.5,
            ignore_channels=None)

        result = mae/fscore_result
        return result
		
class MOFLoss(base.Loss):

    def __init__(self, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_tr):
        y_pr = self.activation(y_pr)
		
        y_pr = y_pr.view(1,-1)
        y_tr = y_tr.view(1,-1)
		
        mae = torch.mean(torch.abs(y_tr[y_pr>=0.1]-y_pr[y_pr >= 0.1]))
	
        y_true = torch.where(y_tr>=0.1,torch.ones(y_tr.size()[0]).cuda(),torch.zeros(y_tr.size()[0]).cuda())
        y_pred = torch.where(y_pr>=0.1,torch.ones(y_pr.size()[0]).cuda(),torch.zeros(y_pr.size()[0]).cuda())
		
        fscore_result = F.f_score(
            y_pred, y_true,
            eps=1e-7,
            beta=1,
            threshold=0.5,
            ignore_channels=None)

        result = mae/fscore_result
        return result
    
class MOFLOG1PLoss(base.Loss):

    def __init__(self, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_tr):
        y_pr = self.activation(y_pr)
		
        y_pr = y_pr.view(1,-1)
        y_tr = y_tr.view(1,-1)
		
        mae = torch.mean(torch.abs(y_tr[y_pr>=np.log1p(0.1)]-y_pr[y_pr >= np.log1p(0.1)]))
	
        y_true = torch.where(y_tr>=np.log1p(0.1),torch.ones(y_tr.size()[0]).cuda(),torch.zeros(y_tr.size()[0]).cuda())
        y_pred = torch.where(y_pr>=np.log1p(0.1),torch.ones(y_pr.size()[0]).cuda(),torch.zeros(y_pr.size()[0]).cuda())
		
        fscore_result = F.f_score(
            y_pred, y_true,
            eps=1e-7,
            beta=1,
            threshold=0.5,
            ignore_channels=None)

        result = mae/fscore_result
        return result


class JaccardLoss(base.Loss):

    def __init__(self, eps=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr, y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class DiceLoss(base.Loss):

    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class L1Loss(nn.L1Loss, base.Loss):
    pass

class SmoothL1Loss(nn.SmoothL1Loss, base.Loss):
    pass

class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(nn.BCELoss, base.Loss):
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass
