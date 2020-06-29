from . import base
from . import functional as F
from .base import Activation
import torch

# from sklearn.metrics import f1_score

# def mae(y_true, y_pred) :
    
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
    
#     y_true = y_true.reshape(1, -1)[0]
    
#     y_pred = y_pred.reshape(1, -1)[0]
    
#     over_threshold = y_true >= 0.1
    
#     return np.mean(np.abs(y_true[over_threshold] - y_pred[over_threshold]))

# def fscore(y_true, y_pred):
    
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
    
#     y_true = y_true.reshape(1, -1)[0]
    
#     y_pred = y_pred.reshape(1, -1)[0]
    
#     remove_NAs = y_true >= 0
    
#     y_true = np.where(y_true[remove_NAs] >= 0.1, 1, 0)
    
#     y_pred = np.where(y_pred[remove_NAs] >= 0.1, 1, 0)
    
#     return(f1_score(y_true, y_pred))

# def maeOverFscore(y_true, y_pred):
    
#     return mae(y_true, y_pred) / (fscore(y_true, y_pred) + 1e-07)


class IoU(base.Metric):
    __name__ = 'iou_score'

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.iou(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )

# class maeOverFscore(base.Metric):
#     def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
#         super().__init__(**kwargs)
#         self.eps = eps
#         self.beta = beta
#         self.threshold = threshold
#         self.activation = Activation(activation)
#         self.ignore_channels = ignore_channels

#     def forward(self, y_pr, y_gt):
#         y_pr = self.activation(y_pr)
#         return F.f_score(
#             y_pr, y_gt,
#             eps=self.eps,
#             beta=self.beta,
#             threshold=self.threshold,
#             ignore_channels=self.ignore_channels,
#         )


class MAEOVERFSCORE(base.Metric):
	def __init__(self, activation=None, ignore_channels=None, **kwargs):
		super().__init__(**kwargs)
		self.activation = Activation(activation)
		self.ignore_channels = ignore_channels

	def forward(self, y_pr, y_tr):
		y_pr = self.activation(y_pr)
		
		y_pr = y_pr.view(1,-1)
		y_tr = y_tr.view(1,-1)
		
		mae = torch.mean(torch.abs(y_tr[y_pr>=0.1]-y_pr[y_pr >= 0.1]))
		
		y_true = torch.where(y_tr[y_tr>=0]>=0.1,torch.ones(y_tr[y_tr>=0].size()[0]).cuda(),torch.zeros(y_tr[y_tr>=0].size()[0]).cuda())
		y_pred = torch.where(y_pr[y_tr>=0]>=0.1,torch.ones(y_pr[y_tr>=0].size()[0]).cuda(),torch.zeros(y_pr[y_tr>=0].size()[0]).cuda())
		
		fscore_result = F.f_score(
			y_pred, y_true,
			eps=1e-7,
			beta=1,
			threshold=0.5,
			ignore_channels=None)

		result = mae/fscore_result
		return result


class Fscore(base.Metric):

    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.f_score(
            y_pr, y_gt,
            eps=self.eps,
            beta=self.beta,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Accuracy(base.Metric):

    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.accuracy(
            y_pr, y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Recall(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.recall(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Precision(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.precision(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
