import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

# class EdgeLoss(nn.Module):
#     def __init__(self):
#         super(EdgeLoss, self).__init__()
#         k = torch.Tensor([[.05, .25, .4, .25, .05]])
#         self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
#         if torch.cuda.is_available():
#             self.kernel = self.kernel.cuda()
#         self.loss = CharbonnierLoss()
#
#     def conv_gauss(self, img):
#         n_channels, _, kw, kh = self.kernel.shape
#         img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
#         return F.conv2d(img, self.kernel, groups=n_channels)
#
#     def laplacian_kernel(self, current):
#         filtered    = self.conv_gauss(current)    # filter
#         down        = filtered[:,:,::2,::2]               # downsample
#         new_filter  = torch.zeros_like(filtered)
#         new_filter[:,:,::2,::2] = down*4                  # upsample
#         filtered    = self.conv_gauss(new_filter) # filter
#         diff = current - filtered
#         return diff
#
#     def forward(self, x, y):
#         loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
#         return loss

class CEL(nn.Module):
    def __init__(self):
        super(CEL, self).__init__()
        #print("You are using `CEL`!")
        self.eps = 1e-6

    def forward(self, pred, target):
        #pred = pred.sigmoid()
        intersection = pred * target
        numerator = (pred - intersection).sum() + (target - intersection).sum()
        denominator = pred.sum() + target.sum()
        return numerator / (denominator + self.eps)

def iou_loss(pred, mask):
    #pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()

############################################ edge loss #################################################
def cross_entropy(logits, labels):
    return torch.mean((1 - labels) * logits + torch.log(1 + torch.exp(-logits)))

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        laplace = torch.FloatTensor([[-1,-1,-1,],[-1,8,-1],[-1,-1,-1]]).view([1,1,3,3])
        # filter shape in Pytorch: out_channel, in_channel, height, width
        self.laplace = nn.Parameter(data=laplace, requires_grad=False).cuda()
    def torchLaplace(self, x):
        edge = F.conv2d(x, self.laplace, padding=1)
        edge = torch.abs(torch.tanh(edge))
        return edge
    def forward(self, y_pred, y_true, mode=None):
        y_true_edge = self.torchLaplace(y_true)
        y_pred_edge = self.torchLaplace(y_pred)
        edge_loss = cross_entropy(y_pred_edge, y_true_edge)
        return edge_loss

######################################## Focal_loss #############################################
class FocalLossV1(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')
        self.class_num = 1

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.class_num):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, logits, label):
        # '''
        # Usage is same as nn.BCEWithLogits:
        #     >>> criteria = FocalLossV1()
        #     >>> logits = torch.randn(8, 19, 384, 384)
        #     >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
        #     >>> loss = criteria(logits, lbs)
        # '''
        #probs = torch.sigmoid(logits)
        probs = logits
        label = self._one_hot_encoder(label)

        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

############################### Floss ############################################################

class FLoss(nn.Module):
    def __init__(self, beta=0.3, log_like=True):
        super(FLoss, self).__init__()
        self.beta = beta
        self.log_like = log_like

    def forward(self, prediction, target):
        EPS = 1e-10
        N = prediction.size(0)
        TP = (prediction * target).view(N, -1).sum(dim=1)
        H = self.beta * target.view(N, -1).sum(dim=1) + prediction.view(N, -1).sum(dim=1)
        fmeasure = (1 + self.beta) * TP / (H + EPS)
        if self.log_like:
            floss = -torch.log(fmeasure)
        else:
            floss  = (1 - fmeasure)
        return floss

####################################################
##### This is focal loss class for multi class #####
####################################################

# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py

class FocalLoss2d(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma=2, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', balance_param=0.25):
        super(FocalLoss2d, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)

        weight = Variable(self.weight)

        # compute the negative likelyhood
        logpt = - F.binary_cross_entropy_with_logits(input, target, pos_weight=weight, reduction=self.reduction)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss