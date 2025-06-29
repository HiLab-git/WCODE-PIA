import torch
from torch.nn import functional as F
from torch.nn import Module

def wce(logits, target, weight, hard_target=True):
        if hard_target:
            target = target.long()
        weight_ce = F.cross_entropy(logits, target,reduction ="none")*weight
        reduce_axes = list(range(1, weight.ndim))
        sum_ce = weight_ce.sum(reduce_axes) / weight.sum(reduce_axes)
        return sum_ce.mean()
    
def wdice( logits, onehot_y, weight):
    weight = weight[:, None]
    prob = logits.softmax(1)
    onehot_y = onehot_y.float()
    intersect = torch.sum(prob * onehot_y * weight)
    y_sum = torch.sum(onehot_y * onehot_y * weight)
    z_sum = torch.sum(prob * prob * weight)
    loss = (2 * intersect + 1e-5) / (z_sum + y_sum + 1e-5)
    loss = 1 - loss
    return loss
    

class WeightedDiceCE(Module):
    def __init__(self):
        super().__init__()
    
    def onehot(self, y, shp_x):
        gt = y.long()
        y_onehot = torch.zeros(shp_x, device=y.device)
        y_onehot.scatter_(1, gt[:,None], 1)    
        return y_onehot
    
    def forward(self,logits, y, weight):
        '''
        x: B x C x D x H x W
        y: B x D x H x W
        weight: B x D x H x W
        '''
        ce_loss = wce(logits, y, weight)
        onehot_y = self.onehot(y,logits.shape)
        dice_loss = wdice(logits, onehot_y, weight)
        return 0.5 * (ce_loss + dice_loss)
        
        
        