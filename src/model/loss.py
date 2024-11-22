import torch
import torch.nn.functional as F

def CharbonnierLoss(input, target, eps=1e-6, reduction='mean'):
    diff = input - target
    loss = torch.sqrt(diff * diff + eps)
    if reduction == 'sum':
        loss = torch.sum(loss)
    elif reduction == 'mean':
        loss = torch.mean(loss)
    else:
        raise NotImplementedError
    return loss

def CosineSimilarityLoss(input, target, eps=1e-8):
    diff = F.cosine_similarity(input, target, dim=1, eps=eps)
    loss = 1.0 - diff.mean()
    return loss

def VanillaGANLoss(input, status, reduction='mean'):
    target = torch.empty_like(input).fill_(int(status))
    loss = F.binary_cross_entropy_with_logits(input, target, reduction=reduction)
    return loss
