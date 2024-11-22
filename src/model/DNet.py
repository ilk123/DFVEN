import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BaseEncoder, BaseEncoderWithoutTail
from .utils import get_patch

class Dnet(nn.Module):
    def __init__(self, opt, **kwargs):
        super(Dnet, self).__init__()
        for kw, args in opt.items():
            setattr(self, kw, args)
        
        for kw, args in kwargs.items():
            setattr(self, kw, args)

        self.encoder_q = BaseEncoder(self.deg_in_nc)
        self.encoder_k = BaseEncoder(self.deg_in_nc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(self.deg_dim, self.deg_K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.deg_m + param_q.data * (1. - self.deg_m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.deg_K % batch_size == 0

        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.deg_K

        self.queue_ptr[0] = ptr
    
    def forward(self, im_q, im_k, is_train=True):
        if is_train:
            embedding, q, degrade = self.encoder_q(im_q)
            q = F.normalize(q, dim=1)
            
            with torch.no_grad():
                self._momentum_update_key_encoder()
                
                _, k, _ = self.encoder_k(im_k)
                k = F.normalize(k, dim=1)

            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            logits = torch.cat([l_pos, l_neg], dim=1)
            logits /= self.deg_T

            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            self._dequeue_and_enqueue(k)
            return embedding, degrade, logits, labels
        else:
            embedding, _, _ = self.encoder_q(im_q)
            return embedding


class DnetWithoutTail(nn.Module):
    def __init__(self, opt, **kwargs):
        super(DnetWithoutTail, self).__init__()
        for kw, args in opt.items():
            setattr(self, kw, args)
        
        for kw, args in kwargs.items():
            setattr(self, kw, args)

        self.encoder_q = BaseEncoderWithoutTail(self.deg_in_nc)
        self.encoder_k = BaseEncoderWithoutTail(self.deg_in_nc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(self.deg_dim, self.deg_K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.deg_m + param_q.data * (1. - self.deg_m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.queue.size()[1]:
            self.queue[:, ptr:ptr+batch_size] = keys.transpose(0, 1)[:, :self.queue.size()[1]-ptr]
            ptr = 0
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
            ptr = (ptr + batch_size) % self.deg_K

        self.queue_ptr[0] = ptr
    
    def forward(self, im_q, im_k, is_train=True):
        if is_train:
            embedding, q = self.encoder_q(im_q)
            q = F.normalize(q, dim=1)
            
            with torch.no_grad():
                self._momentum_update_key_encoder()
                
                _, k = self.encoder_k(im_k)
                k = F.normalize(k, dim=1)

            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            logits = torch.cat([l_pos, l_neg], dim=1)
            logits /= self.deg_T

            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            self._dequeue_and_enqueue(k)
            return embedding, logits, labels
        else:
            embedding, _ = self.encoder_q(im_q)
            return embedding
