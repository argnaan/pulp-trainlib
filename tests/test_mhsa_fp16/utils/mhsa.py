import numpy as np
from torch import float32
from torch import nn
from torch import Tensor 
from torch import cuda
import torch
import math
from torch.nn import functional as F

#define GIST_A  12102203.17133801f
#define GIST_B  1064986823.010288f
#define GIST_C  8388608
#define GIST_D  2139095040 

def fastexp_gist(x):
    with torch.no_grad():
        x_copy = x.type(torch.float32)
        x_copy = x_copy * 12102203.17133801 + 1064986823.010288
        x_copy = torch.where(x_copy < 8388608, 0, x_copy).type(torch.float32)
        x_copy = torch.where(x_copy > 2139095040, 2139095040, x_copy).type(torch.float32)
        n = x_copy.cpu().numpy().astype(np.uint32).view(np.float32)
        
        result = torch.from_numpy(n)
    
    return result


def own_softmax(x):
    maxes = torch.max(x, -1, keepdim=True)[0]
    maxes = maxes.float()
    x_copy = x.float()
    #maxes = torch.swapaxes(maxes, -2, -1) 
    x_exp = torch.exp((x_copy-maxes))
    x_exp = x_exp.bfloat16()
    x_exp_sum = torch.sum(x_exp, -1, keepdim=True)
    return x_exp/x_exp_sum

def own_softmax_fastexp(x):
    maxes = torch.max(x, -1, keepdim=True)[0]
    maxes = maxes.float()
    x_copy = x.float()
    #maxes = torch.swapaxes(maxes, -2, -1) 
    x_exp = fastexp_gist((x_copy-maxes))
    x_exp = x_exp.bfloat16()
    x_exp_sum = torch.sum(x_exp, -1, keepdim=True)

    return x_exp/x_exp_sum



def q_rsqrt(x):
    with torch.no_grad():
        y = np.asarray((x,), dtype=np.float32)
        x2 = y * 0.5
        i = y.view(np.int32)
        i = np.right_shift(i, 1)
        i = 0x5f3759df - i
        y = i.view(np.float32)
        y = y * (1.5 - (x2 * y * y))

        result = torch.from_numpy(y)

    return result

class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, att_dim):
        super().__init__()
        self.proj_in = nn.Linear(dim, 3*att_dim, bias=False)
        self.proj_out = nn.Linear(att_dim, dim, bias=False)
        self.dim = dim
        self.att_dim = att_dim
        self.n_heads = num_heads
        self.head_dim = att_dim // num_heads
        #self.scaling = q_rsqrt(self.head_dim).bfloat16()
        self.scaling= (1 / math.sqrt(self.head_dim))
        self.scores = None # for visualization
        self.softmax = own_softmax_fastexp

    def forward(self, x, tgt_len):
        q, k, v = self.proj_in(x).chunk(3, dim=-1)
        q = q.contiguous().view(tgt_len, self.n_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, self.n_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, self.n_heads, self.head_dim).transpose(0, 1)

        scores = torch.bmm(q, k.transpose(1, 2))  


        assert list(scores.size()) == [self.n_heads, tgt_len, tgt_len]
        
        scores = scores * self.scaling
        
        scores = self.softmax(scores)

        scores = torch.bmm(scores, v)
        assert list(scores.size()) == [self.n_heads, tgt_len, self.head_dim]

        scores = scores.transpose(0, 1).contiguous().view(tgt_len, self.att_dim)
        self.scores = scores

        h = self.proj_out(scores)

        return h