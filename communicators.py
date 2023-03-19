#%%
import torch
import torch as th
from torch import nn
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F
from typing import List

class Communicator(nn.Module):
    def __init__(self, d_model, n_in, **kwargs):
        super().__init__()

    def forward(self, prev_outputs:List[torch.Tensor]):
        '''
        args:
            prev_outputs:
                list of tensors of shape (batch, seq, feature)
                they symbolize a list of outputs of every layer
        '''
        raise NotImplementedError()
    

class LastPass(Communicator):
    def __init__(self, d_model, n_in, **kwargs):
        super().__init__(d_model, n_in)
        pass

    def forward(self, prev_outputs:List[torch.Tensor]):
        return prev_outputs[-1]
      

class WeightedPass(Communicator):
    def __init__(self, d_model, n_in, **kwargs):
        super().__init__(d_model, n_in)
        self.weighter = nn.parameter.Parameter(
            data=torch.tensor([[[1/n_in for _ in range(n_in)]]])
        )

    def forward(self, prev_outputs:List[torch.Tensor]):
        prev_outputs = torch.stack(prev_outputs, dim=3)
        return (prev_outputs * self.weighter).sum(dim=3)
    

class DensePass(Communicator):
    def __init__(self, d_model, n_in, **kwargs):
        super().__init__(d_model, n_in)

        self.lin = nn.Linear(d_model*n_in, d_model)

    def forward(self, prev_outputs:List[torch.Tensor]):
        prev_outputs = torch.cat(prev_outputs, dim=-1)
        return self.lin(prev_outputs)
    

class AttentionPass(Communicator):
    def __init__(self, d_model, n_in, n_heads=1, dropout=0):
        super().__init__(d_model, n_in)

        self.multihead_attn = nn.MultiheadAttention(
            d_model, 
            n_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.learned_queries = nn.parameter.Parameter(
            data=torch.tensor()
        )
    def forward(self, prev_outputs:List[torch.Tensor]):
        prev_outputs = torch.stack(prev_outputs, dim=2)
        batch, seq, _, feature = prev_outputs.shape
        output = self.multihead_attn(
            query=0
        )
        return self.lin(prev_outputs)


# %% 
if __name__ == '__main__':
    pass