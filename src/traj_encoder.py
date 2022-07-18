
'''
----------------------------------------------------------------------
 BSD 3-Clause License

 Copyright (c) 2022, Jonas Wurst
 All rights reserved.
----------------------------------------------------------------------

'''
import torch
from torch import nn
from x_transformers import ContinuousTransformerWrapper, Encoder

class traj_encoder(nn.Module):
    def __init__(self, *, dim_in, dim_out, depth, heads, dim_trans, dim_mlp):
        super().__init__()
        self.emb_token = nn.Parameter(torch.randn(1, 1, dim_in))
        self.model = ContinuousTransformerWrapper(
            dim_in = dim_in,
            max_seq_len = 31,
            attn_layers = Encoder(
                dim = dim_trans,
                depth = depth,
                heads = heads
            )
        )
        self.mlp_head = nn.Sequential(
            nn.Linear(dim_trans, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, dim_out)
        )
    def forward(self,x):
        seq_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        emb_tokens = self.emb_token.expand(seq_unpacked.shape[0], -1, -1)
        seq_unpacked = torch.cat((emb_tokens, seq_unpacked), dim=1)
        mask = (torch.arange(seq_unpacked.shape[1])[None, :] < lens_unpacked[:, None]+1).to(seq_unpacked.get_device())
        x = self.model.forward(seq_unpacked,mask=mask)
        return self.mlp_head(x[:,0,:])