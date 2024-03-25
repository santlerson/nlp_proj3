import torch
import torch.nn as nn


class ExponentialRelativePositionalEncoding(nn.Module):
    def forward(self, x):
        round_nums = torch.arange(0, x.size(1)).to(x.device)
        exp_r = torch.exp(round_nums)
        exp_minus_r = torch.exp(-round_nums)
        x = torch.cat([x, exp_r.repeat(x.size(0), 1).unsqueeze(-1), exp_minus_r.repeat(x.size(0), 1).unsqueeze(-1)],
                      dim=-1)
        return x

    def added_dims(self):
        return 2
