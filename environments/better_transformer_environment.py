
from torch import nn
import torch
import torch.nn.modules.transformer
from environments import transformer_env
from consts import *
POSITIONAL_DIMS = 2
class TransformerPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config["input_dim"]
        dropout = config["dropout"]
        nhead = config["transformer_nheads"]
        hidden_dim = config["hidden_dim"]
        self.user_vectors = None

        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim-POSITIONAL_DIMS),
                                nn.Dropout(dropout),
                                nn.ReLU(),
                                ).double()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout,
                                                        batch_first=True)
        self.main_task = nn.TransformerEncoder(self.encoder_layer, num_layers=config["layers"]).double()

        self.main_task_classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
                                                  nn.ReLU(),
                                                  nn.Linear(hidden_dim // 2, 2),
                                                  nn.LogSoftmax(dim=-1)).double()

    def forward(self, vectors, padding_mask=None, **kwargs):
        x = vectors["x"]
        x = self.fc(x)
        output = []
        # for i in range(DATA_ROUNDS_PER_GAME):
        #     time_output = self.main_task(x[:, :i + 1].contiguous())[:, -1, :]
        #     output.append(time_output)
        # output = torch.stack(output, 1)
        mask = self.create_causal_mask(x)
        round_nums = torch.arange(0, x.size(1)).to(x.device)
        exp_r = torch.exp(round_nums)
        exp_minus_r = torch.exp(-round_nums)
        x = torch.cat([x, exp_r.repeat(x.size(0), 1).unsqueeze(-1), exp_minus_r.repeat(x.size(0), 1).unsqueeze(-1)], dim=-1)
        output = self.main_task(x, mask=mask, src_key_padding_mask=padding_mask)
        output = self.main_task_classifier(output)
        return {"output": output}

    def predict_proba(self, data, update_vectors: bool, vectors_in_input=False):
        assert not update_vectors
        output = self(data)
        output["proba"] = torch.exp(output["output"].flatten())
        return output

    @staticmethod
    def create_causal_mask(x):
        mask = torch.tril(torch.ones(x.size(1), x.size(1))).to(x.device)
        return mask == 0


class BetterTransformerEnv(transformer_env.transformer_env):
    def init_model_arc(self, config):
        self.model = TransformerPredictor(config=config).double()