from environments import better_environment
from torch import nn
from consts import *
from positional import ExponentialRelativePositionalEncoding
import torch


class ComplexAttentionPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.positional_encoder = ExponentialRelativePositionalEncoding()
        begin_round_dim = config["REVIEW_DIM"] + self.positional_encoder.added_dims()
        end_round_dim = begin_round_dim + END_ROUND_ADDITIONAL_DIM
        hidden_dim = config["hidden_dim"]
        self.fc_in = nn.Sequential(nn.Linear(begin_round_dim, hidden_dim * config["transformer_nheads"]),
                                   nn.ReLU(),
                                   nn.Dropout(config["dropout"])).double()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim * config["transformer_nheads"], vdim=end_round_dim,
                                               kdim=end_round_dim,
                                               num_heads=config["transformer_nheads"], batch_first=True,
                                               dropout=config["dropout"])
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim * config["transformer_nheads"],
                                                        nhead=config["transformer_nheads"],
                                                        dim_feedforward=hidden_dim, dropout=config["dropout"],
                                                        batch_first=True, )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=config["layers"], )
        self.fc_out = nn.Sequential(nn.Linear(hidden_dim * config["transformer_nheads"] + begin_round_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Dropout(config["dropout"]),
                                    nn.Linear(hidden_dim, 2),
                                    nn.LogSoftmax(dim=-1)).double()

    def forward(self, vectors, padding_mask=None, **kwargs):
        begin_round = vectors["begin_round"]
        end_round = torch.cat([begin_round, vectors["end_round"]], dim=-1)
        begin_round = self.positional_encoder(begin_round).double()
        end_round = self.positional_encoder(end_round).double()
        labels = vectors["labels"].double()
        past_mask = torch.triu(torch.ones(end_round.size(1), end_round.size(1), device=end_round.device)).bool().to(
            end_round.device
        )

        output = self.fc_in(begin_round)
        output, _ = self.attention(output, end_round, end_round, key_padding_mask=padding_mask,
                                   attn_mask=past_mask)
        output[output.isnan()] = 0
        causal_mask = (torch.tril(
            torch.ones(output.size(1), output.size(1),
                       device=output.device))==0).to(output.device)
        output = self.transformer_encoder(output, src_key_padding_mask=padding_mask,
                                          mask=causal_mask, )
        output = torch.cat([output, begin_round], dim=-1)
        output = self.fc_out(output)
        return {"output": output}

    def predict_proba(self, data, update_vectors: bool, vectors_in_input=False):
        assert not update_vectors
        output = self(data)
        output["proba"] = torch.exp(output["output"].flatten())
        return output


class ComplexAttentionEnvironment(better_environment.BetterEnvironment):
    def predict_proba(self, data, update_vectors: bool, vectors_in_input=False):
        if vectors_in_input:
            output = self.model(data)
        else:
            raise NotImplementedError
            # output = self.model({**data, "user_vector": self.currentDM, "game_vector": self.currentGame})
        output["proba"] = torch.exp(output["output"].flatten())
        # if update_vectors:
        #     self.currentDM = output["user_vector"]
        #     self.currentGame = output["game_vector"]
        return output

    def init_user_vector(self):
        self.currentDM = self.model.init_user()

    def init_game_vector(self):
        self.currentGame = self.model.init_game()

    def get_curr_vectors(self):
        return {"user_vector": 888, }

    def init_model_arc(self, config):
        self.model = ComplexAttentionPredictor(config=config).double()
