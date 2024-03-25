from environments import better_environment
from torch import nn
from consts import *
from positional import ExponentialRelativePositionalEncoding
import torch
class SimpleAttentionPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.positional_encoder = ExponentialRelativePositionalEncoding()
        begin_round_dim = config["REVIEW_DIM"] + self.positional_encoder.added_dims()
        end_round_dim = begin_round_dim + END_ROUND_ADDITIONAL_DIM
        hidden_dim = config["hidden_dim"]
        self.attention = nn.MultiheadAttention(embed_dim=begin_round_dim, vdim=end_round_dim, kdim=end_round_dim,
                                               num_heads=1, batch_first=True, dropout=config["dropout"])
        self.fc = nn.Sequential(nn.Linear(begin_round_dim+begin_round_dim, hidden_dim),
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
        past_mask = torch.triu(torch.ones(labels.size(1), labels.size(1))).bool().to(labels.device)
        output, weights = self.attention(begin_round, end_round, end_round, key_padding_mask=padding_mask,
                                      attn_mask=past_mask)
        output[output.isnan()] = 0
        output = self.fc(torch.cat([output, begin_round], dim=-1))
        return {"output": output, "weights":weights}



    def predict_proba(self, data, update_vectors: bool, vectors_in_input=False):
        assert not update_vectors
        output = self(data)
        output["proba"] = torch.exp(output["output"].flatten())
        return output


class SimpleAttentionEnvironment(better_environment.BetterEnvironment):
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
        self.model = SimpleAttentionPredictor(config=config).double()


