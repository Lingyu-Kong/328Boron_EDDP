import torch
import torch.nn as nn
from torch.optim import Adam
from model.mlp import MLP
from torch_scatter import scatter

class EDDP(nn.Module):
    def __init__(
        self,
        input_size,
        output_sizes,
        use_layer_norm=False,
        activation=nn.SiLU,
        dropout=0.0,
        layernorm_before=False,
        use_bn=True,
        lr=1e-3,
        device=torch.device
    ):
        super().__init__()
        self.mlp=MLP(input_size, output_sizes, use_layer_norm, activation, dropout, layernorm_before, use_bn)
        self.optimizer=Adam(self.parameters(), lr=lr)
        self.device=device
        self.to(device)

    def forward(
        self,
        features,
        node_batch,
        num_graphs,
    ):
        e_si=self.mlp(features)
        e_s=scatter(e_si, node_batch, dim=0, reduce="sum", dim_size=num_graphs)
        return e_s
    
    def eval(
        self,
        features,
        node_batch,
        num_graphs,
        mean=0.0,
        std=1.0,
    ):
        e_si=self.mlp(features)
        e_s=scatter(e_si, node_batch, dim=0, reduce="sum", dim_size=num_graphs)
        e_s=e_s*std+mean
        return e_s.view(-1)
    
    def save_model(self,save_path):
        torch.save(self.state_dict(),save_path)
    
    def load_model(self,load_path):
        self.load_state_dict(torch.load(load_path))
        