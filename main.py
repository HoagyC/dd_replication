import math
from warnings import WarningMessage
import seaborn as sns

import torch
import torch.nn as nn
from tqdm import tqdm

import wandb
from jaxtyping import Float
from dataclasses import dataclass

"""
weight decay = 1e-2
cosine annealing
input uniform [0,1]
rescale ||x||^2 = 1
xavier init (uniform or normal?)
loss fn in meansq error
adamW
50,000 full batch updates
lr schedule 2,500 warmup to 1e-3, cosine decay to zero
"""

N_FEATURES = 10_000
SPARSITY = 0.999
WEIGHT_DECAY = 1e-2
N_BATCHES = 50_000
N_LR_WARMUP_STEPS = 2_500

class DDModel(nn.Module):
    def __init__(self, n_features: int, hidden_width: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.empty(n_features, hidden_width))
        self.biases = nn.Parameter(torch.empty(n_features))
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.biases)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.matmul(x, self.weights)
        output = torch.relu(torch.matmul(hidden, self.weights.T) + self.biases)
        return hidden, output
    
@dataclass
class TrainResults:
    model_dict: dict[str, torch.Tensor]
    inputs: torch.Tensor
    weight_capacities: torch.Tensor
    hidden_capacities: torch.Tensor
    # inputs
    hidden_dim: int
    n_datapoints: int

@torch.no_grad()
def calculate_capacities(mat: Float[torch.Tensor, "objects width"]):
        sizes = torch.norm(mat, dim=1) ** 2 # [objects]
        normalized_mat = mat / torch.norm(mat, dim=1)[:, None]
        cosine_sims = torch.matmul(normalized_mat, mat.T) ** 2 # [objects, objects]
        total_cos_sims = torch.sum(cosine_sims, dim=1) # [objects]
        return sizes / total_cos_sims
    
    
def lr_schedule_creator(warmup_steps: int, total_steps: int):
    """
    warmup 2500 steps
    to 1e-3
    cosine decay to zero
    at 50,000 steps
    """
    assert warmup_steps < total_steps
    def lr_schedule(epoch):
        if epoch < warmup_steps:
            return epoch / warmup_steps
        
        decay_len = total_steps - warmup_steps
        n_steps = epoch - warmup_steps
        return (1 + math.cos(math.pi * n_steps / decay_len)) / 2

    return lr_schedule


def train_model(n_datapoints: int, hidden_dim: int) -> TrainResults:
    wandb.init(project="double_descent")
    model = DDModel(N_FEATURES, hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    scheduler_fn = lr_schedule_creator(N_LR_WARMUP_STEPS, N_BATCHES)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_fn)
    
    # initialize the inputs
    inputs = create_inputs(n_datapoints)
    
    # loss_fn = nn.MSELoss()   
    # for epoch in tqdm(range(N_BATCHES)):
    #     optimizer.zero_grad()
    #     hidden, output = model(inputs)
    #     loss = loss_fn(output, inputs)
    #     loss.backward()
    #     optimizer.step()
    #     scheduler.step()
        
    #     wandb.log({
    #         "epoch": epoch,
    #         "loss": loss.item(),
    #         "lr": scheduler.get_last_lr()[0],
    #         "weight_norm": torch.norm(model.weights).item(),
    #         "bias_norm": torch.norm(model.biases).item(),
    #         "hidden_norm": torch.norm(hidden).item(),
    #     })
    
    hidden, _ = model(inputs)
    weight_capacities = calculate_capacities(model.weights)
    hidden_capacities = calculate_capacities(hidden)
    model.eval()
    model_state_dict = model.state_dict()
    return TrainResults(model_state_dict, inputs.detach(), weight_capacities, hidden_capacities, hidden_dim, n_datapoints)


def create_inputs(n_datapoints: int) -> torch.Tensor:
    rand_sparsity = torch.rand(n_datapoints, N_FEATURES)
    inputs = torch.rand(n_datapoints, N_FEATURES)
    inputs[rand_sparsity < SPARSITY] = 0
    input_magnitudes = torch.norm(inputs, dim=1)
    input_magnitudes[input_magnitudes == 0] = 1
    return inputs / input_magnitudes[:, None]


def main():
    train_model(hidden_dim=2)
    
        
if __name__ == "__main__":
    main()