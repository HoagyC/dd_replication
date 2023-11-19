import json
import math
from pathlib import Path
from typing import Callable, Literal
import fire

import torch
import torch.nn as nn
from jaxtyping import Float
from tqdm import tqdm

import wandb

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

ENABLE_WANDB = False
N_FEATURES = 10_000
SPARSITY = 0.999
WEIGHT_DECAY = 1e-2
N_BATCHES = 50_000
N_LR_WARMUP_STEPS = 2_500
BATCH_SIZE = 128
DATAPOINT_SIZES = [3,5,6,8,10,15,30,50,100,200,500,1000,2000,5000,10000,20000,50000,100000]
EVAL_N_DATAPOINTS = 1_000


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


@torch.no_grad()
def calculate_capacities(mat: Float[torch.Tensor, "objects width"]):
    sizes = torch.norm(mat, dim=1) ** 2  # [objects]
    normalized_mat = mat / torch.norm(mat, dim=1)[:, None]
    cosine_sims = torch.matmul(normalized_mat, mat.T) ** 2  # [objects, objects]
    total_cos_sims = torch.sum(cosine_sims, dim=1)  # [objects]
    return sizes / total_cos_sims


def lr_schedule_creator(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    """
    warmup 2500 steps
    to 1e-3
    cosine decay to zero
    at 50,000 steps
    """
    assert warmup_steps < total_steps

    def lr_schedule(epoch: int) -> float:
        if epoch < warmup_steps:
            return epoch / warmup_steps

        decay_len = total_steps - warmup_steps
        n_steps = epoch - warmup_steps
        return (1 + math.cos(math.pi * n_steps / decay_len)) / 2

    return lr_schedule


def train_model(n_datapoints: int, hidden_dim: int, output_dir: Path, device: torch.device) -> None:
    assert not output_dir.exists(), output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    if ENABLE_WANDB:
        wandb.init(project="double_descent")
    model = DDModel(N_FEATURES, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-3, weight_decay=WEIGHT_DECAY
    )
    scheduler_fn = lr_schedule_creator(N_LR_WARMUP_STEPS, N_BATCHES)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_fn)
    inputs = create_inputs(n_datapoints).to(device)

    for epoch in tqdm(range(N_BATCHES), desc="training"):
        optimizer.zero_grad()
        for i in range(0, n_datapoints, BATCH_SIZE):
            batch_inputs = inputs[i : i + BATCH_SIZE]
            hidden, output = model(batch_inputs)
            loss_multiplier = batch_inputs.shape[0] / inputs.shape[0]
            loss = loss_fn(output, batch_inputs) * loss_multiplier
            loss.backward()
        optimizer.step()
        scheduler.step()

        if ENABLE_WANDB:
            wandb.log(
                {
                    "epoch": epoch,
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "weight_norm": torch.norm(model.weights).item(),
                    "bias_norm": torch.norm(model.biases).item(),
                    "hidden_norm": torch.norm(hidden).item(),
                }
            )

    torch.save(model.state_dict(), output_dir / "model.pt")
    torch.save(inputs, output_dir / "inputs.pt")
    (output_dir / "config.json").write_text(
        json.dumps(
            dict(
                n_datapoints=n_datapoints,
                hidden_dim=hidden_dim,
            )
        )
    )


def load_model(output_dir: Path) -> DDModel:
    config = json.loads((output_dir / "config.json").read_text())
    model = DDModel(n_features=N_FEATURES, hidden_width=config["hidden_dim"])
    model.load_state_dict(torch.load(output_dir / "model.pt"))
    return model


def create_inputs(n_datapoints: int) -> torch.Tensor:
    rand_sparsity = torch.rand(n_datapoints, N_FEATURES)
    inputs = torch.rand(n_datapoints, N_FEATURES)
    inputs[rand_sparsity < SPARSITY] = 0
    input_magnitudes = torch.norm(inputs, dim=1)
    input_magnitudes[input_magnitudes == 0] = 1
    return inputs / input_magnitudes[:, None]


@torch.inference_mode()
def test_model(model: DDModel, batch_size: int, device: torch.device) -> float:
    model.eval()
    eval_inputs = create_inputs(EVAL_N_DATAPOINTS).to(device)
    mean_loss = 0
    for i in tqdm(range(0, EVAL_N_DATAPOINTS, batch_size), desc="eval"):
        batch = eval_inputs[i : i + batch_size]
        _, output = model(batch)
        test_loss = loss_fn(output, batch)
        mean_loss += test_loss.item() * batch.shape[0] / EVAL_N_DATAPOINTS
    return mean_loss
    

def loss_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.sum((predictions - targets) ** 2, dim=1).mean()


def main(device: Literal["cpu", "cuda"] = "cpu"):
    if device == "cuda":
        device = torch.device("cuda")
    elif device == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(device)

    bar = tqdm(DATAPOINT_SIZES, desc="n_datapoints")
    for n_datapoints in bar:
        bar.set_postfix(n_datapoints=n_datapoints)
        path = Path(f"outputs/n_datapoints={n_datapoints}")
        train_model(
            n_datapoints=n_datapoints,
            hidden_dim=2,
            output_dir=path,
            device=device,
        )
        model = load_model(path)
        eval_loss = test_model(model, batch_size=128, device=device)
        print("eval_loss", eval_loss)


if __name__ == "__main__":
    fire.Fire(main)
