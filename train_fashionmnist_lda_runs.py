import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from lda import LDAHead, DNLLLoss, LogisticLoss


class Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, dim)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.proj(x)


class DeepLDA(nn.Module):
    def __init__(self, num_classes, dim, covariance_type="spherical"):
        super().__init__()
        self.encoder = Encoder(dim)
        self.head = LDAHead(num_classes, dim, covariance_type=covariance_type)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_loaders(data_root, batch_size, test_batch_size, num_workers):
    mean = (0.2860,)
    std = (0.3530,)
    pin_memory = torch.cuda.is_available()

    train_tfm = transforms.Compose(
        [
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_ds = datasets.FashionMNIST(
        root=data_root, train=True, transform=train_tfm, download=True
    )
    test_ds = datasets.FashionMNIST(
        root=data_root, train=False, transform=test_tfm, download=True
    )
    train_ld = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_ld = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_ld, test_ld


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ok = tot = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        ok += (logits.argmax(1) == y).sum().item()
        tot += y.size(0)
    return ok / tot


def train_one_run(
    loss_name,
    loss_fn,
    train_ld,
    test_ld,
    device,
    epochs,
    lr,
    dim,
    num_classes,
    covariance_type,
):
    model = DeepLDA(num_classes=num_classes, dim=dim, covariance_type=covariance_type).to(
        device
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_acc = []
    test_acc = []

    for epoch in range(1, epochs + 1):
        model.train()
        acc_sum = n_sum = 0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                pred = logits.argmax(1)
                acc_sum += (pred == y).sum().item()
                n_sum += y.size(0)
        tr_acc = acc_sum / n_sum
        te_acc = evaluate(model, test_ld, device)
        train_acc.append(float(tr_acc))
        test_acc.append(float(te_acc))
        print(
            f"[{loss_name} {epoch:03d}] train acc={tr_acc:.4f} | test acc={te_acc:.4f}"
        )

    return train_acc, test_acc


def run_experiments(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ld, test_ld = build_loaders(
        args.data_root, args.batch_size, args.test_batch_size, args.num_workers
    )

    results = {
        "config": {
            "dataset": "Fashion-MNIST",
            "epochs": args.epochs,
            "runs": args.runs,
            "batch_size": args.batch_size,
            "test_batch_size": args.test_batch_size,
            "num_classes": args.num_classes,
            "embedding_dim": args.dim,
            "covariance_type": args.covariance_type,
            "optimizer": "Adam",
            "lr": args.lr,
            "lambda_reg": args.lambda_reg,
            "seed_base": args.seed_base,
            "deterministic": args.deterministic,
            "device": str(device),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "losses": {},
    }

    loss_specs = [
        ("DNLLLoss", DNLLLoss(lambda_reg=args.lambda_reg)),
        ("LogisticLoss", LogisticLoss()),
    ]

    for loss_name, loss_fn in loss_specs:
        runs = []
        for run_idx in range(args.runs):
            seed = args.seed_base + run_idx
            set_seed(seed, deterministic=args.deterministic)
            train_acc, test_acc = train_one_run(
                loss_name=loss_name,
                loss_fn=loss_fn,
                train_ld=train_ld,
                test_ld=test_ld,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                dim=args.dim,
                num_classes=args.num_classes,
                covariance_type=args.covariance_type,
            )
            runs.append(
                {
                    "run": run_idx + 1,
                    "seed": seed,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                }
            )
        results["losses"][loss_name] = runs

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Deep LDA with DNLLLoss and LogisticLoss on Fashion-MNIST."
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-path", default="results/fashionmnist_lda_runs.json")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--dim", type=int, default=9)
    parser.add_argument("--covariance-type", default="spherical")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    parser.add_argument("--seed-base", type=int, default=1234)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_experiments(args)


if __name__ == "__main__":
    main()
