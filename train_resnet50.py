import argparse
import json
import os
import random
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet50_Weights, resnet50

from lda import LDAHead, DNLLLoss, LogisticLoss

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


class ResNet50Encoder(nn.Module):
    def __init__(self, dim, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(2048, dim)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.proj(x)


class DeepLDA(nn.Module):
    def __init__(self, num_classes, dim, covariance_type="spherical", pretrained=True):
        super().__init__()
        self.encoder = ResNet50Encoder(dim, pretrained=pretrained)
        self.head = LDAHead(num_classes, dim, covariance_type=covariance_type)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


def set_seed(seed, deterministic=False):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_loaders(dataset_name, data_root, batch_size, test_batch_size, num_workers):
    weights_tfm = ResNet50_Weights.IMAGENET1K_V2.transforms()
    mean = weights_tfm.mean
    std = weights_tfm.std
    pin_memory = torch.cuda.is_available()

    train_steps = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_steps = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    if dataset_name == "FashionMNIST":
        train_steps.append(transforms.Grayscale(num_output_channels=3))
        test_steps.append(transforms.Grayscale(num_output_channels=3))
    train_steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])

    dataset_map = {
        "CIFAR100": datasets.CIFAR100,
        "CIFAR10": datasets.CIFAR10,
        "FashionMNIST": datasets.FashionMNIST,
    }
    dataset_cls = dataset_map[dataset_name]
    train_ds = dataset_cls(root=data_root, train=True, transform=transforms.Compose(train_steps), download=True)
    test_ds = dataset_cls(root=data_root, train=False, transform=transforms.Compose(test_steps), download=True)
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
    pretrained,
    tag,
):
    model = DeepLDA(
        num_classes=num_classes,
        dim=dim,
        covariance_type=covariance_type,
        pretrained=pretrained,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_acc = []
    test_acc = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = acc_sum = n_sum = 0
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
                loss_sum += loss.item() * y.size(0)

        tr_acc = acc_sum / n_sum
        te_acc = evaluate(model, test_ld, device)
        train_acc.append(float(tr_acc))
        test_acc.append(float(te_acc))
        print(
            f"[{tag}][{loss_name} {epoch:03d}/{epochs:03d}] "
            f"train loss={loss_sum / n_sum:.4f} acc={tr_acc:.4f} | test acc={te_acc:.4f}",
            flush=True,
        )

    return train_acc, test_acc


def run_experiments(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets_cfg = {
        "FashionMNIST": {"display_name": "Fashion-MNIST", "num_classes": 10, "dim": 9},
        "CIFAR10": {"display_name": "CIFAR-10", "num_classes": 10, "dim": 9},
        "CIFAR100": {"display_name": "CIFAR-100", "num_classes": 100, "dim": 99},
    }
    selected_datasets = args.datasets or list(datasets_cfg)

    results = {
        "meta": {
            "datasets": [datasets_cfg[name]["display_name"] for name in selected_datasets],
            "epochs": args.epochs,
            "runs": args.runs,
            "batch_size": args.batch_size,
            "test_batch_size": args.test_batch_size,
            "covariance_type": args.covariance_type,
            "encoder": "ResNet50",
            "pretrained": not args.no_pretrained,
            "optimizer": "Adam",
            "lr": args.lr,
            "lambda_reg": args.lambda_reg,
            "seed_base": args.seed_base,
            "deterministic": args.deterministic,
            "device": str(device),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "results": {},
    }

    for dataset_name in selected_datasets:
        dataset_cfg = datasets_cfg[dataset_name]
        num_classes = dataset_cfg["num_classes"]
        dim = args.dim if args.dim is not None else dataset_cfg["dim"]
        train_ld, test_ld = build_loaders(
            dataset_name,
            args.data_root,
            args.batch_size,
            args.test_batch_size,
            args.num_workers,
        )
        results["results"][dataset_name] = {"DNLLLoss": [], "LogisticLoss": []}
        loss_specs = [
            ("DNLLLoss", lambda: DNLLLoss(lambda_reg=args.lambda_reg)),
            ("LogisticLoss", LogisticLoss),
        ]

        for loss_name, loss_factory in loss_specs:
            runs = []
            for run_idx in range(args.runs):
                seed = args.seed_base + run_idx
                set_seed(seed, deterministic=args.deterministic)
                train_acc, test_acc = train_one_run(
                    loss_name=loss_name,
                    loss_fn=loss_factory(),
                    train_ld=train_ld,
                    test_ld=test_ld,
                    device=device,
                    epochs=args.epochs,
                    lr=args.lr,
                    dim=dim,
                    num_classes=num_classes,
                    covariance_type=args.covariance_type,
                    pretrained=not args.no_pretrained,
                    tag=f"{dataset_name}/run{run_idx + 1}",
                )
                runs.append(
                    {
                        "run": run_idx + 1,
                        "seed": seed,
                        "num_classes": num_classes,
                        "embedding_dim": dim,
                        "train_acc": train_acc,
                        "test_acc": test_acc,
                        "final_train_acc": train_acc[-1],
                        "final_test_acc": test_acc[-1],
                    }
                )
            results["results"][dataset_name][loss_name] = runs

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ResNet50 LDAHead with DNLLLoss and LogisticLoss."
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-path", default="results/resnet50_lda_runs.json")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["FashionMNIST", "CIFAR10", "CIFAR100"],
        help="Datasets to train. Defaults to FashionMNIST, CIFAR10, and CIFAR100.",
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--covariance-type", default="spherical")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    parser.add_argument("--seed-base", type=int, default=1234)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_experiments(args)


if __name__ == "__main__":
    main()
