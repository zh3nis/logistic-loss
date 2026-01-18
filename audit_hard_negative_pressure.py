import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from lda import LDAHead, DNLLLoss, LogisticLoss


class Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
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
    def __init__(self, num_classes, dim):
        super().__init__()
        self.encoder = Encoder(dim)
        self.head = LDAHead(num_classes, dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datasets(data_root):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_tfm = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
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

    train_ds = datasets.CIFAR100(
        root=data_root, train=True, transform=train_tfm, download=True
    )
    test_ds = datasets.CIFAR100(
        root=data_root, train=False, transform=test_tfm, download=True
    )
    return train_ds, test_ds, test_tfm


def build_loaders(train_ds, test_ds, batch_size, test_batch_size, num_workers):
    pin_memory = torch.cuda.is_available()
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


def select_probe_indices(total_size, probe_size, seed):
    size = min(probe_size, total_size)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total_size, generator=generator)
    return perm[:size].tolist()


def build_probe_loaders(data_root, probe_size, probe_seed, batch_size, num_workers, test_tfm):
    pin_memory = torch.cuda.is_available()
    train_probe_ds = datasets.CIFAR100(
        root=data_root, train=True, transform=test_tfm, download=True
    )
    test_probe_ds = datasets.CIFAR100(
        root=data_root, train=False, transform=test_tfm, download=True
    )

    train_idx = select_probe_indices(len(train_probe_ds), probe_size, probe_seed)
    test_idx = select_probe_indices(len(test_probe_ds), probe_size, probe_seed)

    train_subset = Subset(train_probe_ds, train_idx)
    test_subset = Subset(test_probe_ds, test_idx)

    train_ld = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_ld = DataLoader(
        test_subset,
        batch_size=batch_size,
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


@torch.no_grad()
def collect_audit_tensors(model, loader, device, loss_name):
    model.eval()
    g_vals = []
    margin_vals = []
    s_c_star_vals = []
    p_neg_max_vals = []
    p_mix_vals = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        scores = model(x)
        s_y = scores.gather(1, y.unsqueeze(1)).squeeze(1)
        masked = scores.clone()
        masked.scatter_(1, y.unsqueeze(1), float("-inf"))
        s_c_star, _ = masked.max(dim=1)
        margin = s_y - s_c_star

        s_clamped = scores.clamp(-50.0, 50.0)
        s_c_star_clamped = s_c_star.clamp(-50.0, 50.0)
        p_mix = torch.exp(s_clamped).sum(dim=1)
        p_neg_max = torch.exp(s_c_star_clamped)
        if loss_name == "logistic":
            g = torch.sigmoid(s_c_star)
        else:
            g = torch.exp(s_c_star_clamped)

        g_vals.append(g.detach().cpu())
        margin_vals.append(margin.detach().cpu())
        s_c_star_vals.append(s_c_star.detach().cpu())
        p_neg_max_vals.append(p_neg_max.detach().cpu())
        p_mix_vals.append(p_mix.detach().cpu())

    out = {
        "g": torch.cat(g_vals, dim=0),
        "margin": torch.cat(margin_vals, dim=0),
        "s_c_star": torch.cat(s_c_star_vals, dim=0),
        "p_neg_max": torch.cat(p_neg_max_vals, dim=0),
        "p_mix": torch.cat(p_mix_vals, dim=0),
    }
    return out


def summarize_tensor(tensor):
    tensor = tensor.float()
    return {
        "mean": float(tensor.mean().item()),
        "median": float(tensor.median().item()),
        "p90": float(torch.quantile(tensor, 0.9).item()),
    }


def build_stats_records(epoch, split, metrics):
    s_c_star = metrics["s_c_star"]
    total = s_c_star.numel()
    threshold = torch.quantile(s_c_star, 0.95)
    hard_mask = s_c_star >= threshold

    records = []
    for group, mask in (("all", None), ("hard5", hard_mask)):
        record = {
            "epoch": int(epoch),
            "split": split,
            "group": group,
        }
        if mask is None:
            count = total
        else:
            count = int(mask.sum().item())
        record["count"] = count
        for name, tensor in metrics.items():
            values = tensor if mask is None else tensor[mask]
            stats = summarize_tensor(values)
            record[f"{name}_mean"] = stats["mean"]
            record[f"{name}_median"] = stats["median"]
            record[f"{name}_p90"] = stats["p90"]
        records.append(record)
    return records


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def train_and_audit(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    train_ds, test_ds, test_tfm = build_datasets(args.data_root)
    train_ld, test_ld = build_loaders(
        train_ds, test_ds, args.batch_size, args.test_batch_size, args.num_workers
    )
    probe_train_ld, probe_test_ld = build_probe_loaders(
        args.data_root,
        args.probe_size,
        args.probe_seed,
        args.probe_batch_size,
        args.num_workers,
        test_tfm,
    )

    model = DeepLDA(num_classes=args.num_classes, dim=args.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.loss == "dnll":
        loss_fn = DNLLLoss(lambda_reg=args.lambda_reg)
        loss_name = "dnll"
    else:
        loss_fn = LogisticLoss()
        loss_name = "logistic"

    output_path = args.out
    if output_path is None:
        output_path = f"results/audit_cifar100_{loss_name}_seed{args.seed}.jsonl"
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8"):
        pass

    for epoch in range(1, args.epochs + 1):
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

        train_acc = acc_sum / n_sum
        test_acc = evaluate(model, test_ld, device)
        print(
            f"[{loss_name} {epoch:03d}] train loss={loss_sum/n_sum:.4f} "
            f"acc={train_acc:.4f} | test acc={test_acc:.4f}"
        )

        if epoch % args.audit_every == 0:
            train_metrics = collect_audit_tensors(
                model, probe_train_ld, device, loss_name
            )
            test_metrics = collect_audit_tensors(
                model, probe_test_ld, device, loss_name
            )
            for record in build_stats_records(epoch, "train", train_metrics):
                append_jsonl(output_path, record)
            for record in build_stats_records(epoch, "test", test_metrics):
                append_jsonl(output_path, record)

    final_record = {
        "epoch": int(args.epochs),
        "metric": "final_accuracy",
        "train_acc": float(train_acc),
        "test_acc": float(test_acc),
    }
    append_jsonl(output_path, final_record)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit hard-negative pressure during Deep LDA training on CIFAR-100."
    )
    parser.add_argument("--loss", choices=["dnll", "logistic"], default="dnll")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--audit-every", type=int, default=5)
    parser.add_argument("--probe-size", type=int, default=10000)
    parser.add_argument("--probe-seed", type=int, default=1234)
    parser.add_argument("--out", default=None)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-batch-size", type=int, default=1024)
    parser.add_argument("--probe-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--dim", type=int, default=99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    train_and_audit(args)


if __name__ == "__main__":
    main()
