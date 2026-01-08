"""Run a DNLL lambda_reg sweep on CIFAR-10, CIFAR-100, or FashionMNIST."""
import argparse
import json
import pathlib
import random
import statistics as stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from lda import LDAHead, DNLLLoss


DATASETS = {
    "cifar10": {
        "dataset_cls": datasets.CIFAR10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "image_size": 32,
        "channels": 3,
        "num_classes": 10,
    },
    "cifar100": {
        "dataset_cls": datasets.CIFAR100,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "image_size": 32,
        "channels": 3,
        "num_classes": 100,
    },
    "fashionmnist": {
        "dataset_cls": datasets.FashionMNIST,
        "mean": (0.2860,),
        "std": (0.3530,),
        "image_size": 28,
        "channels": 1,
        "num_classes": 10,
    },
}


def build_loaders(
    dataset_name: str,
    data_root: pathlib.Path,
    batch_size: int,
    test_batch_size: int,
    num_workers: int,
):
    cfg = DATASETS[dataset_name]
    pin_memory = torch.cuda.is_available()
    image_size = cfg["image_size"]

    train_ops = [transforms.RandomCrop(image_size, padding=4)]
    if cfg["channels"] == 3:
        train_ops.append(transforms.RandomHorizontalFlip())
    train_ops.extend([transforms.ToTensor(), transforms.Normalize(cfg["mean"], cfg["std"])])
    train_tfm = transforms.Compose(train_ops)
    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])

    train_ds = cfg["dataset_cls"](root=data_root, train=True, transform=train_tfm, download=True)
    test_ds = cfg["dataset_cls"](root=data_root, train=False, transform=test_tfm, download=True)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_ld = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_ld, test_ld


class Encoder(nn.Module):
    def __init__(self, in_channels: int, dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, dim)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.proj(x)


class DeepLDA(nn.Module):
    def __init__(self, C: int, D: int, in_channels: int):
        super().__init__()
        self.encoder = Encoder(in_channels, D)
        self.head = LDAHead(C, D)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


@torch.no_grad()
def compute_ece(confidence: torch.Tensor, correct: torch.Tensor, n_bins: int = 10) -> float:
    bins = torch.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = bins[1:-1]
    bin_ids = torch.bucketize(confidence, bin_edges, right=True)
    ece = torch.zeros(1)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            bin_acc = correct[mask].mean()
            bin_conf = confidence[mask].mean()
            bin_frac = mask.float().mean()
            ece += torch.abs(bin_acc - bin_conf) * bin_frac
    return ece.item()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, n_bins: int = 10):
    model.eval()
    ok = tot = 0
    confidences = []
    corrects = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        correct = (pred == y).float()
        ok += correct.sum().item()
        tot += y.size(0)
        confidences.append(conf.detach().cpu())
        corrects.append(correct.detach().cpu())
    acc = ok / tot
    confidence = torch.cat(confidences)
    correct = torch.cat(corrects)
    ece = compute_ece(confidence, correct, n_bins=n_bins)
    return acc, ece


def train_single(model, loss_fn, opt, train_ld, test_ld, device, epochs: int, n_bins: int = 15):
    train_acc = []
    test_acc = []
    test_ece = []
    sigma = []

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
        te_acc, te_ece = evaluate(model, test_ld, device, n_bins=n_bins)
        train_acc.append(tr_acc)
        test_acc.append(te_acc)
        test_ece.append(te_ece)
        sigma.append(read_sigma(model.head))
        print(
            f"[{epoch:02d}] train loss={loss_sum/n_sum:.4f} acc={tr_acc:.4f} | "
            f"test acc={te_acc:.4f} ece={te_ece:.4f}"
        )

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "test_ece": test_ece,
        "sigma": sigma,
        "final_test": test_acc[-1],
        "final_test_ece": test_ece[-1],
        "final_sigma": sigma[-1],
    }


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def read_sigma(head: LDAHead) -> float:
    if head.covariance_type == "spherical":
        sigma = torch.exp(0.5 * head.log_cov).item()
    elif head.covariance_type == "diag":
        sigma = torch.exp(0.5 * head.log_cov_diag).mean().item()
    else:
        sigma = torch.sqrt(torch.diagonal(head.covariance)).mean().item()
    return sigma


def run_lambda_sweep(lambda_regs, runs_per_lambda, epochs, train_ld, test_ld, device, num_classes, in_channels):
    results = {}
    rng = random.SystemRandom()
    for lambda_reg in lambda_regs:
        print(f"=== DNLL lambda_reg={lambda_reg} ===")
        runs = []
        for run_idx in range(runs_per_lambda):
            run_seed = rng.randrange(0, 2**31 - 1)
            seed_everything(run_seed)
            print(f"-- run {run_idx + 1}/{runs_per_lambda} (seed={run_seed})")

            model = DeepLDA(C=num_classes, D=num_classes - 1, in_channels=in_channels).to(device)
            opt = torch.optim.Adam(model.parameters())
            loss_fn = DNLLLoss(lambda_reg=lambda_reg)
            run_result = train_single(model, loss_fn, opt, train_ld, test_ld, device, epochs)
            run_result["seed"] = run_seed
            runs.append(run_result)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        finals = [run["final_test"] for run in runs]
        finals_ece = [run["final_test_ece"] for run in runs]
        finals_sigma = [run["final_sigma"] for run in runs]
        results[lambda_reg] = {
            "runs": runs,
            "final_test_mean": stats.mean(finals),
            "final_test_std": stats.pstdev(finals) if len(finals) > 1 else 0.0,
            "final_test_ece_mean": stats.mean(finals_ece),
            "final_test_ece_std": stats.pstdev(finals_ece) if len(finals_ece) > 1 else 0.0,
            "final_sigma_mean": stats.mean(finals_sigma),
            "final_sigma_std": stats.pstdev(finals_sigma) if len(finals_sigma) > 1 else 0.0,
        }
    return results


def save_results_json(path: pathlib.Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"Saved metrics to {path}")


def main():
    parser = argparse.ArgumentParser(description="DNLL lambda_reg sweep (DeepLDA)")
    parser.add_argument("--dataset", choices=sorted(DATASETS.keys()), default="cifar10",
                        help="dataset to use")
    parser.add_argument("--lambda-regs", type=float, nargs="+",
                        default=[0.001, 0.00316227766, 0.01, 0.0316227766, 0.1,
                                 0.316227766, 1.0, 3.16227766, 10.0],
                        help="lambda_reg values to try (space-separated)")
    parser.add_argument("--runs", type=int, default=5, help="runs per lambda_reg")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs per run")
    parser.add_argument("--data-root", type=pathlib.Path, default=pathlib.Path("./data"), help="dataset root directory")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("plots"), help="where to save JSON and plot")
    parser.add_argument("--batch-size", type=int, default=256, help="train batch size")
    parser.add_argument("--test-batch-size", type=int, default=1024, help="test batch size")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    cfg = DATASETS[args.dataset]
    train_ld, test_ld = build_loaders(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.workers,
    )
    results = run_lambda_sweep(
        lambda_regs=args.lambda_regs,
        runs_per_lambda=args.runs,
        epochs=args.epochs,
        train_ld=train_ld,
        test_ld=test_ld,
        device=device,
        num_classes=cfg["num_classes"],
        in_channels=cfg["channels"],
    )

    output_dir = args.output_dir
    metrics_path = output_dir / f"{args.dataset}_dnll_lambda_reg_sweep.json"
    payload = {
        "dataset": args.dataset,
        "lambda_regs": args.lambda_regs,
        "epochs": args.epochs,
        "runs_per_lambda": args.runs,
        "device": str(device),
        "results": results,
    }
    save_results_json(metrics_path, payload)

    print("Final test accuracy (mean ± std):")
    for lambda_reg in args.lambda_regs:
        info = results[lambda_reg]
        print(f"lambda_reg={lambda_reg:g}: {info['final_test_mean']:.4f} ± {info['final_test_std']:.4f}")


if __name__ == "__main__":
    main()
