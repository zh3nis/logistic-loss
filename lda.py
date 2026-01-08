import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LDAHead(nn.Module):
    """LDA classifier with trainable means, shared covariance, and trainable priors."""

    def __init__(self, C, D, covariance_type="spherical", min_scale=1e-4):
        super().__init__()
        if D < 1:
            raise ValueError(f"D must be positive (got D={D}).")
        self.C = C
        self.D = D
        cov_type = str(covariance_type).lower()
        if cov_type not in {"spherical", "diag", "full"}:
            raise ValueError(
                "covariance_type must be one of {'spherical', 'diag', 'full'} "
                f"(got {covariance_type!r})."
            )
        self.covariance_type = cov_type
        self.min_scale = min_scale
        dtype = torch.get_default_dtype()
        # Start class means from a normal distribution instead of a fixed simplex layout.
        self.mu = nn.Parameter(torch.randn(C, D, dtype=dtype) * 6.0 / math.sqrt(2*D))
        self.prior_logits = nn.Parameter(torch.zeros(C, dtype=dtype))
        if self.covariance_type == "spherical":
            self.log_cov = nn.Parameter(torch.zeros(1, dtype=dtype))
        elif self.covariance_type == "diag":
            self.log_cov_diag = nn.Parameter(torch.zeros(D, dtype=dtype))
        else:
            self.raw_tril = nn.Parameter(torch.zeros(D, D, dtype=dtype))

    def _get_cholesky(self, dtype, device):
        raw = torch.tril(self.raw_tril.to(device=device, dtype=dtype))
        diag = torch.diagonal(raw, 0)
        safe_diag = F.softplus(diag) + self.min_scale
        L = raw - torch.diag(diag) + torch.diag(safe_diag)
        return L

    @property
    def cov_diag(self):
        if self.covariance_type == "spherical":
            return torch.exp(self.log_cov).repeat(self.D)
        if self.covariance_type == "diag":
            return torch.exp(self.log_cov_diag)
        return torch.diagonal(self.covariance)

    @property
    def covariance(self):
        """Full covariance matrix Sigma = L L^T."""
        if self.covariance_type != "full":
            raise AttributeError("covariance is only defined for full covariance.")
        L = self._get_cholesky(self.raw_tril.dtype, self.raw_tril.device)
        return L @ L.transpose(-2, -1)

    def forward(self, z):
        if self.covariance_type == "full":
            dtype = z.dtype
            device = z.device
            mu = self.mu.to(device=device, dtype=dtype)
            diff = z.unsqueeze(1) - mu.unsqueeze(0)
            L = self._get_cholesky(dtype, device)
            diff_flat = diff.reshape(-1, self.D).transpose(0, 1)
            solved = torch.linalg.solve_triangular(L, diff_flat, upper=False)
            m2 = (solved * solved).sum(dim=0).reshape(z.shape[0], self.C)
            log_det = 2.0 * torch.log(torch.diagonal(L)).sum()
            log_prior = torch.log_softmax(
                self.prior_logits.to(device=device, dtype=dtype), dim=0
            )
            return log_prior.unsqueeze(0) - 0.5 * (m2 + log_det)

        mu = self.mu.to(z.dtype)
        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        if self.covariance_type == "spherical":
            m2 = (diff * diff).sum(-1)
            log_cov = self.log_cov.to(z.dtype)
            var = torch.exp(log_cov)
            log_det = self.D * log_cov
            log_prior = torch.log_softmax(self.prior_logits, dim=0)
            return log_prior.unsqueeze(0) - 0.5 * (m2 / var + log_det)

        log_cov_diag = self.log_cov_diag.to(z.dtype)
        var = torch.exp(log_cov_diag)
        m2 = (diff * diff / var).sum(-1)
        log_det = log_cov_diag.sum()
        log_prior = torch.log_softmax(self.prior_logits, dim=0)
        return log_prior.unsqueeze(0) - 0.5 * (m2 + log_det)


def dnll_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    lambda_reg: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""
    DNLL: Discriminative Negative Log-Likelihood

        L(x, y) = -input_y(x) + λ * sum_c exp(input_c(x))

    Applicable to any generative classifier with class-wise
    (unnormalized) log-density or log-joint scores.

    Args:
        input:  Tensor (N, C) of class scores δ_c(x).
        target: LongTensor (N,) with class indices in [0, C-1].
        lambda_reg: float ≥ 0, strength of discriminative penalty.
        reduction: "none" | "mean" | "sum".

    Returns:
        Loss reduced according to `reduction`.
    """
    # NLL part: -δ_y(x)
    nll = -input.gather(1, target.unsqueeze(1)).squeeze(1)  # (N,)

    # Discriminative penalty: λ * ∑_c exp(δ_c(x))
    reg = lambda_reg * input.exp().sum(dim=1)               # (N,)

    loss = nll + reg                                        # (N,)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

class DNLLLoss(nn.Module):
    r"""
    DNLL: Discriminative Negative Log-Likelihood

        L(x, y) = -input_y(x) + λ * sum_c exp(input_c(x))

    A drop-in loss module similar to nn.CrossEntropyLoss, but designed
    for generative classifiers whose outputs are log-density scores.
    """
    def __init__(self, lambda_reg: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.lambda_reg = float(lambda_reg)
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dnll_loss(
            input=input,
            target=target,
            lambda_reg=self.lambda_reg,
            reduction=self.reduction,
        )


def logistic_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""
    One-vs-all logistic loss over all classes.

        L(x, y) = -log σ(δ_y(x)) - ∑_{c≠y} log σ(-δ_c(x))

    Equivalent to summing binary cross-entropy terms for each class with a
    one-hot target.

    Args:
        input:  Tensor (N, C) of class scores δ_c(x).
        target: LongTensor (N,) with class indices in [0, C-1].
        reduction: "none" | "mean" | "sum".

    Returns:
        Loss reduced according to `reduction`.
    """
    one_hot = F.one_hot(target, num_classes=input.shape[1]).type_as(input)  # (N, C)
    per_class = F.binary_cross_entropy_with_logits(input, one_hot, reduction="none")  # (N, C)
    loss = per_class.sum(dim=1)  # (N,)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


class LogisticLoss(nn.Module):
    r"""
    One-vs-all logistic loss over all classes.

        L(x, y) = -log σ(δ_y(x)) - ∑_{c≠y} log σ(-δ_c(x))
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return logistic_loss(
            input=input,
            target=target,
            reduction=self.reduction,
        )
