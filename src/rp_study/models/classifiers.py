"""Trainable classifier architectures for supervised experiments."""

from dataclasses import asdict
from typing import List, Tuple

import torch
import torch.nn as nn

from typing import Optional

from ..config import ClassifierConfig
from .initializers import initialize_layer


class _GradRescale(torch.autograd.Function):
    """Identity in the forward pass; multiply the gradient by `r` in backward.

    Inserted after each hidden ReLU, this scales ONLY the backward pass (the
    forward is untouched). Used to cancel the row-centered forward-balanced
    per-layer backward gain g_bwd = 1/r with a per-layer factor r, so the
    cumulative backward amplification (1/r)^depth becomes ~1. Equivalent to a
    geometric per-layer learning rate r^(L-l).
    """

    @staticmethod
    def forward(ctx, x, r):
        ctx.r = r
        return x

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out * ctx.r, None


class DeepFCClassifier(nn.Module):
    """Plain deep fully connected classifier with optional BatchNorm."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        num_classes: int,
        init_strategy: str = "he",
        use_batch_norm: bool = False,
        use_bias: bool = True,
        bn_momentum: float = 0.1,
        bn_eps: float = 1e-5,
        grad_rescale: Optional[float] = None,
        trainable_layers: Optional[List[str]] = None,
        **init_kwargs,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm
        self.grad_rescale = grad_rescale

        self.hidden_layers = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()

        specs: List[Tuple[int, int]] = [(input_dim, hidden_dim)]
        specs.extend((hidden_dim, hidden_dim) for _ in range(depth - 1))
        specs.append((hidden_dim, num_classes))
        n_weight_layers = len(specs)

        for layer_index, (fan_in, fan_out) in enumerate(specs[:-1]):
            linear = nn.Linear(fan_in, fan_out, bias=use_bias and not use_batch_norm)
            initialize_layer(
                linear,
                strategy=init_strategy,
                layer_index=layer_index,
                n_layers=n_weight_layers,
                **init_kwargs,
            )
            self.hidden_layers.append(linear)
            if use_batch_norm:
                self.hidden_norms.append(nn.BatchNorm1d(fan_out, eps=bn_eps, momentum=bn_momentum))

        self.classifier = nn.Linear(hidden_dim, num_classes, bias=use_bias)
        initialize_layer(
            self.classifier,
            strategy=init_strategy,
            layer_index=n_weight_layers - 1,
            n_layers=n_weight_layers,
            **init_kwargs,
        )

        if trainable_layers is not None:
            self._freeze_except(trainable_layers)

    def _freeze_except(self, trainable_layers: List[str]) -> None:
        """Set requires_grad=False on every Linear param NOT in trainable_layers.

        Only touches leaf parameters (weight/bias) -- forward/backward still
        run through frozen layers unchanged, so gradients keep flowing to
        earlier trainable layers via the input-grad path.
        """
        trainable_set = set(trainable_layers)
        for layer_index, linear in enumerate(self.hidden_layers):
            requires_grad = f"fc{layer_index + 1}" in trainable_set
            for param in linear.parameters():
                param.requires_grad = requires_grad
        head_requires_grad = "head" in trainable_set
        for param in self.classifier.parameters():
            param.requires_grad = head_requires_grad

    def trainable_tensor_count(self) -> int:
        return sum(1 for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = x.view(x.shape[0], -1)

        for idx, linear in enumerate(self.hidden_layers):
            x = linear(x)
            if self.use_batch_norm:
                x = self.hidden_norms[idx](x)
            x = torch.relu(x)
            if self.grad_rescale is not None:
                x = _GradRescale.apply(x, self.grad_rescale)
        return self.classifier(x)


class DeepCNNClassifier(nn.Module):
    """Plain deep convolutional classifier with optional BatchNorm."""

    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        max_channels: int,
        depth: int,
        num_classes: int,
        init_strategy: str = "he",
        use_batch_norm: bool = False,
        use_bias: bool = True,
        bn_momentum: float = 0.1,
        bn_eps: float = 1e-5,
        **init_kwargs,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        self.input_channels = input_channels
        self.base_channels = base_channels
        self.max_channels = max_channels
        self.depth = depth
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm

        stage_channels = [
            min(max_channels, base_channels),
            min(max_channels, base_channels * 2),
            min(max_channels, base_channels * 4),
            min(max_channels, base_channels * 8),
            max_channels,
        ]
        stage_lengths = self._split_depth(depth, len(stage_channels))

        specs: List[Tuple[int, int]] = []
        in_channels = input_channels
        for stage_idx, stage_len in enumerate(stage_lengths):
            out_channels = stage_channels[stage_idx]
            for layer_idx_in_stage in range(stage_len):
                specs.append((in_channels if layer_idx_in_stage == 0 else out_channels, out_channels))
            in_channels = out_channels

        self.conv_layers = nn.ModuleList()
        self.conv_norms = nn.ModuleList()
        self.stage_pool_after: List[int] = []
        n_weight_layers = len(specs) + 1  # +1 for final classifier

        spec_idx = 0
        for stage_idx, stage_len in enumerate(stage_lengths):
            out_channels = stage_channels[stage_idx]
            for layer_idx_in_stage in range(stage_len):
                fan_in, fan_out = specs[spec_idx]
                conv = nn.Conv2d(
                    fan_in,
                    fan_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias and not use_batch_norm,
                )
                initialize_layer(
                    conv,
                    strategy=init_strategy,
                    layer_index=spec_idx,
                    n_layers=n_weight_layers,
                    **init_kwargs,
                )
                self.conv_layers.append(conv)
                if use_batch_norm:
                    self.conv_norms.append(
                        nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum)
                    )
                spec_idx += 1

            if stage_idx < len(stage_lengths) - 1:
                self.stage_pool_after.append(spec_idx - 1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(stage_channels[-1], num_classes, bias=use_bias)
        initialize_layer(
            self.classifier,
            strategy=init_strategy,
            layer_index=n_weight_layers - 1,
            n_layers=n_weight_layers,
            **init_kwargs,
        )

    @staticmethod
    def _split_depth(depth: int, n_stages: int) -> List[int]:
        """Split the requested conv depth across stages as evenly as possible."""
        base = depth // n_stages
        remainder = depth % n_stages
        lengths = [base] * n_stages
        for idx in range(remainder):
            lengths[idx] += 1
        # Ensure every stage has at least one conv; borrow from earlier stages if needed.
        for idx in range(n_stages):
            if lengths[idx] == 0:
                donor = next((j for j, value in enumerate(lengths) if value > 1), None)
                if donor is None:
                    raise ValueError(f"depth={depth} is too small for {n_stages} CNN stages")
                lengths[donor] -= 1
                lengths[idx] += 1
        return lengths

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_idx = 0
        for conv_idx, conv in enumerate(self.conv_layers):
            x = conv(x)
            if self.use_batch_norm:
                x = self.conv_norms[norm_idx](x)
                norm_idx += 1
            x = torch.relu(x)
            if conv_idx in self.stage_pool_after:
                x = self.pool(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_classifier(config: ClassifierConfig) -> nn.Module:
    """Instantiate a classifier from a ClassifierConfig."""
    kwargs = dict(config.init_kwargs)
    if config.architecture == "fc":
        return DeepFCClassifier(
            input_dim=config.fc_input_dim,
            hidden_dim=config.fc_hidden_dim,
            depth=config.depth,
            num_classes=config.num_classes,
            init_strategy=config.init_strategy,
            use_batch_norm=config.use_batch_norm,
            use_bias=config.use_bias,
            bn_momentum=config.bn_momentum,
            bn_eps=config.bn_eps,
            grad_rescale=config.grad_rescale,
            trainable_layers=config.trainable_layers,
            **kwargs,
        )
    if config.architecture == "cnn":
        return DeepCNNClassifier(
            input_channels=config.cnn_input_channels,
            base_channels=config.cnn_base_channels,
            max_channels=config.cnn_max_channels,
            depth=config.depth,
            num_classes=config.num_classes,
            init_strategy=config.init_strategy,
            use_batch_norm=config.use_batch_norm,
            use_bias=config.use_bias,
            bn_momentum=config.bn_momentum,
            bn_eps=config.bn_eps,
            **kwargs,
        )
    raise ValueError(f"Unknown architecture: {config.architecture}")


def classifier_config_to_dict(config: ClassifierConfig) -> dict:
    """Serialize a classifier config for experiment logging."""
    return asdict(config)
