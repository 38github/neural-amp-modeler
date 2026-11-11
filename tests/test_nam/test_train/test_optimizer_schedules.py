import pytest
import torch

from nam.train.lightning_module import LightningModule, _cyclic_weight_decay


def test_cyclic_weight_decay_reaches_minimum_midpoint_and_end():
    minimum = 1e-8
    maximum = 3.17e-7
    period = 100

    assert _cyclic_weight_decay(0, minimum, maximum, period) == pytest.approx(minimum)
    assert _cyclic_weight_decay(25, minimum, maximum, period) == pytest.approx(
        (minimum + maximum) / 2
    )
    assert _cyclic_weight_decay(50, minimum, maximum, period) == pytest.approx(maximum)
    assert _cyclic_weight_decay(75, minimum, maximum, period) == pytest.approx(
        (minimum + maximum) / 2
    )
    assert _cyclic_weight_decay(100, minimum, maximum, period) == pytest.approx(minimum)


def test_cyclic_weight_decay_requires_positive_period():
    with pytest.raises(ValueError, match="period_steps must be > 0"):
        _cyclic_weight_decay(0, 1e-8, 1e-7, 0)


def test_configure_optimizers_uses_adamw_and_cyclic_lr():
    module = LightningModule(
        net=torch.nn.Linear(1, 1),
        optimizer_config={"lr": 0.0002, "weight_decay": 1e-8},
        scheduler_config={
            "class": "CyclicLR",
            "kwargs": {
                "base_lr": 0.0002,
                "max_lr": 0.004,
                "step_size_up": 10,
                "cycle_momentum": False,
            },
            "interval": "step",
        },
    )

    configured = module.configure_optimizers()

    assert isinstance(configured["optimizer"], torch.optim.AdamW)
    assert isinstance(
        configured["lr_scheduler"]["scheduler"],
        torch.optim.lr_scheduler.CyclicLR,
    )
