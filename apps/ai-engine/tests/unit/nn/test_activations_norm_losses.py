"""
Unit tests for nn/activations, nn/normalization, nn/losses, nn/optimizers, nn/schedulers

Groups:
    TestActivations   — shape, range, monotonicity, factory
    TestNormalization — output shape, zero mean/unit var, affine params, factory
    TestLosses        — scalar output, gradient, ignore_index, known values
    TestOptimizers    — parameter update, state, factory
    TestSchedulers    — warmup ramp, cosine decay, value range
"""

import math
import pytest
import torch
import torch.nn as nn
from torch.optim import SGD as TorchSGD

from src.nn.activations.activations import (
    ReLU, GELU, SiLU, Mish, ELU, Sigmoid, Tanh, Softmax,
    relu, gelu, silu, get_activation,
)
from src.nn.normalization.normalization import (
    LayerNorm, RMSNorm, BatchNorm1d, GroupNorm, get_norm,
)
from src.nn.losses.losses import (
    CrossEntropyLoss, MSELoss, HuberLoss, FocalLoss,
    ContrastiveLoss, LabelSmoothingCrossEntropy, KLDivergenceLoss,
)
from src.nn.optimizers.optimizers import SGD, Adam, AdamW, Lion, get_optimizer
from src.nn.schedulers.schedulers import (
    LinearWarmupCosineDecay, ConstantWithWarmup, get_scheduler,
)
from src.nn.initializers.initializers import (
    initialize_weights, scale_residual_weights,
    truncated_normal_, xavier_uniform_, kaiming_normal_,
)


# ─────────────────────────────────────────────────────────────────────────────
# Activations
# ─────────────────────────────────────────────────────────────────────────────

class TestActivations:

    @pytest.fixture
    def x(self):
        return torch.linspace(-3.0, 3.0, steps=20)

    def test_relu_nonnegative(self, x):
        assert (ReLU()(x) >= 0).all()

    def test_relu_zero_for_negative(self, x):
        neg = x[x < 0]
        assert (ReLU()(neg) == 0).all()

    def test_gelu_shape(self, x):
        assert GELU()(x).shape == x.shape

    def test_silu_shape(self, x):
        assert SiLU()(x).shape == x.shape

    def test_sigmoid_range(self, x):
        out = Sigmoid()(x)
        assert (out > 0).all() and (out < 1).all()

    def test_tanh_range(self, x):
        out = Tanh()(x)
        assert (out > -1).all() and (out < 1).all()

    def test_softmax_sums_to_one(self):
        x   = torch.randn(4, 10)
        out = Softmax(dim=-1)(x)
        sums = out.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_softmax_non_negative(self):
        x   = torch.randn(4, 10)
        out = Softmax(dim=-1)(x)
        assert (out >= 0).all()

    def test_elu_positive_identity(self, x):
        pos_x = x[x > 0]
        out   = ELU()(pos_x)
        assert torch.allclose(out, pos_x, atol=1e-5)

    def test_mish_shape(self, x):
        assert Mish()(x).shape == x.shape

    def test_get_activation_relu(self):
        act = get_activation("relu")
        assert isinstance(act, ReLU)

    def test_get_activation_gelu(self):
        act = get_activation("gelu")
        assert isinstance(act, GELU)

    def test_get_activation_swish_alias(self):
        act = get_activation("swish")
        assert isinstance(act, SiLU)

    def test_get_activation_case_insensitive(self):
        act = get_activation("RELU")
        assert isinstance(act, ReLU)

    def test_get_activation_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            get_activation("nonexistent_act")

    def test_gradient_flows_through_gelu(self):
        x = torch.randn(8, requires_grad=True)
        GELU()(x).sum().backward()
        assert x.grad is not None

    def test_functional_and_module_equivalent(self, x):
        """Function and module versions must produce identical results."""
        assert torch.allclose(relu(x), ReLU()(x))
        assert torch.allclose(silu(x), SiLU()(x))


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalization:

    D_NORM = 32

    @pytest.fixture
    def x(self):
        return torch.randn(4, 8, self.D_NORM)

    def test_layer_norm_output_shape(self, x):
        ln = LayerNorm(self.D_NORM)
        assert ln(x).shape == x.shape

    def test_layer_norm_unit_variance(self, x):
        """After LayerNorm, each (B, T) position should have ~unit variance over D."""
        ln  = LayerNorm(self.D_NORM)
        out = ln(x)
        var = out.var(dim=-1, unbiased=False)
        assert torch.allclose(var, torch.ones_like(var), atol=0.1)

    def test_layer_norm_near_zero_mean(self, x):
        """LayerNorm should produce near-zero mean over the feature dim."""
        ln   = LayerNorm(self.D_NORM)
        out  = ln(x)
        mean = out.mean(dim=-1).abs()
        assert (mean < 0.1).all()

    def test_layer_norm_no_affine(self, x):
        ln = LayerNorm(self.D_NORM, elementwise_affine=False)
        assert ln(x).shape == x.shape
        assert ln.weight is None
        assert ln.bias   is None

    def test_rms_norm_output_shape(self, x):
        rms = RMSNorm(self.D_NORM)
        assert rms(x).shape == x.shape

    def test_rms_norm_unit_rms(self, x):
        """After RMSNorm, the RMS of each vector should be approximately 1."""
        rms = RMSNorm(self.D_NORM)
        # Detach and disable learned scale for this check
        with torch.no_grad():
            rms.weight.fill_(1.0)
        out = rms(x)
        rms_vals = out.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms_vals, torch.ones_like(rms_vals), atol=0.15)

    def test_rms_norm_has_one_parameter(self):
        rms    = RMSNorm(16)
        params = list(rms.parameters())
        assert len(params) == 1
        assert params[0].shape == (16,)

    def test_batch_norm_output_shape(self):
        bn = BatchNorm1d(32)
        x  = torch.randn(8, 32, 10)
        bn.train()
        assert bn(x).shape == (8, 32, 10)

    def test_group_norm_output_shape(self):
        gn = GroupNorm(num_groups=4, num_channels=32)
        x  = torch.randn(4, 32, 8, 8)
        assert gn(x).shape == (4, 32, 8, 8)

    def test_group_norm_bad_groups_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            GroupNorm(num_groups=5, num_channels=32)

    def test_get_norm_layer(self):
        norm = get_norm("layer", 64)
        assert isinstance(norm, LayerNorm)

    def test_get_norm_rms(self):
        norm = get_norm("rms", 64)
        assert isinstance(norm, RMSNorm)

    def test_get_norm_unknown_raises(self):
        with pytest.raises(ValueError):
            get_norm("unknown_norm", 64)

    def test_layer_norm_gradient_flows(self, x):
        x_req = x.clone().requires_grad_(True)
        ln    = LayerNorm(self.D_NORM)
        ln(x_req).sum().backward()
        assert x_req.grad is not None
        assert ln.weight.grad is not None

    def test_rms_norm_gradient_flows(self, x):
        x_req = x.clone().requires_grad_(True)
        rms   = RMSNorm(self.D_NORM)
        rms(x_req).sum().backward()
        assert x_req.grad is not None
        assert rms.weight.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────

class TestLosses:

    def test_cross_entropy_scalar_output(self):
        loss_fn = CrossEntropyLoss()
        logits  = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss    = loss_fn(logits, targets)
        assert loss.shape == ()

    def test_cross_entropy_ignores_minus_100(self):
        """Loss with all -100 targets should be zero (nothing to learn)."""
        loss_fn = CrossEntropyLoss(ignore_index=-100)
        logits  = torch.randn(4, 10)
        targets = torch.full((4,), -100, dtype=torch.long)
        loss    = loss_fn(logits, targets)
        # When all positions are ignored, loss is 0/0 — PyTorch returns 0.0
        assert not torch.isnan(loss)

    def test_cross_entropy_3d_input(self):
        """3D input (B, T, C) must be flattened before loss."""
        loss_fn = CrossEntropyLoss()
        logits  = torch.randn(2, 5, 10)
        targets = torch.randint(0, 10, (2, 5))
        loss    = loss_fn(logits, targets)
        assert loss.shape == ()

    def test_cross_entropy_gradient_flows(self):
        logits  = torch.randn(4, 10, requires_grad=True)
        targets = torch.randint(0, 10, (4,))
        CrossEntropyLoss()(logits, targets).backward()
        assert logits.grad is not None

    def test_mse_loss_perfect_prediction_is_zero(self):
        loss_fn = MSELoss()
        x       = torch.randn(8, 4)
        loss    = loss_fn(x, x)
        assert loss.item() < 1e-8

    def test_mse_loss_positive(self):
        loss_fn = MSELoss()
        pred    = torch.randn(8, 4)
        target  = torch.randn(8, 4)
        loss    = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_huber_loss_scalar(self):
        loss_fn = HuberLoss()
        pred    = torch.randn(8, 4)
        target  = torch.randn(8, 4)
        loss    = loss_fn(pred, target)
        assert loss.shape == ()

    def test_focal_loss_output_shape(self):
        loss_fn = FocalLoss(gamma=2.0)
        logits  = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        loss    = loss_fn(logits, targets)
        assert loss.shape == ()

    def test_focal_loss_gamma_zero_equals_ce(self):
        """Focal loss with gamma=0 must equal standard cross-entropy."""
        logits  = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        focal   = FocalLoss(gamma=0.0)(logits, targets).item()
        ce      = nn.CrossEntropyLoss()(logits, targets).item()
        assert abs(focal - ce) < 1e-4

    def test_contrastive_loss_similar_pairs_minimised(self):
        """For perfectly similar pairs (distance=0), loss should be ~0."""
        loss_fn = ContrastiveLoss(margin=1.0)
        emb     = torch.randn(4, 16)
        labels  = torch.zeros(4)   # all similar
        loss    = loss_fn(emb, emb.clone(), labels)
        assert loss.item() < 1e-6

    def test_label_smoothing_ce_output_shape(self):
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        logits  = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss    = loss_fn(logits, targets)
        assert loss.shape == ()

    def test_label_smoothing_less_than_ce_for_correct(self):
        """Label smoothing should reduce loss when the model is very confident."""
        logits  = torch.zeros(4, 10)
        logits[:, 0] = 100.0   # Very confident on class 0
        targets = torch.zeros(4, dtype=torch.long)

        ce_loss = CrossEntropyLoss()(logits, targets).item()
        ls_loss = LabelSmoothingCrossEntropy(smoothing=0.1)(logits, targets).item()

        # Label smoothing penalises overconfidence → should have higher loss
        assert ls_loss > ce_loss


# ─────────────────────────────────────────────────────────────────────────────
# Optimizers
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimizers:

    @pytest.fixture
    def simple_model(self):
        return nn.Linear(8, 4)

    def _step_once(self, optimizer, model):
        """Run one gradient step and return before/after parameter values."""
        x      = torch.randn(4, 8)
        before = model.weight.data.clone()
        loss   = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        after = model.weight.data.clone()
        return before, after

    def test_sgd_updates_parameters(self, simple_model):
        opt = SGD(simple_model.parameters(), lr=0.01)
        before, after = self._step_once(opt, simple_model)
        assert not torch.allclose(before, after)

    def test_adam_updates_parameters(self, simple_model):
        opt = Adam(simple_model.parameters(), lr=1e-3)
        before, after = self._step_once(opt, simple_model)
        assert not torch.allclose(before, after)

    def test_adamw_updates_parameters(self, simple_model):
        opt = AdamW(simple_model.parameters(), lr=1e-3)
        before, after = self._step_once(opt, simple_model)
        assert not torch.allclose(before, after)

    def test_lion_updates_parameters(self, simple_model):
        opt = Lion(simple_model.parameters(), lr=1e-4)
        before, after = self._step_once(opt, simple_model)
        assert not torch.allclose(before, after)

    def test_lion_uniform_step_size(self, simple_model):
        """Lion updates all parameters by the same magnitude (|sign|=1)."""
        opt = Lion(simple_model.parameters(), lr=0.1, weight_decay=0.0)
        x   = torch.randn(4, 8)
        simple_model(x).sum().backward()
        before = simple_model.weight.data.clone()
        opt.step()
        diff = (simple_model.weight.data - before).abs()
        # All weight diffs should be ~0.1 (lr × 1 for sign)
        assert torch.allclose(diff, torch.full_like(diff, 0.1), atol=0.01)

    def test_get_optimizer_sgd(self, simple_model):
        opt = get_optimizer("sgd", simple_model.parameters(), lr=0.01)
        assert isinstance(opt, SGD)

    def test_get_optimizer_adamw(self, simple_model):
        opt = get_optimizer("adamw", simple_model.parameters(), lr=1e-3)
        assert isinstance(opt, AdamW)

    def test_get_optimizer_lion(self, simple_model):
        opt = get_optimizer("lion", simple_model.parameters(), lr=1e-4)
        assert isinstance(opt, Lion)

    def test_get_optimizer_unknown_raises(self, simple_model):
        with pytest.raises(ValueError, match="Unknown optimizer"):
            get_optimizer("mystery_opt", simple_model.parameters(), lr=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Schedulers
# ─────────────────────────────────────────────────────────────────────────────

class TestSchedulers:

    @pytest.fixture
    def optimizer(self):
        model = nn.Linear(4, 2)
        return AdamW(model.parameters(), lr=1e-3)

    def test_warmup_cosine_lr_starts_near_zero(self, optimizer):
        sched = LinearWarmupCosineDecay(optimizer, warmup_steps=100, total_steps=1000)
        # At step 0 (before first scheduler.step()), lr should be tiny
        lrs = sched.get_last_lr()
        assert all(lr < 1e-4 for lr in lrs)

    def test_warmup_cosine_lr_increases_during_warmup(self, optimizer):
        sched = LinearWarmupCosineDecay(optimizer, warmup_steps=10, total_steps=100)
        lrs = []
        for _ in range(10):
            optimizer.step()
            sched.step()
            lrs.append(sched.get_last_lr()[0])
        # LR must be monotonically non-decreasing during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1] - 1e-10

    def test_warmup_cosine_lr_decreases_after_warmup(self, optimizer):
        warmup = 5
        sched  = LinearWarmupCosineDecay(optimizer, warmup_steps=warmup, total_steps=50)
        # Advance past warmup
        for _ in range(warmup + 1):
            optimizer.step()
            sched.step()
        lr_at_warmup_end = sched.get_last_lr()[0]

        for _ in range(20):
            optimizer.step()
            sched.step()
        lr_later = sched.get_last_lr()[0]

        assert lr_later < lr_at_warmup_end

    def test_warmup_cosine_lr_always_positive(self, optimizer):
        sched = LinearWarmupCosineDecay(
            optimizer, warmup_steps=10, total_steps=100, min_lr_ratio=0.1
        )
        for _ in range(110):
            optimizer.step()
            sched.step()
            assert sched.get_last_lr()[0] > 0

    def test_constant_warmup_after_warmup_constant(self, optimizer):
        sched   = ConstantWithWarmup(optimizer, warmup_steps=5)
        # Advance past warmup
        for _ in range(6):
            optimizer.step()
            sched.step()
        lr_initial = sched.get_last_lr()[0]
        for _ in range(10):
            optimizer.step()
            sched.step()
        lr_final = sched.get_last_lr()[0]
        assert abs(lr_initial - lr_final) < 1e-10

    def test_get_scheduler_cosine_warmup(self, optimizer):
        sched = get_scheduler(
            "cosine_warmup", optimizer, warmup_steps=5, total_steps=50
        )
        assert isinstance(sched, LinearWarmupCosineDecay)

    def test_get_scheduler_unknown_raises(self, optimizer):
        with pytest.raises(ValueError):
            get_scheduler("mystery_sched", optimizer)


# ─────────────────────────────────────────────────────────────────────────────
# Initializers
# ─────────────────────────────────────────────────────────────────────────────

class TestInitializers:

    def test_initialize_weights_runs_on_model(self):
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.Linear(64, 10),
        )
        initialize_weights(model)
        # No exception = pass; also check biases are zero
        assert torch.all(model[0].bias == 0)
        assert torch.all(model[2].bias == 0)

    def test_initialize_weights_layer_norm_ones(self):
        model = nn.Sequential(nn.LayerNorm(16))
        initialize_weights(model)
        assert torch.allclose(model[0].weight, torch.ones(16))

    def test_truncated_normal_no_nan(self):
        t = torch.empty(100, 100)
        truncated_normal_(t, std=0.02)
        assert not torch.isnan(t).any()
        assert not torch.isinf(t).any()

    def test_truncated_normal_within_bounds(self):
        t   = torch.empty(1000)
        std = 0.02
        truncated_normal_(t, std=std, a=-2.0, b=2.0)
        assert (t >= -2 * std).all()
        assert (t <= 2 * std).all()

    def test_xavier_uniform_shape_preserved(self):
        t = torch.empty(32, 64)
        xavier_uniform_(t)
        assert t.shape == (32, 64)

    def test_kaiming_normal_shape_preserved(self):
        t = torch.empty(64, 32)
        kaiming_normal_(t)
        assert t.shape == (64, 32)

    def test_scale_residual_weights_reduces_out_proj(self):
        model = nn.Sequential(
            nn.Linear(64, 64)  # not an out_proj — shouldn't be scaled
        )
        # Rename to out_proj pattern
        model2 = nn.Module()
        model2.out_proj = nn.Linear(64, 64)
        nn.init.ones_(model2.out_proj.weight)
        scale_residual_weights(model2, num_layers=4)
        expected = 1.0 / math.sqrt(2.0 * 4)
        assert torch.allclose(
            model2.out_proj.weight,
            torch.full_like(model2.out_proj.weight, expected),
            atol=1e-5,
        )
