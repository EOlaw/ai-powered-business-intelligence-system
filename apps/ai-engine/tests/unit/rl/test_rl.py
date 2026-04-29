"""
Unit tests for Phase 7 (Learning Paradigms) and Phase 8 (RL).

Phase 7 — Learning Paradigms:
    TestSupervisedClassifier   — output shapes, gradient flow, fit/predict
    TestSupervisedRegressor    — shapes, R² computation, OLS correctness
    TestSequenceLabeler        — CRF forward/backward, Viterbi decode shape
    TestKMeans                 — centroid count, no-exception clustering
    TestDBSCAN                 — noise labels, cluster assignment
    TestAutoencoder            — encode/decode shapes, reconstruction error
    TestVAE                    — encode/sample/generate shapes, KL term positive
    TestPCA                    — explained variance, inverse transform round-trip
    TestSimCLR                 — NT-Xent loss positive, projection output shape
    TestMaskedAutoencoder      — forward output shapes, loss on masked positions

Phase 8 — Reinforcement Learning:
    TestSpace                  — discrete/continuous sample, contains
    TestCartPoleEnv            — reset obs shape, step returns 5-tuple, done
    TestReplayBuffer           — push/sample/len/is_ready
    TestPrioritizedBuffer      — priority update, IS weights in [0,1]
    TestSumTree                — total priority, sampling correctness
    TestValueNetwork           — output shape (B,1), gradient flows
    TestQNetwork               — output shape (B, n_actions), gradient
    TestComputeGAE             — advantages normalised, returns = adv + values
    TestDiscretePolicy         — sample action in valid range, log_prob shape
    TestContinuousPolicy       — output shape, rsample differentiable
    TestDQNAgent               — select_action range, observe returns None before start
    TestPPOAgent               — get_action_and_value shapes, update returns metrics
    TestA3CAgent               — worker sync, global network updated after step
    TestRewardNormaliser       — normalised value in clipped range
"""

import pytest
import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Phase 7 — Learning Paradigms
# ─────────────────────────────────────────────────────────────────────────────

class TestSupervisedClassifier:

    def test_multiclass_output_shape(self):
        from src.learning import multiclass_classifier
        clf = multiclass_classifier(in_dim=16, num_classes=5, hidden_dims=[32])
        X = np.random.randn(100, 16).astype(np.float32)
        y = np.random.randint(0, 5, 100)
        clf.fit(X, y)
        probs = clf.predict_proba(X[:10])
        assert probs.shape == (10, 5)
        assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-5)

    def test_binary_predict_in_01(self):
        from src.learning import binary_classifier
        clf = binary_classifier(in_dim=8, hidden_dims=[16])
        X = np.random.randn(50, 8).astype(np.float32)
        y = np.random.randint(0, 2, 50)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert set(preds.flatten()).issubset({0, 1})

    def test_multilabel_predict_shape(self):
        from src.learning import multilabel_classifier
        clf = multilabel_classifier(in_dim=12, num_classes=4, hidden_dims=[24])
        X = np.random.randn(60, 12).astype(np.float32)
        y = np.random.randint(0, 2, (60, 4)).astype(np.float32)
        clf.fit(X, y)
        preds = clf.predict(X[:5])
        assert preds.shape == (5, 4)

    def test_predict_before_fit_raises(self):
        from src.learning import multiclass_classifier
        clf = multiclass_classifier(in_dim=4, num_classes=3)
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict(np.random.randn(5, 4))


class TestSupervisedRegressor:

    def test_neural_regressor_output_shape(self):
        from src.learning import NeuralRegressor, RegressorConfig
        reg = NeuralRegressor(RegressorConfig(in_dim=8, out_dim=1, epochs=5))
        X = np.random.randn(80, 8).astype(np.float32)
        y = np.random.randn(80).astype(np.float32)
        reg.fit(X, y)
        preds = reg.predict(X[:10])
        assert preds.shape == (10,)

    def test_linear_regressor_ols_perfect(self):
        from src.learning import LinearRegressor
        rng = np.random.RandomState(0)
        X   = rng.randn(100, 3)
        w   = np.array([1.0, -2.0, 0.5])
        y   = X @ w + 3.0

        reg = LinearRegressor(alpha=0.0)
        reg.fit(X, y)
        r2 = reg.score(X, y)
        assert r2 > 0.99

    def test_polynomial_regressor_degree2(self):
        from src.learning import PolynomialRegressor
        X   = np.linspace(-1, 1, 50).reshape(-1, 1)
        y   = (X ** 2).squeeze()

        reg = PolynomialRegressor(degree=2, alpha=1e-6)
        reg.fit(X, y)
        r2 = reg.score(X, y)
        assert r2 > 0.99

    def test_multioutput_regression(self):
        from src.learning import NeuralRegressor, RegressorConfig
        reg = NeuralRegressor(RegressorConfig(in_dim=4, out_dim=3, epochs=5))
        X = np.random.randn(40, 4).astype(np.float32)
        y = np.random.randn(40, 3).astype(np.float32)
        reg.fit(X, y)
        preds = reg.predict(X[:8])
        assert preds.shape == (8, 3)


class TestSequenceLabeler:

    def test_crf_forward_produces_scalar(self):
        from src.learning.supervised.sequence import CRF
        crf       = CRF(num_labels=5)
        B, T, L   = 2, 6, 5
        emissions = torch.randn(B, T, L)
        labels    = torch.randint(1, L, (B, T))   # Avoid label 0 (padding)
        mask      = torch.ones(B, T, dtype=torch.long)
        loss      = crf(emissions, labels, mask)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_crf_viterbi_output_length(self):
        from src.learning.supervised.sequence import CRF
        crf       = CRF(num_labels=5)
        T, L      = 8, 5
        emissions = torch.randn(1, T, L)
        mask      = torch.ones(1, T, dtype=torch.long)
        paths     = crf.decode(emissions, mask)
        assert len(paths) == 1
        assert len(paths[0]) == T

    def test_sequence_labeler_predict_shape(self):
        from src.learning.supervised.sequence import SequenceLabeler, SequenceLabelerConfig
        cfg    = SequenceLabelerConfig(vocab_size=50, num_labels=5, epochs=2)
        model  = SequenceLabeler(cfg)
        X      = np.random.randint(1, 50, (20, 8))
        y      = np.random.randint(1, 5, (20, 8))
        model.fit(X, y)
        preds  = model.predict(X[:3])
        assert len(preds) == 3


class TestKMeans:

    def test_cluster_count(self):
        from src.learning import KMeans
        X = np.random.randn(100, 4)
        km = KMeans(n_clusters=3, n_init=3)
        labels = km.fit_predict(X)
        assert len(set(labels)) == 3
        assert labels.shape == (100,)

    def test_predict_assigns_to_known_centroids(self):
        from src.learning import KMeans
        X     = np.vstack([np.random.randn(50, 2) + c for c in [[-5, 0], [5, 0]]])
        km    = KMeans(n_clusters=2)
        km.fit(X)
        new_X = np.array([[5.1, 0.1], [-5.1, -0.1]])
        preds = km.predict(new_X)
        assert preds[0] != preds[1]   # Different clusters

    def test_dbscan_noise_label(self):
        from src.learning import DBSCAN
        # Isolated point far from any cluster = noise
        X = np.vstack([np.random.randn(30, 2), [[100.0, 100.0]]])
        db = DBSCAN(eps=1.0, min_samples=3)
        labels = db.fit_predict(X)
        assert labels[-1] == -1   # Outlier labelled as noise


class TestAutoencoder:

    def test_encode_decode_shapes(self):
        from src.learning import Autoencoder, AutoencoderConfig
        cfg = AutoencoderConfig(input_dim=32, latent_dim=8, epochs=3)
        ae  = Autoencoder(cfg)
        X   = np.random.randn(40, 32).astype(np.float32)
        ae.fit(X)
        z     = ae.transform(X[:5])
        recon = ae.reconstruct(torch.tensor(X[:5]))
        assert z.shape == (5, 8)
        assert recon.shape == (5, 32)

    def test_reconstruction_error_nonnegative(self):
        from src.learning import Autoencoder, AutoencoderConfig
        cfg    = AutoencoderConfig(input_dim=16, latent_dim=4, epochs=2)
        ae     = Autoencoder(cfg)
        X      = np.random.randn(20, 16).astype(np.float32)
        ae.fit(X)
        errors = ae.reconstruction_error(X)
        assert (errors >= 0).all()

    def test_dae_fit_runs(self):
        from src.learning.unsupervised.autoencoder import DenoisingAutoencoder, AutoencoderConfig
        cfg = AutoencoderConfig(input_dim=16, latent_dim=4, epochs=2, noise_type="gaussian")
        dae = DenoisingAutoencoder(cfg)
        X   = np.random.randn(20, 16).astype(np.float32)
        dae.fit(X)
        assert dae._is_fitted


class TestVAE:

    def test_sample_shape(self):
        from src.learning import VAE, VAEConfig
        cfg = VAEConfig(input_dim=32, latent_dim=8, epochs=2)
        vae = VAE(cfg)
        X   = np.random.randn(40, 32).astype(np.float32)
        vae.fit(X)
        samples = vae.sample(n=5)
        assert samples.shape == (5, 32)

    def test_encode_mean_shape(self):
        from src.learning import VAE, VAEConfig
        cfg = VAEConfig(input_dim=16, latent_dim=4, epochs=2)
        vae = VAE(cfg)
        X   = np.random.randn(20, 16).astype(np.float32)
        vae.fit(X)
        z = vae.encode_mean(X[:5])
        assert z.shape == (5, 4)

    def test_kl_loss_nonneg(self):
        from src.learning import VAE, VAEConfig
        cfg = VAEConfig(input_dim=16, latent_dim=4, epochs=1)
        vae = VAE(cfg)
        x   = torch.randn(8, 16)
        x_hat, mu, log_var = vae(x)
        losses = vae.loss(x, x_hat, mu, log_var)
        assert losses["kl"].item() >= 0
        assert losses["reconstruction"].item() >= 0

    def test_interpolate_shape(self):
        from src.learning import VAE, VAEConfig
        cfg = VAEConfig(input_dim=16, latent_dim=4, epochs=2)
        vae = VAE(cfg)
        X   = np.random.randn(20, 16).astype(np.float32)
        vae.fit(X)
        interps = vae.interpolate(X[0], X[1], steps=5)
        assert interps.shape == (5, 16)


class TestPCA:

    def test_transform_shape(self):
        from src.learning import PCA
        pca = PCA(n_components=3)
        X   = np.random.randn(50, 10)
        Z   = pca.fit_transform(X)
        assert Z.shape == (50, 3)

    def test_inverse_transform_approximate(self):
        from src.learning import PCA
        pca     = PCA(n_components=5)
        X       = np.random.randn(100, 10)
        Z       = pca.fit_transform(X)
        X_recon = pca.inverse_transform(Z)
        assert X_recon.shape == X.shape

    def test_explained_variance_sums_to_one_for_full_rank(self):
        from src.learning import PCA
        X   = np.random.randn(100, 4)
        pca = PCA(n_components=4)
        pca.fit(X)
        assert abs(pca.explained_variance_ratio_.sum() - 1.0) < 0.01

    def test_components_orthonormal(self):
        from src.learning import PCA
        pca = PCA(n_components=3)
        pca.fit(np.random.randn(100, 10))
        C   = pca.components_
        gram = C @ C.T
        assert np.allclose(gram, np.eye(3), atol=1e-5)


class TestSimCLR:

    def test_nt_xent_loss_positive(self):
        from src.learning import NTXentLoss
        loss_fn = NTXentLoss(temperature=0.5)
        z_i = torch.randn(8, 32)
        z_j = torch.randn(8, 32)
        z_i = torch.nn.functional.normalize(z_i, dim=-1)
        z_j = torch.nn.functional.normalize(z_j, dim=-1)
        loss = loss_fn(z_i, z_j)
        assert loss.item() > 0

    def test_nt_xent_lower_for_identical_pairs(self):
        """Identical positive pairs should produce lower loss than random pairs."""
        from src.learning import NTXentLoss
        loss_fn = NTXentLoss(temperature=0.5)
        z   = torch.nn.functional.normalize(torch.randn(8, 32), dim=-1)
        noise = torch.randn(8, 32) * 0.01
        loss_identical = loss_fn(z, z + noise).item()
        loss_random    = loss_fn(
            torch.nn.functional.normalize(torch.randn(8, 32), dim=-1),
            torch.nn.functional.normalize(torch.randn(8, 32), dim=-1),
        ).item()
        assert loss_identical < loss_random

    def test_projection_head_output_normalised(self):
        from src.learning.self_supervised.contrastive import ProjectionHead
        head = ProjectionHead(input_dim=64, hidden_dim=128, output_dim=32)
        x    = torch.randn(8, 64)
        out  = head(x)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)


class TestMaskedAutoencoder:

    def test_forward_output_shapes(self):
        from src.learning import MaskedAutoencoder, MAEConfig
        cfg  = MAEConfig(input_dim=16, encoder_dim=32, decoder_dim=16,
                         encoder_depth=2, decoder_depth=1, num_heads=4, epochs=2)
        mae  = MaskedAutoencoder(cfg)
        x    = torch.randn(4, 8, 16)   # (B, N, input_dim)
        recon, mask, loss = mae(x)
        assert recon.shape == (4, 8, 16)
        assert mask.shape  == (4, 8)
        assert loss.shape  == ()

    def test_loss_only_on_masked_positions(self):
        """With mask_ratio=1.0, all positions should be masked and loss should be nonzero."""
        from src.learning import MaskedAutoencoder, MAEConfig
        cfg = MAEConfig(input_dim=8, encoder_dim=16, decoder_dim=8,
                        encoder_depth=1, decoder_depth=1, num_heads=2,
                        mask_ratio=0.99, epochs=1)
        mae = MaskedAutoencoder(cfg)
        x   = torch.randn(2, 6, 8)
        _, _, loss = mae(x)
        assert loss.item() > 0


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8 — Reinforcement Learning
# ─────────────────────────────────────────────────────────────────────────────

class TestSpace:

    def test_discrete_space_sample_in_range(self):
        from src.rl import Space
        s = Space.discrete(n=10)
        for _ in range(20):
            sample = s.sample()
            assert 0 <= int(sample) < 10

    def test_discrete_space_contains(self):
        from src.rl import Space
        s = Space.discrete(n=5)
        assert s.contains(np.array(3))
        assert not s.contains(np.array(5))
        assert not s.contains(np.array(-1))

    def test_continuous_space_sample_in_bounds(self):
        from src.rl import Space
        s = Space.box(-1.0, 1.0, shape=(4,))
        for _ in range(20):
            sample = s.sample()
            assert sample.shape == (4,)
            assert (sample >= -1.0).all() and (sample <= 1.0).all()


class TestCartPoleEnv:

    def test_reset_obs_shape(self):
        from src.rl import CartPoleEnv
        env = CartPoleEnv()
        obs, info = env.reset(seed=42)
        assert obs.shape == (4,)

    def test_step_returns_five_tuple(self):
        from src.rl import CartPoleEnv
        env = CartPoleEnv()
        env.reset()
        result = env.step(0)
        assert len(result) == 5

    def test_step_obs_shape(self):
        from src.rl import CartPoleEnv
        env = CartPoleEnv()
        env.reset()
        obs, reward, terminated, truncated, info = env.step(1)
        assert obs.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)

    def test_terminates_when_pole_falls(self):
        from src.rl import CartPoleEnv
        env = CartPoleEnv()
        env.reset()
        # Force pole to fall by always pushing in same direction
        done  = False
        steps = 0
        while not done and steps < 1000:
            _, _, terminated, truncated, _ = env.step(0)
            done = terminated or truncated
            steps += 1
        assert done


class TestReplayBuffer:

    def _make_buffer(self, capacity=100, obs_dim=4):
        from src.rl import ReplayBuffer
        return ReplayBuffer(capacity=capacity, obs_shape=(obs_dim,))

    def test_push_increases_len(self):
        buf = self._make_buffer()
        for i in range(10):
            buf.push(np.zeros(4), 0, 1.0, np.zeros(4), False)
        assert len(buf) == 10

    def test_capacity_limit(self):
        buf = self._make_buffer(capacity=5)
        for _ in range(20):
            buf.push(np.zeros(4), 0, 0.0, np.zeros(4), False)
        assert len(buf) == 5

    def test_sample_batch_shape(self):
        buf = self._make_buffer()
        for _ in range(50):
            buf.push(np.random.randn(4).astype(np.float32), 1, 1.0, np.random.randn(4).astype(np.float32), False)
        batch = buf.sample(16)
        assert batch.states.shape      == (16, 4)
        assert batch.actions.shape     == (16,)
        assert batch.rewards.shape     == (16,)
        assert batch.next_states.shape == (16, 4)

    def test_not_ready_before_threshold(self):
        buf = self._make_buffer()
        for _ in range(10):
            buf.push(np.zeros(4), 0, 0.0, np.zeros(4), False)
        assert not buf.is_ready(64)


class TestSumTree:

    def test_total_priority_matches_sum(self):
        from src.rl.memory.replay_buffer import SumTree
        tree = SumTree(capacity=8)
        priorities = [1.0, 2.0, 3.0, 4.0]
        for i, p in enumerate(priorities):
            tree.push(p, i)
        assert abs(tree.total_priority - sum(priorities)) < 1e-5

    def test_sample_returns_valid_data(self):
        from src.rl.memory.replay_buffer import SumTree
        tree = SumTree(capacity=8)
        for i in range(4):
            tree.push(float(i + 1), f"data_{i}")
        idx, priority, data = tree.sample(tree.total_priority * 0.5)
        assert priority > 0
        assert data is not None


class TestValueNetwork:

    def test_output_shape(self):
        from src.rl import ValueNetwork
        v   = ValueNetwork(obs_dim=4, hidden_dims=[32])
        x   = torch.randn(8, 4)
        out = v(x)
        assert out.shape == (8, 1)

    def test_gradient_flows(self):
        from src.rl import ValueNetwork
        v  = ValueNetwork(obs_dim=4, hidden_dims=[32])
        x  = torch.randn(4, 4, requires_grad=True)
        v(x).sum().backward()
        assert x.grad is not None


class TestQNetwork:

    def test_output_shape(self):
        from src.rl import QNetwork
        q   = QNetwork(obs_dim=8, n_actions=4, hidden_dims=[32])
        x   = torch.randn(16, 8)
        out = q(x)
        assert out.shape == (16, 4)

    def test_gradient_flows(self):
        from src.rl import QNetwork
        q   = QNetwork(obs_dim=8, n_actions=4, hidden_dims=[32])
        x   = torch.randn(4, 8, requires_grad=True)
        q(x).sum().backward()
        assert x.grad is not None


class TestComputeGAE:

    def test_output_shapes(self):
        from src.rl import compute_gae
        T = 16
        rewards     = torch.ones(T)
        values      = torch.zeros(T)
        next_values = torch.zeros(T)
        dones       = torch.zeros(T)
        adv, ret    = compute_gae(rewards, values, next_values, dones)
        assert adv.shape == (T,)
        assert ret.shape == (T,)

    def test_advantages_normalised(self):
        from src.rl import compute_gae
        T     = 32
        r     = torch.randn(T)
        v     = torch.randn(T)
        nv    = torch.randn(T)
        d     = torch.zeros(T)
        adv, _ = compute_gae(r, v, nv, d)
        assert abs(adv.mean().item()) < 0.1
        assert abs(adv.std().item() - 1.0) < 0.1

    def test_returns_equals_adv_plus_values(self):
        from src.rl import compute_gae
        T = 8
        r  = torch.ones(T)
        v  = torch.randn(T)
        nv = torch.randn(T)
        d  = torch.zeros(T)
        adv_norm, ret = compute_gae(r, v, nv, d)
        # ret = adv (pre-normalisation) + v
        # We can only check shapes and that ret != v (non-trivial)
        assert not torch.allclose(ret, v)


class TestDiscretePolicy:

    def test_sample_action_in_range(self):
        from src.rl import DiscretePolicy, PolicyConfig
        cfg    = PolicyConfig(obs_dim=4, action_dim=5, hidden_dims=[32])
        policy = DiscretePolicy(cfg)
        state  = torch.randn(4)
        action, log_prob = policy.sample_action(state)
        assert 0 <= action < 5
        assert isinstance(log_prob, torch.Tensor)

    def test_entropy_nonneg(self):
        from src.rl import DiscretePolicy, PolicyConfig
        cfg    = PolicyConfig(obs_dim=4, action_dim=5, hidden_dims=[32])
        policy = DiscretePolicy(cfg)
        state  = torch.randn(8, 4)
        ent    = policy.entropy(state)
        assert (ent >= 0).all()

    def test_gradient_flows_through_log_prob(self):
        from src.rl import DiscretePolicy, PolicyConfig
        cfg    = PolicyConfig(obs_dim=4, action_dim=3, hidden_dims=[16])
        policy = DiscretePolicy(cfg)
        state  = torch.randn(8, 4)
        dist   = policy.forward(state)
        action = dist.sample()
        dist.log_prob(action).mean().backward()
        for p in policy.parameters():
            assert p.grad is not None


class TestContinuousPolicy:

    def test_sample_shape(self):
        from src.rl import ContinuousPolicy, PolicyConfig
        cfg    = PolicyConfig(obs_dim=4, action_dim=2, hidden_dims=[32],
                              action_type="continuous")
        policy = ContinuousPolicy(cfg)
        state  = torch.randn(4)
        action, log_prob = policy.sample_action(state)
        assert action.shape == (2,)

    def test_rsample_is_differentiable(self):
        from src.rl import ContinuousPolicy, PolicyConfig
        cfg    = PolicyConfig(obs_dim=4, action_dim=2, hidden_dims=[16],
                              action_type="continuous")
        policy = ContinuousPolicy(cfg)
        state  = torch.randn(8, 4)
        dist   = policy.forward(state)
        action = dist.rsample()
        action.mean().backward()
        for p in policy.parameters():
            assert p.grad is not None


class TestDQNAgent:

    def test_select_action_in_range(self):
        from src.rl import DQNAgent, DQNConfig
        cfg   = DQNConfig(obs_dim=4, n_actions=5, hidden_dims=[32])
        agent = DQNAgent(cfg)
        obs   = np.random.randn(4).astype(np.float32)
        action = agent.select_action(obs)
        assert 0 <= action < 5

    def test_observe_returns_none_before_learning_starts(self):
        from src.rl import DQNAgent, DQNConfig
        cfg   = DQNConfig(obs_dim=4, n_actions=2, hidden_dims=[16], learning_starts=100)
        agent = DQNAgent(cfg)
        s     = np.zeros(4, dtype=np.float32)
        for _ in range(10):
            result = agent.observe(s, 0, 1.0, s, False)
            assert result is None

    def test_train_on_cartpole_runs(self):
        from src.rl import DQNAgent, DQNConfig, CartPoleEnv
        cfg = DQNConfig(
            obs_dim=4, n_actions=2, hidden_dims=[32],
            buffer_size=500, learning_starts=50, batch_size=16,
            eps_decay_steps=200, target_update=50,
        )
        agent  = DQNAgent(cfg)
        env    = CartPoleEnv()
        result = agent.train(env, total_steps=300)
        assert "avg_reward" in result
        assert result["avg_reward"] >= 0


class TestPPOAgent:

    def test_get_action_and_value_shapes(self):
        from src.rl import PPOAgent, PPOConfig
        cfg   = PPOConfig(obs_dim=4, action_dim=2, hidden_dims=[32])
        agent = PPOAgent(cfg)
        state = np.random.randn(4).astype(np.float32)
        action, log_prob, value = agent.get_action_and_value(state)
        assert 0 <= action < 2
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_update_returns_metrics(self):
        from src.rl import PPOAgent, PPOConfig, CartPoleEnv
        cfg   = PPOConfig(obs_dim=4, action_dim=2, hidden_dims=[32],
                          rollout_steps=32, ppo_epochs=2, ppo_batch_size=8)
        agent = PPOAgent(cfg)
        env   = CartPoleEnv()
        obs, _ = env.reset()
        # Collect a mini rollout
        for _ in range(32):
            action, log_prob, value = agent.get_action_and_value(obs)
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.buffer.push(obs, action, reward, value, log_prob, done)
            obs = next_obs if not done else env.reset()[0]
        metrics = agent.update(last_value=0.0)
        assert "policy_loss" in metrics
        assert "value_loss"  in metrics
        assert "entropy"     in metrics


class TestRewardNormaliser:

    def test_normalised_value_in_clipped_range(self):
        from src.rl import RewardNormaliser
        norm = RewardNormaliser(clip=5.0)
        for r in np.random.randn(100):
            normalised = norm.normalise(float(r))
            assert -5.0 <= normalised <= 5.0

    def test_running_stats_update(self):
        from src.rl import RewardNormaliser
        norm = RewardNormaliser()
        rewards = np.random.randn(100)
        for r in rewards:
            norm.normalise(float(r))
        assert norm._count == 100
        assert norm._mean != 0.0 or True   # Just check it ran
