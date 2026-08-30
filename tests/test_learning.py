"""Tests for the learning module."""

import numpy as np
import pytest
import torch

from gcl.learning.environment import (
    CommitmentAction,
    CommitmentEnv,
    CommitmentEnvConfig,
    CommitmentObservation,
    Task,
    TaskDifficulty,
    TaskGenerator,
)
from gcl.learning.policy import (
    CommitmentPolicy,
    HeuristicPolicy,
    RandomPolicy,
    SimpleCommitmentPolicy,
)
from gcl.learning.reward import (
    CalibrationTracker,
    RewardComputer,
    RewardConfig,
)
from gcl.learning.training import PPOConfig, PPOTrainer, RolloutBuffer


class TestTask:
    """Tests for Task class."""

    def test_create_task(self):
        """Test creating a task."""
        task = Task(
            difficulty=TaskDifficulty.MEDIUM,
            features=np.array([0.5, 0.3, 0.2]),
            true_success_probability=0.6,
        )
        
        assert task.difficulty == TaskDifficulty.MEDIUM
        assert task.true_success_probability == 0.6
        assert len(task.features) == 3

    def test_task_defaults(self):
        """Test task default values."""
        task = Task()
        
        assert task.id is not None
        assert task.difficulty == TaskDifficulty.MEDIUM
        assert task.true_success_probability == 0.5


class TestTaskGenerator:
    """Tests for TaskGenerator."""

    def test_generate_single_task(self):
        """Test generating a single task."""
        generator = TaskGenerator(feature_dim=8, seed=42)
        tasks = generator.generate()
        
        assert len(tasks) == 1
        assert len(tasks[0].features) == 8

    def test_generate_batch(self):
        """Test generating multiple tasks."""
        generator = TaskGenerator(seed=42)
        tasks = generator.generate(batch_size=10)
        
        assert len(tasks) == 10

    def test_generate_specific_difficulty(self):
        """Test generating tasks with specific difficulty."""
        generator = TaskGenerator(seed=42)
        
        easy_tasks = generator.generate(difficulty=TaskDifficulty.EASY, batch_size=10)
        for task in easy_tasks:
            assert task.difficulty == TaskDifficulty.EASY
            assert task.true_success_probability >= 0.8

    def test_reproducibility(self):
        """Test that same seed produces same tasks."""
        gen1 = TaskGenerator(seed=42)
        gen2 = TaskGenerator(seed=42)
        
        tasks1 = gen1.generate(batch_size=5)
        tasks2 = gen2.generate(batch_size=5)
        
        for t1, t2 in zip(tasks1, tasks2):
            assert t1.true_success_probability == t2.true_success_probability
            np.testing.assert_array_equal(t1.features, t2.features)


class TestCommitmentObservation:
    """Tests for CommitmentObservation."""

    def test_to_array(self):
        """Test converting observation to array."""
        obs = CommitmentObservation(
            task_features=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            reputation_score=1.5,
            available_stake=0.8,
            active_commitment_count=2,
            recent_success_rate=0.7,
        )
        
        arr = obs.to_array()
        
        assert len(arr) == 12  # 8 features + 4 state values
        assert arr[8] == pytest.approx(1.5)  # reputation
        assert arr[9] == pytest.approx(0.8)  # stake
        assert arr[10] == pytest.approx(2)  # count
        assert arr[11] == pytest.approx(0.7)  # success rate

    def test_from_array(self):
        """Test creating observation from array."""
        arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.5, 0.8, 2, 0.7])
        obs = CommitmentObservation.from_array(arr, feature_dim=8)
        
        assert obs.reputation_score == 1.5
        assert obs.available_stake == 0.8


class TestCommitmentAction:
    """Tests for CommitmentAction."""

    def test_from_array(self):
        """Test creating action from array."""
        arr = np.array([0.8, 0.7, 0.2])
        action = CommitmentAction.from_array(arr)
        
        assert action.commit is True  # 0.8 > 0.5
        assert action.confidence == pytest.approx(0.7)
        assert action.stake_fraction == pytest.approx(0.2)

    def test_to_array(self):
        """Test converting action to array."""
        action = CommitmentAction(commit=True, confidence=0.75, stake_fraction=0.15)
        arr = action.to_array()
        
        assert arr[0] == 1.0
        assert arr[1] == 0.75
        assert arr[2] == 0.15


class TestCommitmentEnv:
    """Tests for CommitmentEnv."""

    @pytest.fixture
    def env(self):
        """Create a test environment."""
        config = CommitmentEnvConfig(
            feature_dim=8,
            max_steps=50,
            initial_stake=100.0,
        )
        return CommitmentEnv(config=config, seed=42)

    def test_reset(self, env):
        """Test environment reset."""
        obs, info = env.reset()
        
        assert obs.shape == (12,)  # 8 features + 4 state
        assert "step" in info
        assert info["step"] == 0

    def test_step_commit(self, env):
        """Test stepping with a commit action."""
        env.reset()
        
        # Commit with high confidence
        action = np.array([0.9, 0.8, 0.1])
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == (12,)
        assert "outcome" in info
        assert info["outcome"] in ["success", "failure", "insufficient_stake"]

    def test_step_abstain(self, env):
        """Test stepping with abstain action."""
        env.reset()
        
        # Don't commit
        action = np.array([0.2, 0.5, 0.1])
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert info["outcome"] == "abstained"
        assert reward < 0  # Small penalty for abstaining

    def test_episode_termination(self, env):
        """Test that episode terminates after max steps."""
        env.reset()
        
        for _ in range(50):
            action = np.array([0.2, 0.5, 0.1])  # Abstain
            obs, reward, terminated, truncated, info = env.step(action)
            
            if truncated:
                break
        
        assert truncated  # Should truncate at max_steps

    def test_observation_space(self, env):
        """Test observation space."""
        assert env.observation_space.shape == (12,)

    def test_action_space(self, env):
        """Test action space."""
        assert env.action_space.shape == (3,)
        assert env.action_space.low.tolist() == [0.0, 0.0, 0.0]
        assert env.action_space.high.tolist() == [1.0, 1.0, 1.0]


class TestRewardComputer:
    """Tests for RewardComputer."""

    @pytest.fixture
    def computer(self):
        """Create a reward computer."""
        return RewardComputer()

    def test_compute_commitment_reward_success(self, computer):
        """Test reward for successful commitment."""
        from gcl.core.commitment import (
            ActionSpec,
            Consequence,
            ContextRegion,
            FailureMode,
            GroundedCommitment,
            VerificationResult,
            VerificationStatus,
        )
        from gcl.core.predicates import Predicate
        
        commitment = GroundedCommitment(
            issuer="agent-1",
            trigger_conditions=[Predicate(name="t", expression="True")],
            promised_behavior=ActionSpec(action_type="test"),
            success_condition=Predicate(name="s", expression="True"),
            failure_modes=[
                FailureMode(
                    name="f",
                    condition=Predicate(name="fc", expression="False"),
                    consequence=Consequence(
                        consequence_type="test",
                        magnitude=0.5,
                        description="Test",
                    ),
                    severity=0.5,
                )
            ],
            stake=10.0,
            confidence=0.8,
            valid_contexts=ContextRegion(description="Test"),
        )
        
        result = VerificationResult(
            status=VerificationStatus.SUCCESS,
            commitment_id=commitment.id,
        )
        
        reward = computer.compute_commitment_reward(
            commitment=commitment,
            result=result,
            true_success_prob=0.8,
            confidence=0.8,
        )
        
        assert reward.total > 0
        assert reward.base_reward > 0
        assert reward.stake_reward > 0

    def test_compute_abstain_reward(self, computer):
        """Test reward for abstaining."""
        # Abstain on impossible task
        reward = computer.compute_abstain_reward(
            true_success_prob=0.05,
            task_difficulty="impossible",
        )
        
        assert reward.strategic_reward > 0  # Bonus for correct abstain
        
        # Abstain on easy task
        reward = computer.compute_abstain_reward(
            true_success_prob=0.9,
            task_difficulty="easy",
        )
        
        assert reward.strategic_reward < 0  # Penalty for wrong abstain


class TestCalibrationTracker:
    """Tests for CalibrationTracker."""

    def test_record_and_compute(self):
        """Test recording outcomes and computing calibration."""
        tracker = CalibrationTracker(num_bins=10)
        
        # Record some outcomes
        # High confidence, mostly success
        for _ in range(8):
            tracker.record(0.85, True)
        for _ in range(2):
            tracker.record(0.85, False)
        
        # Low confidence, mostly failure
        for _ in range(3):
            tracker.record(0.25, True)
        for _ in range(7):
            tracker.record(0.25, False)
        
        stats = tracker.get_statistics()
        
        assert stats["total_samples"] == 20
        assert 0 <= stats["expected_calibration_error"] <= 1

    def test_calibration_curve(self):
        """Test getting calibration curve."""
        tracker = CalibrationTracker(num_bins=5)
        
        # Perfect calibration
        for conf in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for _ in range(10):
                success = np.random.random() < conf
                tracker.record(conf, success)
        
        curve = tracker.get_calibration_curve()
        
        assert len(curve) == 5


class TestCommitmentPolicy:
    """Tests for CommitmentPolicy."""

    @pytest.fixture
    def policy(self):
        """Create a policy."""
        return CommitmentPolicy(obs_dim=12, hidden_dim=32, num_layers=2)

    def test_forward(self, policy):
        """Test forward pass."""
        obs = torch.randn(4, 12)
        commit_logits, confidence_params, stake_params, value = policy.forward(obs)
        
        assert commit_logits.shape == (4, 1)
        assert confidence_params.shape == (4, 2)
        assert stake_params.shape == (4, 2)
        assert value.shape == (4, 1)

    def test_get_action(self, policy):
        """Test getting action."""
        obs = torch.randn(1, 12)
        action, log_prob, value = policy.get_action(obs)
        
        assert action.shape == (1, 3)
        assert 0 <= action[0, 0] <= 1  # Commit
        assert 0 <= action[0, 1] <= 1  # Confidence
        assert 0 <= action[0, 2] <= 1  # Stake

    def test_get_action_deterministic(self, policy):
        """Test deterministic action."""
        obs = torch.randn(1, 12)
        
        # Multiple calls should give same result
        action1, _, _ = policy.get_action(obs, deterministic=True)
        action2, _, _ = policy.get_action(obs, deterministic=True)
        
        np.testing.assert_array_almost_equal(action1, action2)

    def test_evaluate_actions(self, policy):
        """Test evaluating actions."""
        obs = torch.randn(4, 12)
        actions = torch.rand(4, 3)
        
        log_prob, entropy, value = policy.evaluate_actions(obs, actions)
        
        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)
        assert value.shape == (4, 1)


class TestRandomPolicy:
    """Tests for RandomPolicy."""

    def test_get_action(self):
        """Test random action."""
        policy = RandomPolicy(seed=42)
        obs = np.random.randn(12)
        
        action, log_prob, value = policy.get_action(obs)
        
        assert action.shape == (3,)
        assert all(0 <= a <= 1 for a in action)


class TestHeuristicPolicy:
    """Tests for HeuristicPolicy."""

    def test_get_action_easy_task(self):
        """Test action on easy task."""
        policy = HeuristicPolicy(difficulty_threshold=0.5)
        
        # Easy task (low difficulty indicator)
        obs = np.array([0.2] + [0.0] * 11)
        action, _, _ = policy.get_action(obs)
        
        assert action[0] == 1.0  # Should commit

    def test_get_action_hard_task(self):
        """Test action on hard task."""
        policy = HeuristicPolicy(difficulty_threshold=0.5)
        
        # Hard task (high difficulty indicator)
        obs = np.array([0.8] + [0.0] * 11)
        action, _, _ = policy.get_action(obs)
        
        assert action[0] == 0.0  # Should not commit


class TestRolloutBuffer:
    """Tests for RolloutBuffer."""

    def test_add_and_get(self):
        """Test adding transitions and getting tensors."""
        buffer = RolloutBuffer()
        
        for i in range(10):
            buffer.add(
                obs=np.random.randn(12).astype(np.float32),
                action=np.random.rand(3).astype(np.float32),
                reward=float(i),
                value=float(i * 0.1),
                log_prob=-0.5,
                done=i == 9,
            )
        
        assert len(buffer) == 10
        
        # Compute returns
        buffer.compute_returns_and_advantages(
            last_value=0.0,
            gamma=0.99,
            gae_lambda=0.95,
        )
        
        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert len(buffer.advantages) == 10

    def test_clear(self):
        """Test clearing buffer."""
        buffer = RolloutBuffer()
        buffer.add(
            obs=np.zeros(12),
            action=np.zeros(3),
            reward=0.0,
            value=0.0,
            log_prob=0.0,
            done=False,
        )
        
        assert len(buffer) == 1
        buffer.clear()
        assert len(buffer) == 0


class TestPPOTrainer:
    """Tests for PPOTrainer."""

    @pytest.fixture
    def trainer(self):
        """Create a trainer."""
        env_config = CommitmentEnvConfig(max_steps=20)
        env = CommitmentEnv(config=env_config, seed=42)
        
        ppo_config = PPOConfig(
            total_timesteps=200,
            n_steps=50,
            batch_size=16,
            n_epochs=2,
        )
        
        return PPOTrainer(env, config=ppo_config, device="cpu", seed=42)

    def test_train_short(self, trainer):
        """Test short training run."""
        metrics = trainer.train()
        
        assert metrics.timesteps >= 200
        assert metrics.episodes > 0
        assert len(metrics.episode_rewards) > 0

    def test_evaluate(self, trainer):
        """Test evaluation."""
        trainer.env.reset()
        
        eval_metrics = trainer.evaluate(n_episodes=3)
        
        assert "mean_reward" in eval_metrics
        assert "mean_length" in eval_metrics
