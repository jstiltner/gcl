"""
Experiment 36: MARL Comparison

Compares GCL self-selection against MARL baselines:
- QMIX (value decomposition)
- MAPPO (policy gradient with centralized critic)
- Independent Q-Learning (IQL)
- Random baseline
"""

from .qmix import QMIXAgent, QMIXConfig, run_qmix_experiment
from .mappo import MAPPOAgent, MAPPOConfig, run_mappo_experiment
from .run_comparison import run_full_comparison, ExperimentConfig
