from lm_agents.trainers.base import Trainer


class RLFTTrainer(Trainer):
    """RL fine-tuning (PPO / GRPO) on multi-turn rollouts with environment rewards."""

    def train(self) -> None:
        raise NotImplementedError
