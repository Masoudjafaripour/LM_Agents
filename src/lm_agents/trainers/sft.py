from lm_agents.trainers.base import Trainer


class SFTTrainer(Trainer):
    """Supervised fine-tuning on fixed agent trajectories (loss on assistant turns only)."""

    def train(self) -> None:
        raise NotImplementedError
