from lm_agents.trainers.base import Trainer


class OPDTrainer(Trainer):
    """On-policy distillation: sample student rollouts, minimize per-token reverse KL to the teacher."""

    def train(self) -> None:
        raise NotImplementedError
