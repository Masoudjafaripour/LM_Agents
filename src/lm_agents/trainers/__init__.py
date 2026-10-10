from lm_agents.trainers.opd import OPDTrainer
from lm_agents.trainers.rlft import RLFTTrainer
from lm_agents.trainers.sft import SFTTrainer

TRAINERS = {"sft": SFTTrainer, "opd": OPDTrainer, "rlft": RLFTTrainer}
