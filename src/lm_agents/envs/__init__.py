from lm_agents.envs.base import Env, StepResult

ENV_REGISTRY: dict[str, type[Env]] = {}


def register_env(name: str):
    def wrap(cls: type[Env]) -> type[Env]:
        ENV_REGISTRY[name] = cls
        return cls

    return wrap


def make_env(name: str, **kwargs) -> Env:
    if name not in ENV_REGISTRY:
        raise KeyError(f"Unknown env '{name}'. Registered: {sorted(ENV_REGISTRY)}")
    return ENV_REGISTRY[name](**kwargs)


# Import env subpackages so their @register_env decorators run.
from lm_agents.envs import humaneval, swebench  # noqa: F401

__all__ = ["ENV_REGISTRY", "Env", "StepResult", "make_env", "register_env"]
