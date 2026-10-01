from .base import BaseStrategy
from .cot import CotStrategy
from .instruction import InstructionStrategy
from .eval_patient import EvalPatientStrategy
from .self_eval import SelfEvalStrategy

STRATEGY_REGISTRY = {
    "cot": CotStrategy,
    "instruction": InstructionStrategy,
    "eval_patient": EvalPatientStrategy,
    "self_eval": SelfEvalStrategy,
}


def get_strategy(name: str) -> BaseStrategy:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name]()
