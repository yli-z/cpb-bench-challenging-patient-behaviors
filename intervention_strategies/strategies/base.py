from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class StrategyResult:
    generated_response: str
    extra_fields: dict = field(default_factory=dict)


class BaseStrategy(ABC):
    """Base class for all intervention strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @property
    @abstractmethod
    def output_dir(self) -> str:
        """Subdirectory name for saving results."""

    @abstractmethod
    async def process_case(self, case: dict, formatted_history: str,
                           conversation_segment: list, llm) -> StrategyResult:
        """Process a single case and return the result.

        Args:
            case: The case dict (row.to_dict()).
            formatted_history: Conversation formatted as a string.
            conversation_segment: Original conversation segment list.
            llm: The LLM model instance.

        Returns:
            StrategyResult with generated_response and optional extra_fields.
        """
