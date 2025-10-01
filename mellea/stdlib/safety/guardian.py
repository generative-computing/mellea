"""Risk checking with Granite Guardian models via existing backends."""

from enum import Enum
from typing import Dict, List, Optional, Union, Literal

from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib.base import CBlock, Context, ChatContext
from mellea.stdlib.chat import Message
from mellea.stdlib.funcs import _run_async_in_thread
from mellea.stdlib.requirement import Requirement, ValidationResult


class GuardianRisk(Enum):
    """Risk definitions for Granite Guardian models.

    Based on https://github.com/ibm-granite/granite-guardian but updated for 3.3 8B support.
    """

    HARM = "harm"
    GROUNDEDNESS = "groundedness"
    PROFANITY = "profanity"
    ANSWER_RELEVANCE = "answer_relevance"
    JAILBREAK = "jailbreak"
    FUNCTION_CALL = "function_call"
    SOCIAL_BIAS = "social_bias"
    VIOLENCE = "violence"
    SEXUAL_CONTENT = "sexual_content"
    UNETHICAL_BEHAVIOR = "unethical_behavior"

    @classmethod
    def get_available_risks(cls) -> List[str]:
        """Get list of all available risk types."""
        return [risk.value for risk in cls]


BackendType = Literal["huggingface", "ollama"]


def _parse_safety_result(result: Optional[str], logger) -> str:
    """Parse the model output to a Guardian label: Yes/No/Failed.

    Guardian returns yes/no between <score> and </score> tags.
    Handles case variations (Yes/yes, No/no) and whitespace.
    """
    if not result:
        logger.warning("Guardian returned empty result")
        return "Failed"

    s = str(result).lower()

    # Extract from <score>yes/no</score> tags
    if "<score>" in s and "</score>" in s:
        score = s.split("<score>")[1].split("</score>")[0].strip()
        if score == "yes":
            return "Yes"
        if score == "no":
            return "No"

    logger.warning(f"Could not parse safety result: {result}")
    return "Failed"


class GuardianCheck(Requirement):
    """Enhanced risk checking using Granite Guardian 3.3 8B with multiple backend support."""

    def __init__(
        self,
        risk: Union[str, GuardianRisk, None] = None,
        *,
        backend_type: BackendType = "ollama",
        model_version: Optional[str] = None,
        device: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
        thinking: bool = False,
        custom_criteria: Optional[str] = None,
        context_text: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
    ):
        """Initialize GuardianCheck using existing backends with minimal glue."""
        super().__init__(check_only=True, validation_fn=lambda c: self._guardian_validate(c))

        # Handle risk specification with custom criteria priority
        if custom_criteria:
            # When custom_criteria is provided, risk becomes optional
            if risk is None:
                self._risk = "custom"  # Default fallback risk identifier
            elif isinstance(risk, GuardianRisk):
                self._risk = risk.value
            else:
                self._risk = risk
        else:
            # When no custom_criteria, risk is required
            if risk is None:
                raise ValueError("Either 'risk' or 'custom_criteria' must be provided")
            if isinstance(risk, GuardianRisk):
                self._risk = risk.value
            else:
                self._risk = risk

        self._custom_criteria = custom_criteria
        self._thinking = thinking
        self._backend_type = backend_type
        self._context_text = context_text
        self._tools = tools

        # Choose defaults and initialize the chosen backend directly.
        if model_version is None:
            model_version = (
                "ibm-granite/granite-guardian-3.3-8b"
                if backend_type == "huggingface"
                else "ibm/granite3.3-guardian:8b"
            )

        if backend_type == "huggingface":
            from mellea.backends.huggingface import LocalHFBackend
            self._backend = LocalHFBackend(model_id=model_version)
        elif backend_type == "ollama":
            from mellea.backends.ollama import OllamaModelBackend
            self._backend = OllamaModelBackend(model_id=model_version, base_url=ollama_url)
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")

        # Provide a predictable attribute for the example to print.
        try:
            setattr(self._backend, "model_version", model_version)
        except Exception:
            pass

        self._logger = FancyLogger.get_logger()

    def get_effective_risk(self) -> str:
        """Get the effective risk criteria to use for validation."""
        return self._custom_criteria if self._custom_criteria else self._risk

    @classmethod
    def get_available_risks(cls) -> List[str]:
        """Get list of all available standard risk types."""
        return GuardianRisk.get_available_risks()

    def __deepcopy__(self, memo):
        """Custom deepcopy to handle unpicklable backend objects."""
        from copy import deepcopy
        # Create a new instance without calling __init__
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        # Copy all attributes except the backend (which contains locks)
        for k, v in self.__dict__.items():
            if k == '_backend':
                # Share the backend reference instead of copying it
                setattr(result, k, v)
            elif k == '_logger':
                # Share the logger reference
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

    def _guardian_validate(self, ctx: Context) -> ValidationResult:
        """Validate the last turn using Granite Guardian via selected backend."""
        # Define async validation logic
        async def _async_validate():
            logger = self._logger

            last_turn = ctx.last_turn()
            if last_turn is None:
                logger.warning("No last turn found in context")
                return ValidationResult(False, reason="No content to validate")

            # Build a fresh chat context for the guardian model.
            gctx = ChatContext()

            effective_risk = self.get_effective_risk()

            if (self._risk == "groundedness" or effective_risk == "groundedness") and self._context_text:
                gctx = gctx.add(Message("user", f"Document: {self._context_text}"))

            # Add the last user message if present.
            if last_turn.model_input is not None:
                if isinstance(last_turn.model_input, CBlock) and last_turn.model_input.value is not None:
                    gctx = gctx.add(Message("user", last_turn.model_input.value))
                elif isinstance(last_turn.model_input, Message):
                    gctx = gctx.add(Message(last_turn.model_input.role, last_turn.model_input.content))
                else:
                    gctx = gctx.add(Message("user", str(last_turn.model_input)))

            # Add the assistant response, optionally including tool call info for function_call risk.
            if last_turn.output is not None:
                assistant_text = last_turn.output.value or ""
                if getattr(last_turn.output, "tool_calls", None) and (self._risk == "function_call" or effective_risk == "function_call"):
                    calls = []
                    for name, tc in last_turn.output.tool_calls.items():
                        calls.append(f"{name}({getattr(tc, 'args', {})})")
                    if calls:
                        suffix = f" [Function calls: {', '.join(calls)}]"
                        assistant_text = (assistant_text + suffix) if assistant_text else suffix
                if assistant_text:
                    gctx = gctx.add(Message("assistant", assistant_text))

            # Ensure we have something to validate.
            history = gctx.view_for_generation() or []
            if len(history) == 0:
                logger.warning("No messages found to validate")
                return ValidationResult(False, reason="No messages to validate")

            # Backend options (mapped by backends internally to their specific keys).
            model_options: Dict[str, object] = {}
            if self._backend_type == "ollama":
                # Ollama templates expect the risk as the system prompt
                model_options["system"] = effective_risk
                model_options.update({
                    "temperature": 0.0,
                    "num_predict": 4000 if self._thinking else 50,
                    "stream": False,
                    "think": True if self._thinking else None,
                })
            else:  # huggingface
                # HF chat template for guardian expects guardian_config instead of a system message
                guardian_cfg: Dict[str, object] = {"risk": effective_risk}
                if self._custom_criteria:
                    guardian_cfg["custom_criteria"] = self._custom_criteria
                if self._context_text and (self._risk == "groundedness" or effective_risk == "groundedness"):
                    guardian_cfg["context"] = self._context_text

                model_options.update({
                    "guardian_config": guardian_cfg,
                    "max_new_tokens": 4000 if self._thinking else 50,
                    "stream": False,
                })

            # Attach tools for function_call checks.
            # Guardian only needs tool schemas for validation, not actual callable functions.
            if (self._risk == "function_call" or effective_risk == "function_call") and self._tools:
                model_options["tools"] = self._tools

            # Generate the guardian decision with a blank assistant turn.
            mot, _ = self._backend.generate_from_context(
                Message("assistant", ""), gctx, model_options=model_options
            )
            await mot.avalue()

            # Prefer explicit thinking if available, else try to split from output text.
            trace = getattr(mot, "_thinking", None)
            text = mot.value or ""
            if trace is None and "</think>" in text:
                parts = text.split("</think>")
                if len(parts) > 1:
                    trace = parts[0].replace("<think>", "").strip()
                    text = parts[1].strip()

            label = _parse_safety_result(text, logger)
            is_safe = label == "No"

            reason_parts = [f"Guardian check for '{effective_risk}': {label}"]
            if trace:
                reason_parts.append(f"Reasoning: {trace}")

            return ValidationResult(result=is_safe, reason="; ".join(reason_parts), thunk=mot)

        # Run the async validation using mellea's standard pattern
        return _run_async_in_thread(_async_validate())
