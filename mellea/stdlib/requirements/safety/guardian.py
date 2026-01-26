"""Risk checking with Granite Guardian models via existing backends."""

import warnings
from enum import Enum
from typing import Literal

from ....core import (
    Backend,
    BaseModelSubclass,
    CBlock,
    Context,
    FancyLogger,
    Requirement,
    ValidationResult,
)
from ...components import Message
from ...context import ChatContext


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
    def get_available_risks(cls) -> list[str]:
        """Get list of all available risk types."""
        return [risk.value for risk in cls]


BackendType = Literal["huggingface", "ollama"]


def _parse_safety_result(result: str | None, logger) -> str:
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
    """Enhanced risk checking using Granite Guardian 3.3 8B with multiple backend support.

    [DEPRECATED as of V 0.4 -- Use Intrinsics instead]
    """

    def __init__(
        self,
        risk: str | GuardianRisk | None = None,
        *,
        backend_type: BackendType = "ollama",
        model_version: str | None = None,
        device: str | None = None,
        ollama_url: str = "http://localhost:11434",
        thinking: bool = False,
        custom_criteria: str | None = None,
        context_text: str | None = None,
        tools: list[dict] | None = None,
        backend: Backend | None = None,
    ):
        """Initialize GuardianCheck using existing backends with minimal glue.

        Args:
            risk: The type of risk to check for (harm, jailbreak, etc.)
            backend_type: Type of backend to use ("ollama" or "huggingface")
            model_version: Specific model version to use
            device: Device for model inference (for HuggingFace)
            ollama_url: URL for Ollama server
            thinking: Enable thinking/reasoning mode
            custom_criteria: Custom criteria for validation
            context_text: Context document for groundedness checks
            tools: Tool schemas for function call validation
            backend: Pre-initialized backend to reuse (avoids loading model multiple times)
        """
        super().__init__(check_only=True)

        warnings.warn(
            "GuardianCheck is deprecated as of version 0.4. "
            "Use the Guardian Intrinsics instead ",
            DeprecationWarning,
            stacklevel=2,
        )

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
        self._context_text = context_text
        self._tools = tools

        # Use provided backend or create a new one
        if backend is not None:
            self._backend = backend
            # Infer backend_type from the provided backend
            from mellea.backends.huggingface import LocalHFBackend
            from mellea.backends.ollama import OllamaModelBackend

            if isinstance(backend, LocalHFBackend):
                self._backend_type = "huggingface"
            elif isinstance(backend, OllamaModelBackend):
                self._backend_type = "ollama"
            else:
                # Keep the provided backend_type as fallback
                self._backend_type = backend_type
        else:
            self._backend_type = backend_type
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

                self._backend = OllamaModelBackend(
                    model_id=model_version, base_url=ollama_url
                )
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
    def get_available_risks(cls) -> list[str]:
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
            if k == "_backend":
                # Share the backend reference instead of copying it
                setattr(result, k, v)
            elif k == "_logger":
                # Share the logger reference
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

    async def validate(
        self,
        backend: Backend,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
    ) -> ValidationResult:
        """Validate conversation using Granite Guardian via selected backend."""
        logger = self._logger

        # Build a fresh chat context for the guardian model (keep it minimal).
        gctx = ChatContext()

        effective_risk = self.get_effective_risk()

        # For groundedness: add doc only for Ollama; HF receives context via guardian_config
        if (
            (self._risk == "groundedness" or effective_risk == "groundedness")
            and self._context_text
            and self._backend_type == "ollama"
        ):
            gctx = gctx.add(Message("user", f"Document: {self._context_text}"))

        # Try to reuse chat history directly when available.
        messages = None
        try:
            from ...components.chat import as_chat_history

            messages = as_chat_history(ctx)
        except Exception:
            messages = None

        if messages:
            for m in messages:
                gctx = gctx.add(m)
        else:
            # Fallback: build from the last turn only
            last_turn = ctx.last_turn()
            if last_turn is None:
                logger.warning("No last turn found in context")
                return ValidationResult(False, reason="No content to validate")

            if last_turn.model_input is not None:
                gctx = gctx.add(last_turn.model_input)

            if last_turn.output is not None:
                # For function call risk, append tool call info as text; otherwise add thunk directly.
                if self._risk == "function_call" or effective_risk == "function_call":
                    content = last_turn.output.value or ""
                    tcalls = getattr(last_turn.output, "tool_calls", None)
                    if tcalls:
                        calls = [
                            f"{name}({getattr(tc, 'args', {})})"
                            for name, tc in tcalls.items()
                        ]
                        if calls:
                            suffix = f" [Tool calls: {', '.join(calls)}]"
                            content = (content + suffix) if content else suffix
                    if content:
                        gctx = gctx.add(Message("assistant", content))
                else:
                    gctx = gctx.add(last_turn.output)

        # Ensure we have something to validate.
        history = gctx.view_for_generation() or []
        if len(history) == 0:
            logger.warning("No messages found to validate")
            return ValidationResult(False, reason="No messages to validate")

        # Backend options (mapped by backends internally to their specific keys).
        guardian_options: dict[str, object] = {}
        if self._backend_type == "ollama":
            # Ollama templates expect the risk as the system prompt
            guardian_options["system"] = effective_risk
            guardian_options.update(
                {
                    "temperature": 0.0,
                    "num_predict": 4000 if self._thinking else 50,
                    "stream": False,
                    "think": True if self._thinking else None,
                }
            )
        else:  # huggingface
            # HF chat template for Guardian expects guardian_config and (optionally) documents
            guardian_cfg: dict[str, object] = {"criteria_id": effective_risk}
            if self._custom_criteria:
                # When using custom criteria, provide it as free-text criteria
                guardian_cfg["criteria_text"] = self._custom_criteria

            guardian_options.update(
                {
                    "guardian_config": guardian_cfg,
                    "think": self._thinking,  # Passed to apply_chat_template
                    # "add_generation_prompt": True,  # Guardian template requires a generation prompt. Mellea always does this for hugging face generation.
                    "max_new_tokens": 4000 if self._thinking else 50,
                    "stream": False,
                }
            )

            # Provide documents parameter for groundedness
            if self._context_text and (
                self._risk == "groundedness" or effective_risk == "groundedness"
            ):
                guardian_options["documents"] = [
                    {"doc_id": "0", "text": self._context_text}
                ]

        # Attach tools for function_call checks.
        # Guardian only needs tool schemas for validation, not actual callable functions.
        if (
            self._risk == "function_call" or effective_risk == "function_call"
        ) and self._tools:
            guardian_options["tools"] = self._tools

        # Generate the guardian decision.
        # For Ollama: add blank assistant turn to trigger generation
        # For HuggingFace: use CBlock (won't be added to conversation, add_generation_prompt handles the judge role)
        if self._backend_type == "ollama":
            action = Message("assistant", "")
        else:
            # Use a CBlock for HuggingFace - it won't be added as a message
            action = CBlock("")  # type: ignore

        mot, val_ctx = await self._backend.generate_from_context(
            action, gctx, model_options=guardian_options
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

        return ValidationResult(
            result=is_safe, reason="; ".join(reason_parts), thunk=mot, context=val_ctx
        )
