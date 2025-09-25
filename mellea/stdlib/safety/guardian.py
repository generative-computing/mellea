"""Risk checking with Guardian models."""

import json
import torch
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union, Literal
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import requests
except ImportError:
    requests = None

from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib.base import CBlock, Context
from mellea.stdlib.chat import Message
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


def _parse_safety_result(result: str, logger) -> str:
    """Parse the safety result from model output."""
    result_lower = result.lower().strip()

    # Handle score tags format: <score> yes/no </score>
    if "<score>" in result_lower and "</score>" in result_lower:
        score_content = result_lower.split("<score>")[1].split("</score>")[0].strip()
        if "yes" in score_content:
            return "Yes"
        elif "no" in score_content:
            return "No"

    # Check for explicit yes/no at start
    if result_lower.startswith("yes"):
        return "Yes"
    elif result_lower.startswith("no"):
        return "No"

    # Check for yes/no anywhere in response
    if " yes " in result_lower or result_lower.endswith(" yes"):
        return "Yes"
    elif " no " in result_lower or result_lower.endswith(" no"):
        return "No"

    # Check for safety indicators
    if any(word in result_lower for word in ["safe", "no risk", "acceptable"]):
        return "No"
    elif any(word in result_lower for word in ["unsafe", "risky", "harmful", "violation"]):
        return "Yes"

    # Default to failed parsing
    logger.warning(f"Could not parse safety result: {result}")
    return "Failed"


class GuardianBackend:
    """Abstract base for different Guardian backends."""

    def __init__(self, model_version: str, device: Optional[str] = None):
        self.model_version = model_version
        self.device = device
        self._logger = FancyLogger.get_logger()

    def validate_messages(self, messages: List[Dict[str, str]], risk: str,
                         thinking: bool = False, tools: Optional[List[Dict]] = None) -> Tuple[str, Optional[str], Dict]:
        """Validate messages for specified risk. Returns (result, trace, raw_data)."""
        raise NotImplementedError


class HuggingFaceGuardianBackend(GuardianBackend):
    """HuggingFace-based Guardian backend for local model inference."""

    def __init__(self, model_version: str = "ibm-granite/granite-guardian-3.0-8b", device: Optional[str] = None):
        super().__init__(model_version, device)
        self._model = None
        self._tokenizer = None

        # Auto-device selection if not specified
        if device is None:
            device_name: str = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            self.device = torch.device(device_name)

    def _load_model(self):
        """Lazy load model and tokenizer."""
        if self._model is None:
            self._logger.info(f"Loading Granite Guardian model: {self.model_version}")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_version,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            self._model.to(self.device)
            self._model.eval()

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_version)

    def validate_messages(self, messages: List[Dict[str, str]], risk: str,
                         thinking: bool = False, tools: Optional[List[Dict]] = None,
                         context_text: Optional[str] = None) -> Tuple[str, Optional[str], Dict]:
        """Validate messages using HuggingFace backend."""
        self._load_model()

        guardian_config = {"risk_name": risk}

        # Apply chat template with thinking mode support
        input_ids = self._tokenizer.apply_chat_template(
            messages,
            guardian_config=guardian_config,
            add_generation_prompt=True,
            return_tensors="pt",
            think=thinking  # Enable thinking mode if supported
        ).to(self._model.device)

        input_len = input_ids.shape[1]

        # Generate with appropriate tokens for thinking mode
        max_tokens = 2000 if thinking else 20

        with torch.no_grad():
            output = self._model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=max_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Parse output
        full_response = self._tokenizer.decode(
            output.sequences[:, input_len:][0],
            skip_special_tokens=True
        ).strip()

        # Extract thinking trace if present
        trace = None
        result = full_response

        if thinking and "<think>" in full_response:
            parts = full_response.split("</think>")
            if len(parts) > 1:
                trace = parts[0].replace("<think>", "").strip()
                result = parts[1].strip()

        # Determine safety result
        label = _parse_safety_result(result, self._logger)

        return label, trace, {"full_response": full_response, "model": self.model_version}


class OllamaGuardianBackend(GuardianBackend):
    """Ollama-based Guardian backend for local model inference."""

    def __init__(self, model_version: str = "ibm/granite3.3-guardian:8b",
                 ollama_url: str = "http://localhost:11434"):
        super().__init__(model_version)
        self.ollama_url = ollama_url

        if requests is None:
            raise ImportError("requests library is required for Ollama backend. Install with: pip install requests")

    def validate_messages(self, messages: List[Dict[str, str]], risk: str,
                         thinking: bool = False, tools: Optional[List[Dict]] = None,
                         context_text: Optional[str] = None) -> Tuple[str, Optional[str], Dict]:
        """Validate messages using Ollama backend."""

        # Prepare messages for Guardian checking
        guardian_messages = [{"role": "system", "content": risk}]

        # For groundedness/context relevance, add document context
        if risk in ["groundedness", "context_relevance"] and context_text:
            guardian_messages.append({"role": "document", "content": context_text})

        guardian_messages.extend(messages)

        payload = {
            "model": self.model_version,
            "messages": guardian_messages,
            "stream": False,
            "think": thinking
        }

        # For function call validation, add tools to the payload
        if risk == "function_call" and tools:
            payload["tools"] = tools

        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            data = response.json()

            # Extract content and trace
            content = data.get("message", {}).get("content", "")
            trace = (
                data.get("message", {}).get("reasoning") or
                data.get("message", {}).get("thinking")
            )

            # Parse safety result
            label = _parse_safety_result(content, self._logger)

            return label, trace, data

        except Exception as e:
            self._logger.error(f"Ollama Guardian request failed: {e}")
            return "Failed", None, {"error": str(e)}


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
        """Initialize GuardianCheck with enhanced Granite Guardian 3.3 8B support.

        Args:
            risk: The risk type to check for. Can be GuardianRisk enum or custom string.
            backend_type: Backend to use - "huggingface" for local inference or "ollama" for Ollama.
            model_version: Model version to use. Defaults based on backend:
                - HuggingFace: "ibm-granite/granite-guardian-3.0-8b"
                - Ollama: "ibm/granite3.3-guardian:8b"
            device: Computational device ("cuda"/"mps"/"cpu"). Auto-selected if None.
            ollama_url: Ollama server URL for ollama backend.
            thinking: Enable thinking mode for detailed reasoning traces.
            custom_criteria: Custom risk criteria string (overrides standard risk types).
            context_text: Reference text for groundedness/context relevance checking.
            tools: Available tools for function call validation.
        """
        super().__init__(
            check_only=True, validation_fn=lambda c: self._guardian_validate(c)
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
        self._backend_type = backend_type
        self._context_text = context_text
        self._tools = tools

        # Set default model versions based on backend
        if model_version is None:
            if backend_type == "huggingface":
                model_version = "ibm-granite/granite-guardian-3.3-8b"
            else:  # ollama
                model_version = "ibm/granite3.3-guardian:8b"

        # Initialize backend
        if backend_type == "huggingface":
            self._backend = HuggingFaceGuardianBackend(model_version, device)
        elif backend_type == "ollama":
            self._backend = OllamaGuardianBackend(model_version, ollama_url)
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")

        self._logger = FancyLogger.get_logger()

    def get_effective_risk(self) -> str:
        """Get the effective risk criteria to use for validation."""
        return self._custom_criteria if self._custom_criteria else self._risk

    def supports_thinking_mode(self) -> bool:
        """Check if current backend supports thinking mode."""
        return True  # Both backends now support thinking mode

    @classmethod
    def get_available_risks(cls) -> List[str]:
        """Get list of all available standard risk types."""
        return GuardianRisk.get_available_risks()

    def _guardian_validate(self, ctx: Context) -> ValidationResult:
        """Enhanced validation using Granite Guardian 3.3 8B with thinking mode support.

        Args:
            ctx: The context object containing the conversation to validate.

        Returns:
            ValidationResult: Validation result with optional reasoning trace.
        """

        messages: List[Dict[str, str]] = []

        last_turn = ctx.last_turn()
        if last_turn is None:
            self._logger.warning("No last turn found in context")
            return ValidationResult(False, reason="No content to validate")

        # Extract messages from context
        if last_turn.model_input:
            user_msg = last_turn.model_input

            if isinstance(user_msg, CBlock) and user_msg.value is not None:
                messages.append({"role": "user", "content": user_msg.value})
            elif isinstance(user_msg, Message) and user_msg.content != "":
                messages.append({"role": user_msg.role, "content": user_msg.content})
            else:
                messages.append({"role": "user", "content": str(user_msg)})

        # Handle both text content and function calls
        if last_turn.output:
            assistant_content = ""

            # Add text content if available
            if last_turn.output.value:
                assistant_content = last_turn.output.value

            # Add function call information for FUNCTION_CALL risk validation
            if (hasattr(last_turn.output, 'tool_calls') and last_turn.output.tool_calls and
                self._risk == "function_call"):

                # Convert function calls to a text format that Guardian can validate
                function_calls_text = []
                for name, tool_call in last_turn.output.tool_calls.items():
                    call_info = f'{name}({tool_call.args})'
                    function_calls_text.append(call_info)

                function_calls_str = ', '.join(function_calls_text)

                if assistant_content:
                    assistant_content += f" [Function calls: {function_calls_str}]"
                else:
                    assistant_content = f"[Function calls: {function_calls_str}]"

            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})

        if not messages:
            self._logger.warning("No messages found to validate")
            return ValidationResult(False, reason="No messages to validate")

        # Use the appropriate risk criteria
        effective_risk = self.get_effective_risk()

        try:
            # Validate using the backend
            label, trace, raw_data = self._backend.validate_messages(
                messages, effective_risk, self._thinking, tools=self._tools, context_text=self._context_text
            )

            # Log the validation details
            self._logger.debug(f"Guardian validation - Risk: {effective_risk}, Result: {label}")
            if trace and self._thinking:
                self._logger.debug(f"Guardian reasoning: {trace}")

            # Determine validation result
            is_safe = label == "No"

            # Create detailed reason
            reason_parts = [f"Guardian check for '{effective_risk}': {label}"]

            if trace:
                reason_parts.append(f"Reasoning: {trace}")

            return ValidationResult(
                result=is_safe,
                reason="; ".join(reason_parts)
            )

        except Exception as e:
            self._logger.error(f"Guardian validation failed: {e}")
            return ValidationResult(False, reason=f"Validation error: {str(e)}")
