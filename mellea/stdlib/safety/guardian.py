"""Risk checking with Guardian models."""

import json
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union, Literal

try:
    import requests
except ImportError:
    requests = None

from mellea.backends.ollama import OllamaModelBackend
from mellea.backends.huggingface import LocalHFBackend
from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib.base import CBlock, Context, Component, ModelOutputThunk
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
    if result is None:
        logger.warning("Guardian returned None result")
        return "Failed"

    result_lower = str(result).lower().strip()

    # Handle chat template artifacts - remove any remaining tags
    result_lower = result_lower.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()

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
    if any(word in result_lower for word in ["safe", "no risk", "acceptable", "not harmful"]):
        return "No"
    elif any(word in result_lower for word in ["unsafe", "risky", "harmful", "violation", "inappropriate", "offensive"]):
        return "Yes"

    # Special handling for refusal responses (model refusing to evaluate harmful content)
    if any(phrase in result_lower for phrase in ["can't engage", "won't engage", "don't engage", "inappropriate", "personal attack"]):
        return "Yes"  # If model refuses, likely because content is harmful

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

    def _get_result_sync(self, result_thunk) -> str:
        """Get result from ModelOutputThunk synchronously."""
        import asyncio

        try:
            # Try direct value access first (might be already resolved)
            if hasattr(result_thunk, 'value') and result_thunk.value is not None:
                return str(result_thunk.value)
        except Exception:
            pass

        try:
            # Try to get the underlying value directly
            if hasattr(result_thunk, '_underlying_value') and result_thunk._underlying_value:
                return str(result_thunk._underlying_value)
        except Exception:
            pass

        try:
            # If we have a generation task, wait for it
            if hasattr(result_thunk, '_generate') and result_thunk._generate:
                # Create new event loop if needed
                try:
                    loop = asyncio.get_running_loop()
                    # If we're in an async context, this is more complex
                    raise RuntimeError("In async context - need special handling")
                except RuntimeError:
                    # No running loop or we're in one - use asyncio.run
                    asyncio.run(result_thunk._generate)
                    return str(getattr(result_thunk, '_underlying_value', ""))
        except Exception as e:
            self._logger.warning(f"Async result handling failed: {e}")

        # Final fallback
        return str(result_thunk) if result_thunk else ""

    def _prepare_guardian_messages(self, messages: List[Dict[str, str]], risk: str,
                                  thinking: bool, context_text: Optional[str] = None,
                                  tools: Optional[List[Dict]] = None) -> List[Dict[str, str]]:
        """Prepare messages in Guardian format exactly like example script."""
        guardian_messages = []

        # System message contains ONLY the risk type (like example script)
        guardian_messages.append({"role": "system", "content": risk})

        # For groundedness, add document context as separate message (like example script)
        if risk == "groundedness" and context_text:
            guardian_messages.append({"role": "document 0", "content": context_text})

        # Add the original conversation messages exactly as provided
        guardian_messages.extend(messages)

        # NO additional instruction messages - Guardian model knows what to do
        # This matches the example script pattern exactly

        return guardian_messages


class HuggingFaceGuardianBackend(GuardianBackend):
    """HuggingFace-based Guardian backend that wraps LocalHFBackend."""

    def __init__(self, model_version: str = "ibm-granite/granite-guardian-3.3-8b", device: Optional[str] = None):
        super().__init__(model_version, device)

        # Create custom config if device is specified, otherwise let LocalHFBackend auto-detect
        custom_config = None
        if device is not None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_version)
            model = AutoModelForCausalLM.from_pretrained(
                model_version,
                torch_dtype=torch.bfloat16
            )
            torch_device = torch.device(device)
            model = model.to(torch_device)
            custom_config = (tokenizer, model, torch_device)

        # Wrap the existing LocalHFBackend
        self._hf_backend = LocalHFBackend(
            model_id=model_version,
            custom_config=custom_config
        )
        self._logger.info(f"Initialized HuggingFace Guardian backend with model: {model_version}")

    def validate_messages(self, messages: List[Dict[str, str]], risk: str,
                         thinking: bool = False, tools: Optional[List[Dict]] = None,
                         context_text: Optional[str] = None) -> Tuple[str, Optional[str], Dict]:
        """Validate messages using wrapped LocalHFBackend with event loop."""

        # Create async wrapper to handle event loop
        async def run_validation():
            # Prepare messages in Guardian format (like example script)
            guardian_messages = self._prepare_guardian_messages(messages, risk, thinking, context_text, tools)

            # Use the backend's native chat template capabilities
            from mellea.stdlib.base import LinearContext, ContextTurn

            ctx = LinearContext()

            # Add all Guardian messages to context
            for msg in guardian_messages:
                if msg["role"] in ["user", "assistant", "system"]:
                    ctx.insert_turn(ContextTurn(Message(msg["role"], msg["content"]), None))
                elif msg["role"].startswith("document"):
                    # Handle document messages for groundedness
                    ctx.insert_turn(ContextTurn(Message("user", f"Document: {msg['content']}"), None))

            # Prepare model options
            model_options = {
                "max_new_tokens": 2000 if thinking else 50,
                "do_sample": False,
                "temperature": 0.0,
                "system": risk  # System prompt is just the risk type
            }

            if thinking:
                model_options["think"] = True

            # Add an empty assistant message to trigger generation
            generation_prompt = Message("assistant", "")

            # Use native chat template generation
            if hasattr(self._hf_backend, 'generate_from_chat_context'):
                result_thunk = self._hf_backend.generate_from_chat_context(
                    generation_prompt, ctx, model_options=model_options
                )
            else:
                result_thunk = self._hf_backend.generate_from_context(
                    generation_prompt, ctx, model_options=model_options
                )

            # Wait for async result
            result_value = result_thunk.value
            # Handle None or empty results
            return str(result_value) if result_value is not None else ""

        # Run the async validation in a new event loop
        import asyncio
        try:
            full_response = asyncio.run(run_validation())
        except Exception as e:
            self._logger.error(f"HuggingFace validation failed: {e}")
            return "Failed", None, {"error": str(e)}

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
    """Ollama-based Guardian backend that wraps OllamaModelBackend."""

    def __init__(self, model_version: str = "ibm/granite3.3-guardian:8b",
                 ollama_url: str = "http://localhost:11434"):
        super().__init__(model_version)
        self.ollama_url = ollama_url

        # Wrap the existing OllamaModelBackend
        self._ollama_backend = OllamaModelBackend(
            model_id=model_version,
            base_url=ollama_url
        )
        self._logger.info(f"Initialized Ollama Guardian backend with model: {model_version}")

    def validate_messages(self, messages: List[Dict[str, str]], risk: str,
                         thinking: bool = False, tools: Optional[List[Dict]] = None,
                         context_text: Optional[str] = None) -> Tuple[str, Optional[str], Dict]:
        """Validate messages using wrapped OllamaModelBackend with event loop."""

        # Create async wrapper to handle event loop
        async def run_validation():
            # Prepare messages in Guardian format (like example script)
            guardian_messages = self._prepare_guardian_messages(messages, risk, thinking, context_text, tools)

            # Use the backend's native chat template capabilities
            from mellea.stdlib.base import LinearContext, ContextTurn

            ctx = LinearContext()

            # Add all Guardian messages to context
            for msg in guardian_messages:
                if msg["role"] in ["user", "assistant", "system"]:
                    ctx.insert_turn(ContextTurn(Message(msg["role"], msg["content"]), None))
                elif msg["role"].startswith("document"):
                    # Handle document messages for groundedness
                    ctx.insert_turn(ContextTurn(Message("user", f"Document: {msg['content']}"), None))

            # Prepare model options
            model_options = {
                "temperature": 0.0,
                "num_predict": 2000 if thinking else 50,
                "stream": False,
                "system": risk  # System prompt is just the risk type
            }

            if thinking:
                model_options["think"] = True

            # Add tools for function call validation
            if risk == "function_call" and tools:
                model_options["tools"] = self._convert_tools_to_functions(tools)

            # Add an empty assistant message to trigger generation
            generation_prompt = Message("assistant", "")

            # Use native chat template generation
            result_thunk = self._ollama_backend.generate_from_chat_context(
                generation_prompt, ctx, model_options=model_options
            )

            # Wait for async result
            result_value = result_thunk.value
            # Handle None or empty results
            return str(result_value) if result_value is not None else ""

        # Run the async validation in a new event loop
        import asyncio
        try:
            full_response = asyncio.run(run_validation())
        except Exception as e:
            self._logger.error(f"Ollama validation failed: {e}")
            return "Failed", None, {"error": str(e)}

        # Extract thinking trace if present
        trace = None
        result = full_response

        if thinking and "<think>" in str(full_response):
            parts = str(full_response).split("</think>")
            if len(parts) > 1:
                trace = parts[0].replace("<think>", "").strip()
                result = parts[1].strip()

        # Parse safety result
        label = _parse_safety_result(result, self._logger)

        return label, trace, {"full_response": full_response, "model": self.model_version}


    def _convert_tools_to_functions(self, tools: List[Dict]) -> List[callable]:
        """Convert tool definitions to callable functions for Ollama backend."""
        functions = []
        for tool in tools:
            # Create a dummy function that matches the tool signature
            def dummy_func(**kwargs):
                return f"Tool {tool['name']} called with args: {kwargs}"

            dummy_func.__name__ = tool['name']
            dummy_func.__doc__ = tool.get('description', '')
            functions.append(dummy_func)

        return functions


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
            ctx (LegacyContext): The context object containing the last turn of the conversation.

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
