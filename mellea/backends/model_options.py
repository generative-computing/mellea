"""Common ModelOptions for Backend Generation."""

from typing import Any

from ..core import MelleaLogger


class ModelOption:
    """A type that wraps around model options.

    Uses sentinel values (wrapped by @@@) to provide backend and model-agnostic keys for common model options.

    Create a dictionary containing model options like this:

    ```python
    from mellea.backends import ModelOption
    model_options = {
        ModelOption.TEMPERATURE : 0.0,
        ModelOption.SYSTEM_PROMPT : "You are a helpful assistant"
    }
    ```

    Attributes:
        TOOLS (str): Sentinel key for a list or dict of `MelleaTool` instances to expose for tool calling.
        TOOL_CHOICE (str): Key for tool choice strategy (passed through to the backend).
        MAX_NEW_TOKENS (str): Sentinel key for the maximum number of new tokens to generate.
        SYSTEM_PROMPT (str): Sentinel key for the system prompt string.
        TEMPERATURE (str): Key for the sampling temperature (passed through to the backend).
        CONTEXT_WINDOW (str): Sentinel key for the context window size.
        THINKING (str): Sentinel key for enabling/configuring reasoning/thinking mode.
        SEED (str): Sentinel key for the random seed for reproducible generation.
        STREAM (str): Sentinel key for enabling streaming responses.
        STREAM_TIMEOUT (str): Sentinel key for the per-chunk streaming timeout in seconds
            (applied to every chunk, including time-to-first-token).
            ``None`` disables the timeout. Defaults to ``120.0`` when not set.
        STOP_SEQUENCES (str): Sentinel key for a `list[str]` of strings that, when
            encountered in the model output, cause generation to halt.
        LOGITS (str): Sentinel key for requesting per-token processed logit scores (post-LogitsProcessor).
        RAW_LOGITS (str): Sentinel key for requesting per-token raw LM-head logits (pre-LogitsProcessor).
    """

    TOOLS = "@@@tools@@@"
    """Must be a list[MelleaTool] or a dict[str, MelleaTool]. Use `MelleaTool.from_callable()` or the `@tool` decorator to wrap plain callables."""

    TOOL_CHOICE = "tool_choice"
    """Controls which tool the model should use. Can be "none", "auto", or a specific tool name."""

    MAX_NEW_TOKENS = "@@@max_new_tokens@@@"
    SYSTEM_PROMPT = "@@@system_prompt@@@"
    TEMPERATURE = "temperature"
    CONTEXT_WINDOW = "@@@context_window@@@"
    THINKING = "@@@thinking@@@"
    """Enable or configure reasoning/thinking mode.

    Accepted values:

    * ``True`` — sets ``chat_template_kwargs.enable_thinking=True`` (vLLM,
      Ollama OpenAI-compat) **and** ``reasoning_effort="medium"`` (OpenAI
      o-series, DeepSeek). Use this to engage reasoning on vLLM-served models
      such as Qwen3, Gemma 4, and GLM-4.5 — ``"medium"`` alone is silently
      ignored by those servers.
    * ``False`` — sets ``chat_template_kwargs.enable_thinking=False`` to
      suppress the think block. ``reasoning_effort`` is not sent (passing
      ``False`` would be an invalid value for OpenAI).
    * ``"low"`` / ``"medium"`` / ``"high"`` — passed directly as
      ``reasoning_effort`` (OpenAI/DeepSeek only; no-op on vLLM).
    """
    SEED = "@@@seed@@@"
    STREAM = "@@@stream@@@"
    STREAM_TIMEOUT = "@@@stream_timeout@@@"
    """Per-chunk timeout in seconds (applied to every chunk, including time-to-first-token).
    If no chunk arrives within this window the stream is aborted with ``TimeoutError``.
    ``None`` disables the timeout. Defaults to ``120.0`` seconds when not set."""
    STOP_SEQUENCES = "@@@stop_sequences@@@"
    """Must be a `list[str]`. Generation halts when the model emits any of these strings.

    Backends translate this to their native parameter (`stop` for OpenAI/Ollama/LiteLLM
    chat backends, `stop_strings` for Hugging Face, `stop_sequences` for the WatsonX
    text-generation endpoint).
    """
    LOGITS = "@@@logits@@@"
    """When `True`, request per-token processed logit scores (``output_scores``).

    These are logits *after* the ``LogitsProcessor`` chain has run —
    temperature scaling, top-k/top-p masking, repetition penalty, etc.
    Exposed on ``mot.generation.logits`` as a tuple of 1-D tensors of shape
    ``(vocab_size,)``, one per generated token.

    Only supported by the HuggingFace local backend. Backends that cannot
    return logits (OpenAI, Ollama, LiteLLM, WatsonX) log a warning when this
    option is set and leave ``generation.logits`` as ``None``.

    **Streaming not supported**: when ``ModelOption.STREAM=True``, logit
    scores are not available and ``mot.generation.logits`` will be ``None``.

    See also ``ModelOption.RAW_LOGITS`` for unprocessed LM-head output.
    """

    RAW_LOGITS = "@@@raw_logits@@@"
    """When `True`, request per-token raw logits (``output_logits``).

    These are the raw, unprocessed logits straight from the LM head —
    the model's actual output *before* any ``LogitsProcessor`` transforms
    (temperature, top-k/top-p, repetition penalty, etc.).
    Exposed on ``mot.generation.raw_logits`` as a tuple of 1-D tensors of
    shape ``(vocab_size,)``, one per generated token.

    Only supported by the HuggingFace local backend.

    **Streaming not supported**: when ``ModelOption.STREAM=True``, raw
    logits are not available and ``mot.generation.raw_logits`` will be ``None``.

    See also ``ModelOption.LOGITS`` for processor-transformed scores.
    """

    @staticmethod
    def replace_keys(options: dict, from_to: dict[str, str]) -> dict[str, Any]:
        """Return a new dict with selected keys in `options` renamed according to `from_to`.

        Returns a new dict with the keys in `options` replaced with the corresponding value for that key in `from_to`.

        * Any key with value == None is treated the same as the key missing.

        * If the destination key already exists in `options`, the original value is kept in the output.

        * Regardless of the presence of the destination key in `options`,
          the source key is always absent in the output.

        Example:
        ```python
        >>> options = {"k1": "v1", "k2": "v2", "M1": "m1"}
        >>> from_to = {"k1": "M1", "k2": "M2"}

        >>> new_options = replace_keys(options, from_to)
        >>> print(new_options)
        ... {"M1": "m1", "M2": "v2"}
        ```

        * Notice that "M1" keeps the original value "m1", rather than "v1".
        * Notice that both "k1" and "k2" are absent in the output.

        Args:
            options (dict): The source dictionary whose keys may be renamed.
            from_to (dict[str, str]): Mapping of old key names to new key names.

        Returns:
            dict[str, Any]: A new dictionary with the specified keys renamed.
        """
        new_options = {}

        # Because a model_options dictionary could have both the backend specific
        # and mellea @@@ version of the same key; we have extra logging here.
        conflict_log: list[str] = []

        # Copy over all the keys.
        for k, v in options.items():
            new_options[k] = v

        # Replace the keys in new_options with those specified in from_to.
        new_options_keys = list(new_options.keys())
        for old_key in new_options_keys:
            # This will usually be a @@@<>@@@ ModelOption.<> key.
            new_key = from_to.get(old_key, None)
            if new_key:
                # Skip if old_key and new_key are the same (no-op replacement)
                if old_key == new_key:
                    continue

                if new_options.get(new_key, None) is not None:
                    # The key already has a value associated with it in the dict. Leave it be.
                    conflict_log.append(
                        f"- old_key ({old_key}) to new_key ({new_key}): lost value associated with old_key ({new_options[old_key]}) and kept original value of new_key ({new_options[new_key]})"
                    )
                else:
                    new_options[new_key] = new_options[old_key]

                # Always clean up the old_keys that had mappings.
                del new_options[old_key]

        if len(conflict_log) > 0:
            text_line = (
                "Encountered conflict(s) when replacing keys. Could not replace keys for:\n"
                + "\n".join(conflict_log)
            )
            MelleaLogger.get_logger().warning(f"{text_line}")
        return new_options

    @staticmethod
    def remove_special_keys(model_options: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of `model_options` with all sentinel-valued keys removed.

        Sentinel keys are those whose names start with `@@@` (e.g. `ModelOption.TOOLS`).
        These are Mellea-internal keys that must not be forwarded to backend APIs.

        Args:
            model_options (dict[str, Any]): A model options dictionary that may contain sentinel keys.

        Returns:
            dict[str, Any]: A new dictionary with all `@@@`-prefixed keys omitted.
        """
        new_options = {}
        for k, v in model_options.items():
            if not k.startswith("@@@"):
                new_options[k] = v
        return new_options

    @staticmethod
    def merge_model_options(
        persistent_opts: dict[str, Any], overwrite_opts: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Merge two model-options dicts, with `overwrite_opts` taking precedence on conflicts.

        Creates a new dict that contains all keys and values from persistent opts and overwrite opts.
        If there are duplicate keys, overwrite opts key value pairs will be used.

        Args:
            persistent_opts (dict[str, Any]): Base model options (lower precedence).
            overwrite_opts (dict[str, Any] | None): Per-call model options that override
                `persistent_opts` on key conflicts; `None` is treated as empty.

        Returns:
            dict[str, Any]: A new merged dictionary.
        """
        new_options = {}

        for k, v in persistent_opts.items():
            new_options[k] = v

        if overwrite_opts is not None:
            for k, v in overwrite_opts.items():
                new_options[k] = v
        return new_options
