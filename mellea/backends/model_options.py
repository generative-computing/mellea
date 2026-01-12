"""Common ModelOptions for Backend Generation."""

from typing import Any

from ..core import FancyLogger


class ModelOption:
    """A type that wraps around model options.

    Uses sentinel values (wrapped by @@@) to provide backend and model-agnostic keys for common model options.

    Create a dictionary containing model options like this:

    ```python
    from mellea.backends.types import ModelOption
    model_options = {
        ModelOption.TEMPERATURE : 0.0,
        ModelOption.SYSTEM_PROMPT : "You are a helpful assistant"
    }
    ```
    """

    TOOLS = "@@@tools@@@"
    """Must be a list[Callable] or a dict[str, Callable] where str is the name of the function."""

    MAX_NEW_TOKENS = "@@@max_new_tokens@@@"
    SYSTEM_PROMPT = "@@@system_prompt@@@"
    TEMPERATURE = "temperature"
    CONTEXT_WINDOW = "@@@context_window@@@"
    THINKING = "@@@thinking@@@"
    SEED = "@@@seed@@@"
    STREAM = "@@@stream@@@"

    @staticmethod
    def replace_keys(options: dict, from_to: dict[str, str]) -> dict[str, Any]:
        """Returns a new dict with the keys in `options` replaced with the corresponding value for that key in `from_to`.

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
            FancyLogger.get_logger().warning(f"{text_line}")
        return new_options

    @staticmethod
    def remove_special_keys(model_options) -> dict[str, Any]:
        """Removes all sentiel-valued keys (i.e., those that start with @@@)."""
        new_options = {}
        for k, v in model_options.items():
            if not k.startswith("@@@"):
                new_options[k] = v
        return new_options

    @staticmethod
    def merge_model_options(
        persistent_opts: dict[str, Any], overwrite_opts: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Creates a new dict that contains all keys and values from persistent opts and overwrite opts. If there are duplicate keys, overwrite opts key value pairs will be used."""
        new_options = {}

        for k, v in persistent_opts.items():
            new_options[k] = v

        if overwrite_opts is not None:
            for k, v in overwrite_opts.items():
                new_options[k] = v
        return new_options
