import pathlib
from typing import cast

import granite_common

from mellea.stdlib.base import CBlock, Component, TemplateRepresentation


class Intrinsic(Component):
    """A component for rewriting messages using intrinsics.

    Intrinsics are special components that transform a chat completion request.
    These transformations typically take the form of:
    - parameter changes (typically structured outputs)
    - adding new messages to the chat
    - editing existing messages
    (- adding documents? TODO: JAL investigate if this happens or if the docs are always user provided / already there...)
    """

    def __init__(
        self,
        intrinsic_name: str,
        config_file: str | pathlib.Path | None = None,
        config_dict: dict | None = None,
        base_model_name: str
        | None = None,  # TODO: JAL. allow this to take a model_id? if so, need to change equivalence logic elsewhere...
        model_name: str | None = None,
        intrinsic_kwargs: dict | None = None,
        # compatibility: list[AdapterType] = [AdapterType.ALORA, AdapterType.LORA] # TODO: JAL. Change this into an enum or literal list like roles...
    ) -> None:
        # TODO: JAL. Allow specifying an alora / lora directly as well...? instead of just by name...?
        #            if so, need to change matching logic...

        self.intrinsic_name = intrinsic_name

        # The base_model_name is used for looking up config files if needed.
        self.base_model_name = base_model_name

        # The model_name is the model actually used for the intrinsic.
        # Mellea currently only supports this being adapters. Typically don't need
        # to specify this.
        self.model_name = model_name

        if intrinsic_kwargs is None:
            intrinsic_kwargs = {}
        # TODO: JAL. Document this. This is what subclasses should use to pass needed
        #            kwargs to transform call.
        self.intrinsic_kwargs = intrinsic_kwargs

        # TODO: JAL. Potentially remove this config code here. Not necessarily a
        # real reason to do this here except for being able to do it once
        # and precaching special handling...

        # If any of the optional params are specified, attempt to set up the
        # config for the intrinsic here.
        config: dict | None = None
        if config_file is not None or config_dict is not None:
            config = granite_common.intrinsics.util.make_config_dict(
                config_file=config_file, config_dict=config_dict
            )
            config = cast(
                dict, config
            )  # Can remove if util function gets exported properly.

        if config is None and self.base_model_name is not None:
            io_yaml_file = granite_common.intrinsics.util.obtain_io_yaml(
                self.intrinsic_name, self.base_model_name
            )
            config = granite_common.intrinsics.util.make_config_dict(
                config_file=io_yaml_file
            )
            config = cast(
                dict, config
            )  # Can remove if util function gets exported properly.

        self.config: dict | None = config

    def parts(self) -> list[Component | CBlock]:
        """The set of all the constituent parts of the `Intrinsic`.

        Will need to be implemented by subclasses since not all intrinsics are output
        as text / messages. TODO: JAL.
        """
        raise NotImplementedError("parts isn't implemented by default")

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Formats the `Intrinsic` into a `TemplateRepresentation` or string.

        Returns: a `TemplateRepresentation` or string
        TODO: JAL. see if the base intrinsic should implement this.
                   ideally, this would be the intrinsic's instruction directive with the args populated
                   however, the rewriter's transform function handles that...
        """
        raise NotImplementedError("format_for_llm isn't implemented by default")

    def transform(self) -> None:
        ...
        # TODO: JAL. need to see if this function should be defined at the component level or if the backend will call this...
        #   there has to be some part of the intrinsic that defines the args / changes that take place; maybe that's done in granite common though...
        # The steps required here are different per backend... need to see where we want to define this...
