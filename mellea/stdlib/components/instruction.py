"""``Instruction`` component for instruct/validate/repair loops.

``Instruction`` is the primary component type used with ``MelleaSession.instruct``. It
packages a task ``description``, a list of ``Requirement`` constraints, optional
in-context-learning examples, a grounding context dict, user variables for Jinja2
template interpolation, and output/input prefix overrides into a single renderable
unit. The session's sampling strategy evaluates each requirement against the model's
output and may repair or resample until all requirements pass.
"""

from __future__ import annotations

from copy import deepcopy

import jinja2

from ...core import (
    CBlock,
    Component,
    ImageBlock,
    ModelOutputThunk,
    Requirement,
    TemplateRepresentation,
    blockify,
)
from ..requirements.requirement import reqify


class Instruction(Component[str]):
    """The Instruction in an instruct/validate/repair loop.

    Args:
        description (str | CBlock | None): The task description shown to the model.
        requirements (list[Requirement | str] | None): Constraints the output must satisfy.
        icl_examples (list[str | CBlock] | None): In-context-learning examples.
        grounding_context (dict[str, str | CBlock | Component] | None): Named context
            passages injected into the prompt.
        user_variables (dict[str, str] | None): Jinja2 variable substitutions applied
            to all string parameters.
        prefix (str | CBlock | None): A prefix prepended before the model's generation.
        output_prefix (str | CBlock | None): A prefix prepended to the model's output token
            stream (currently unsupported; must be ``None``).
        images (list[ImageBlock] | None): Images to include in the prompt.

    Attributes:
        requirements (list[Requirement]): The resolved list of requirement instances
            attached to this instruction.
    """

    def __init__(
        self,
        description: str | CBlock | None = None,
        requirements: list[Requirement | str] | None = None,
        icl_examples: list[str | CBlock] | None = None,
        grounding_context: dict[str, str | CBlock | Component] | None = None,
        user_variables: dict[str, str] | None = None,
        prefix: str | CBlock | None = None,
        output_prefix: str | CBlock | None = None,
        images: list[ImageBlock] | None = None,
    ):
        """Initialize Instruction, converting all string inputs to CBlocks and applying any Jinja2 variables."""
        requirements = [] if requirements is None else requirements
        icl_examples = [] if icl_examples is None else icl_examples
        grounding_context = dict() if grounding_context is None else grounding_context

        # Retain raw template before Jinja substitution for telemetry.
        # Template text is the static prompt structure; variables may contain user data.
        self._template_description: str | None = (
            description if isinstance(description, str) else None
        )
        self._user_variables: dict[str, str] | None = (
            dict(user_variables) if user_variables else None
        )

        # Apply templates. All inputs must be strings if provided.
        if user_variables is not None:
            if description is not None:
                assert type(description) is str, (
                    "description must be a string when user_variables are provided"
                )
                description = Instruction.apply_user_dict_from_jinja(
                    user_variables, description
                )

            if prefix is not None:
                assert type(prefix) is str, (
                    "prefix must be a string when user_variables are provided"
                )
                prefix = Instruction.apply_user_dict_from_jinja(user_variables, prefix)

            # The following code would have to be un-commented-out if the assertion between this line and the code block is removed.
            assert output_prefix is None, (
                "output_prefix is not currently supported. The output_prefix serves as a prefix for the assistant's next message, and can be useful for 'priming' the model toward the right sort of answer. However, doing this requires using 'raw' endpoints instead of chat endpoints. Support for output_prefix will be re-enabled when we switch to span-first backend design."
            )
            # if output_prefix is not None:
            #     assert (
            #         type(output_prefix) == str
            #     ), "output prefix must be a string when user_variables are provided"
            #     output_prefix = Instruction.apply_user_dict_from_jinja(
            #         user_variables, output_prefix
            #     )

            for i, req in enumerate(requirements):
                assert type(req) is str or isinstance(req, Requirement), (
                    "requirements must be strings or Requirements when user_variables are provided"
                )
                if type(req) is str:
                    requirements[i] = Instruction.apply_user_dict_from_jinja(
                        user_variables, req
                    )
                elif isinstance(req, Requirement):
                    r = deepcopy(req)
                    r.description = Instruction.apply_user_dict_from_jinja(
                        user_variables,
                        req.description,  # type: ignore
                    )
                    requirements[i] = r

            for i, ex in enumerate(icl_examples):
                assert type(ex) is str, (
                    "icl_examples must be strings when user_variables are provided"
                )
                icl_examples[i] = Instruction.apply_user_dict_from_jinja(
                    user_variables, ex
                )

            for key in grounding_context:
                g = grounding_context[key]
                assert type(g) is str, (
                    "documents must be strings when user_variables are provided"
                )
                grounding_context[key] = Instruction.apply_user_dict_from_jinja(
                    user_variables, g
                )  # type: ignore

        self._description = blockify(description) if description is not None else None
        self._requirements: list[Requirement] = [reqify(r) for r in requirements]
        self._icl_examples: list[CBlock | Component] = [
            blockify(e) for e in icl_examples
        ]

        # Map all string values to CBlocks in the grounding context.
        self._grounding_context: dict[str, CBlock | Component] = {
            k: blockify(v) if isinstance(v, str) else v
            for k, v in grounding_context.items()
        }
        self._prefix = blockify(prefix) if prefix is not None else None
        self._output_prefix = (
            blockify(output_prefix) if output_prefix is not None else None
        )
        self._images = images
        self._repair_string: str | None = None

    def parts(self):
        """Returns all of the constituent parts of an Instruction."""
        # Add all of the optionally defined CBlocks/Components then filter Nones at the end.
        cs = [self._description, self._prefix, self._output_prefix]
        match self._grounding_context:
            case CBlock():
                cs.append(self._grounding_context)
            case _:
                cs.extend(list(self._grounding_context.values()))
        cs.extend(self._requirements)
        cs.extend(self._icl_examples)

        filtered: list[Component | CBlock] = list(filter(lambda x: x is not None, cs))  # type: ignore
        return filtered

    def format_for_llm(self) -> TemplateRepresentation:
        """Format this instruction for the language model.

        Returns:
            TemplateRepresentation: A template representation containing the
            description, requirements, in-context examples, grounding context,
            and optional prefix/repair fields.
        """
        return TemplateRepresentation(
            obj=self,
            args={
                "description": str(self._description),
                "requirements": [
                    r.description
                    for r in self._requirements
                    if r.description is not None
                    and r.description != ""
                    and not r.check_only
                ],
                "icl_examples": [str(e) for e in self._icl_examples],
                "grounding_context": self._grounding_context,
                "prefix": self._prefix if self._prefix is not None else None,
                "output_prefix": (
                    self._output_prefix if self._output_prefix is not None else None
                ),
                "repair": self._repair_string,
            },
            tools=None,
            images=self._images,
            template_order=["*", "Instruction"],
        )

    def prompt_template_metadata(self) -> tuple[str, dict[str, str], None] | None:
        """Return prompt template metadata for telemetry.

        The raw template text is emitted unconditionally.  Variables are only
        emitted when content capture is enabled (they may contain user data).

        Returns:
            Tuple of ``(template_text, variables, version)`` when a string
            description was provided, otherwise ``None``.
        """
        if self._template_description is None:
            return None
        return self._template_description, dict(self._user_variables or {}), None

    @staticmethod
    def apply_user_dict_from_jinja(user_dict: dict[str, str], s: str) -> str:
        """Render a Jinja2 template string using the provided variable dictionary.

        Args:
            user_dict (dict[str, str]): Mapping of Jinja2 variable names to their
                string replacement values.
            s (str): A string treated as a Jinja2 template to be rendered.

        Returns:
            str: The rendered string with all Jinja2 placeholders substituted.
        """
        assert s is not None
        return jinja2.Template(s).render(user_dict)

    @property
    def requirements(self) -> list[Requirement]:
        """Returns a list of Requirement instances."""
        return self._requirements

    def copy_and_repair(self, repair_string: str) -> Instruction:
        """Create a deep copy of this instruction with the repair string set.

        Args:
            repair_string (str): The repair feedback string to attach, typically
                describing which requirements failed and why.

        Returns:
            Instruction: A new ``Instruction`` identical to this one but with
            ``_repair_string`` set to ``repair_string``.
        """
        res = deepcopy(self)
        res._repair_string = repair_string
        return res

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the model output. Returns string value for now."""
        return computed.value if computed.value is not None else ""
