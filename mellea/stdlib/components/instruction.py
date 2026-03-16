"""Instructions."""

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
    """The Instruction in an instruct/validate/repair loop."""

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
        """Initializes an instruction. All strings will be converted into CBlocks.

        Args:
            description (str): The description of the instruction.
            requirements (List[Requirement | str]): A list of requirements that the instruction can be validated against.
            icl_examples (List[str | CBlock]): A list of in-context-learning examples that the instruction can be validated against.
            grounding_context (Dict[str, str | CBlock | Component]): A list of grounding contexts that the instruction can use. They can bind as variables using a (key: str, value: str | ContentBlock) tuple.
            user_variables (Dict[str, str]): A dict of user-defined variables used to fill in Jinja placeholders in other parameters. This requires that all other provided parameters are provided as strings.
            prefix (Optional[str | CBlock]): A prefix string or ContentBlock to use when generating the instruction.
            output_prefix (Optional[str | CBlock]): A string or ContentBlock that defines a prefix for the output generation. Usually you do not need this.
            images (Optional[List[ImageCBlock]]): A list of images to use in the instruction.
        """
        requirements = [] if requirements is None else requirements
        icl_examples = [] if icl_examples is None else icl_examples
        grounding_context = dict() if grounding_context is None else grounding_context
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
        """Formats the instruction for Formatter use."""
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

    @staticmethod
    def apply_user_dict_from_jinja(user_dict: dict[str, str], s: str) -> str:
        """Treats s as a jinja string and user_dict as the template values dictionary."""
        assert s is not None
        return jinja2.Template(s).render(user_dict)

    @property
    def requirements(self) -> list[Requirement]:
        """Returns a list of Requirement instances."""
        return self._requirements

    def copy_and_repair(self, repair_string: str) -> Instruction:
        """Creates a copy of the instruction and adds/overwrites the repair string."""
        res = deepcopy(self)
        res._repair_string = repair_string
        return res

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the model output. Returns string value for now."""
        return computed.value if computed.value is not None else ""
