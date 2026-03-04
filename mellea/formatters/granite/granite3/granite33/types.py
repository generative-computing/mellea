# SPDX-License-Identifier: Apache-2.0

"""Dataclasses that are specific to the Granite 3.3 family of models."""

# Third Party
import pydantic

# First Party
from ...base.types import Document, VLLMExtraBody
from ...granite3.types import Granite3ChatCompletion


class Granite33ChatCompletion(Granite3ChatCompletion):
    """Class that represents the inputs to a Granite 3.3 model generation call."""

    @pydantic.field_validator("extra_body")
    @classmethod
    def _validate_documents(cls, extra_body: VLLMExtraBody) -> VLLMExtraBody:
        """Granite 3.3 documents must have document IDs."""
        if extra_body.documents:
            for i, d in enumerate(extra_body.documents):
                if not isinstance(d, Document):
                    raise TypeError(
                        f"Expected Document at position {i} but found "
                        f"{d} of type {type(d)}"
                    )
                if d.doc_id is None:
                    raise ValueError(
                        f"Document at position {i} lacks a `doc_id` "
                        f"field. This field is required for Granite "
                        f"3.3."
                    )
        return extra_body
