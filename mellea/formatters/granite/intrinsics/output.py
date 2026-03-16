# SPDX-License-Identifier: Apache-2.0

"""Classes and functions that implement common aspects of output processing for intrinsics."""

# Standard
import abc
import collections
import copy
import enum
import json
import logging
import math
import pathlib
from typing import Any

# First Party
from ..base.io import ChatCompletionResultProcessor
from ..base.types import (
    ChatCompletion,
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    Document,
)

# Local
from . import json_util
from .input import sentence_delimiter
from .util import make_config_dict

_logger = logging.getLogger(__name__)


class _MappingType(enum.Enum):
    PASSTHRU = 1
    """Pass through the value from the model output using the type of the raw output
    schema"""


class TransformationRule(abc.ABC):
    """Base class for transformation rules to apply to JSON outputs of intrinsics.

    Args:
        config (dict): Configuration of the parent output processor, as parsed YAML.
        input_path_expr (list[str | int | None]): Path expression matching all
            instances of the field that this rule transforms. Elements can be
            strings for object fields, integers for list indices, or `None`
            for wildcard matches.

    Attributes:
        YAML_NAME (str | None): The name used to identify this rule in YAML
            configuration files. Subclasses must set this to a non-`None` string.
    """

    YAML_NAME: str | None = None
    """Subclasses should set this to the name of the rule in YAML config files."""

    def __init__(self, config: dict, input_path_expr: list[str | int | None]):
        """Initialize TransformationRule with a config dict and a path expression."""
        self.config = config
        self.input_path_expr = input_path_expr

    def _type_check(self, attr_name, attr_value, expected_type):
        """Type-check an argument from the YAML config file.

        Convenience method for subclasses to type-check an argument from the YAML
        config file.
        """
        if not isinstance(attr_value, expected_type):
            raise TypeError(
                f"'{attr_name}' parameter for '{self.rule_name()}' rule "
                f" must be of type {expected_type}."
                f"list. Received value {attr_value}"
            )

    def _is_input_path(self, path: tuple) -> bool:
        """:param path: JSON path as returned by :func:`scalar_paths()`

        :returns: True if this rule should be applied to the indicated path.
        """
        if len(path) != len(self.input_path_expr):
            return False
        for expr_elem, path_elem in zip(self.input_path_expr, path, strict=True):
            # None means "wildcard"
            if expr_elem is not None and path_elem != expr_elem:
                return False
        return True

    def _matching_paths(self, parsed_json: Any) -> list[tuple]:
        """Get paths that match this rule's input path spec.

        :param parsed_json: Output of running model results through
            :func:`json.loads()`, plus applying zero or more transformation rules.
        :returns: List of paths within `parsed_json` that match this rule's input
            path spec.
        """
        return [p for p in json_util.all_paths(parsed_json) if self._is_input_path(p)]

    def rule_name(self) -> str:
        """Return the YAML name that identifies this transformation rule.

        Returns:
            str: The value of `YAML_NAME` for this rule subclass.

        Raises:
            ValueError: If `YAML_NAME` has not been set by the subclass.
        """
        if self.YAML_NAME is None:
            raise ValueError(f"Attempted to fetch missing rule name for {type(self)}")
        return self.YAML_NAME

    # pylint: disable=unused-argument
    def _prepare(
        self,
        parsed_json: Any,
        reparsed_json: Any,
        logprobs: ChatCompletionLogProbs | None,
        chat_completion: ChatCompletion | None,
    ) -> dict:
        """Prepare data structures computed once per output.

        Subclasses may override this method to prepare data structures that should be
        computed once per output.

        :returns: Dict that will be passed to all calls to :func:`self._transform()`
        """
        return {}

    def apply(
        self,
        parsed_json: Any,
        reparsed_json: Any,
        logprobs: ChatCompletionLogProbs | None,
        chat_completion: ChatCompletion | None,
    ) -> Any:
        """Apply this transformation rule to the parsed model output.

        Args:
            parsed_json (Any): Output of running model results through
                :func:`json.loads()`, plus applying zero or more prior
                transformation rules.
            reparsed_json (Any): Output of running the same model results
                through :func:`json_util.reparse_json_with_offsets()`,
                preserving position information on literal values.
            logprobs (ChatCompletionLogProbs | None): Optional logprobs result
                associated with the original model output string, or `None`
                if no logprobs were present.
            chat_completion (ChatCompletion | None): The chat completion request
                that produced this output. Required by some rules.

        Returns:
            Any: Transformed copy of `parsed_json` after applying this rule.
        """
        paths = self._matching_paths(parsed_json)
        prepare_output = self._prepare(
            parsed_json, reparsed_json, logprobs, chat_completion
        )

        # print(f"{self.rule_name()}: Matching paths are {paths}")

        # If we get here, we need to modify a JSON object or array
        # Don't modify input
        result = copy.deepcopy(parsed_json)
        for path in paths:
            result = self._apply_at_path(result, path, prepare_output)
        return result

    @abc.abstractmethod
    def _apply_at_path(self, result: Any, path: tuple, prepare_output: dict) -> Any:
        """Apply this rule at a single JSON path.

        Subclasses should override this method to implement their transformation
         at a single location within the target JSON document.

        :param result: Parsed JSON representation of the transformed output at the
            current stage of transformation.  A copy of the original.
        :param path: Current path to transform locally
        :param prepare_output: Dictionary of global data that this object's
            :func:`self._prepare()` method has set aside

        :returns: A modified version of `result`, which may be modified in place or
            a fresh copy.
        """
        raise NotImplementedError()


class InPlaceTransformation(TransformationRule):
    """Base class for TransformationRules that replace values in place in JSON.

    Base class for `TransformationRule`s that replace values in place in the source
    JSON. The values replaced can be a scalar, object, or list.
    """

    def _apply_at_path(self, result: Any, path: tuple, prepare_output: dict) -> Any:
        original_value = json_util.fetch_path(result, path)
        transformed_value = self._transform(original_value, path, prepare_output)
        result = json_util.replace_path(result, path, transformed_value)
        return result

    @abc.abstractmethod
    def _transform(self, value: Any, path: tuple, prepare_output: dict) -> Any:
        """Transform a single scalar value from a larger JSON expression.

        Subclasses should override this method to transform a single scalar value
        from a larger JSON expression.

        :param value: Original value pulled out of the JSON expression, with position
            information attached to any embedded scalars.
        :param path: Location in the input JSON where the value was found
        :param prepare_output: Results from calling :func:`self._prepare()`

        :returns: New value for the indicated element of the JSON expression.
        """
        raise NotImplementedError()


class AddFieldsTransformation(TransformationRule):
    """Base class for TransformationRules that add values to JSON.

    Base class for `TransformationRule`s that add one or more values adjacent to
    an existing value in the source JSON.
    """

    def _apply_at_path(self, result: Any, path: tuple, prepare_output: dict) -> Any:
        """Subclasses should modify this.

        :param result: Parsed JSON representation of the transformed output at the
            current stage of transformation.  A copy of the original.
        :param path: Current path to transform locally
        :param prepare_output: Dictionary of global data that this object's
            :func:`self._prepare()` method has set aside

        :returns: A modified version of `result`, which may be modified in place or
            a fresh copy.
        """
        if len(path) == 0:
            raise ValueError(
                "Expected path to field of JSON object, but received zero-length path."
            )

        parent_path = path[:-1]
        parent_object = json_util.fetch_path(result, parent_path)
        if not isinstance(parent_object, dict):
            raise TypeError(
                f"Expected JSON object at path {parent_path} but found value of type "
                f"{type(parent_object)}"
            )

        original_value = parent_object[path[-1]]
        new_values = self._transform(original_value, path, prepare_output)

        # Make a copy, just in case.
        new_parent = parent_object.copy() | new_values
        result = json_util.replace_path(result, parent_path, new_parent)
        return result

    @abc.abstractmethod
    def _transform(self, value: Any, path: tuple, prepare_output: dict) -> dict:
        """Transform a single scalar value into new fields.

        Subclasses should override this method to transform a single scalar value
        from a larger JSON expression.

        :param value: Original value pulled out of the JSON expression, with position
            information attached to any embedded scalars.
        :param path: Location in the input JSON where the value was found
        :param prepare_output: Results from calling :func:`self._prepare()`

        :returns: Mapping from name of added field to value.
        """
        raise NotImplementedError()


##################################################
# Rule implementation classes start here


class TokenToFloat(InPlaceTransformation):
    """Transformation rule that decodes token logprobs to a floating point number.

    The floating point number replaces the original categorical value in the JSON.

    Args:
        config (dict): Configuration of the parent output processor, as parsed YAML.
        input_path_expr (list[str | int | None]): Path expression matching all
            instances of the field that this rule transforms.
        categories_to_values (dict[str | int | bool, float] | None): Mapping
            from categorical labels to floating-point values. Defaults to
            `None`.

    Attributes:
        YAML_NAME (str): YAML configuration key for this rule; always
            `"likelihood"`.
    """

    YAML_NAME = "likelihood"

    def __init__(
        self,
        config: dict,
        input_path_expr: list[str | int | None],
        /,
        categories_to_values: dict[str | int | bool, float] | None = None,
    ):
        """Initialize TokenToFloat with an optional mapping from categorical labels to float values."""
        super().__init__(config, input_path_expr)
        self.categories_to_values = categories_to_values

    def _prepare(
        self,
        parsed_json: Any,
        reparsed_json: Any,
        logprobs: ChatCompletionLogProbs | None,
        chat_completion: ChatCompletion | None,
    ) -> dict:
        if logprobs is not None and not isinstance(logprobs, ChatCompletionLogProbs):
            raise TypeError(
                f"Expected ChatCompletionLogProbs, but received {type(logprobs)}"
            )
        if logprobs is None:
            raise TypeError("This rule requires logprobs.  Received None for logprobs.")
        begin_to_token = json_util.make_begin_to_token_table(logprobs)

        return {
            "begin_to_token": begin_to_token,
            "reparsed_json": reparsed_json,
            "logprobs": logprobs,
        }

    def _transform(self, value: Any, path: tuple, prepare_output: dict) -> Any:
        # Retrieve values that are computed during self._prepare()
        begin_to_token = prepare_output["begin_to_token"]
        logprobs = prepare_output["logprobs"]
        reparsed_json = prepare_output["reparsed_json"]

        json_literal = json_util.fetch_path(reparsed_json, path)
        if not isinstance(json_literal, json_util.JsonLiteralWithPosition):
            raise TypeError(
                f"Expected literal with position, but received '{value}' "
                f"of type {type(value)}"
            )
        if value != json_literal.value:
            # Sanity check: Can't apply this rule on a path for which the tokens for
            # the original logprobs are no longer present.
            raise ValueError(
                f"At path {path}, reparsed value '{json_literal}' differs from "
                f"current value '{value}'. This rule cannot decode logprobs under this "
                f"circumstance."
            )

        value_str_offset = json_literal.begin
        if isinstance(json_literal.value, str):
            # Skip double quote at beginning of string literal
            value_str_offset += 1

        if begin_to_token is None or value_str_offset not in begin_to_token:
            # Categorical value didn't start on a token boundary. Check whether the
            # value was the suffix of a token.
            if begin_to_token is None:
                raise ValueError("begin_to_token is None but logprobs is not None")
            prev_token_offset = value_str_offset
            while prev_token_offset > 0 and prev_token_offset not in begin_to_token:
                prev_token_offset -= 1
            prev_token_ix = begin_to_token[prev_token_offset]
            prev_token = logprobs.content[prev_token_ix].token
            if prev_token.endswith(json.dumps(json_literal.value)):
                # Can't decode other tokens since we don't have logprobs for the
                # beginning of the literal.
                if self.categories_to_values is None:
                    raise ValueError("categories_to_values is None")
                return self.categories_to_values[json_literal.value]  # type: ignore[index]

            raise ValueError(
                f"Value '{json_literal}' starts at position "
                f"{value_str_offset}, "
                f"but there is no token at that position."
            )

        first_token_ix = begin_to_token[value_str_offset]
        values = []
        weights = []

        # Decode top token.
        # Assume that probability of first token == probability of entire literal
        if (
            self.categories_to_values is not None
            and json_literal.value in self.categories_to_values
        ):  # type: ignore[operator]
            values.append(self.categories_to_values[json_literal.value])  # type: ignore[index]
            weights.append(math.exp(logprobs.content[first_token_ix].logprob))

        # Decode remaining tokens.
        # Here we assume that the first category that shares a prefix with the token is
        # what the completion would have been had that token been the top-1.
        top_logprob: ChatCompletionLogProb
        for top_logprob in logprobs.content[first_token_ix].top_logprobs:
            if top_logprob.token == logprobs.content[first_token_ix].token:
                # Some inference engines will output the top-1 token both in logprobs
                # and in top_logprobs some of the time. Don't double-count when that
                # happens.
                continue
            if self.categories_to_values is not None:
                for category, value_for_category in self.categories_to_values.items():
                    if str(category).startswith(top_logprob.token):
                        # Use the first prefix match
                        values.append(value_for_category)
                        weights.append(math.exp(top_logprob.logprob))
                        break

        # Make the weights sum to 1 and return weighted sum, aka expected value
        if len(values) == 0:
            # No match --> default to 0
            return 0.0
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        return sum(w * v for w, v in zip(weights, values, strict=True))


def _desplit_sentences(
    target_text: str, tag: str, first_sentence_num: int
) -> dict[str, list]:
    """Undo sentence splitting performed during input processing.

    Subroutine of :class:`DecodeSentences` rule. Undoes the sentence splitting
    that we sometimes do during input processing.

    :param target_text: Text that has been split into sentences by inserting sentence
        boundary markers.
    :param tag: String such as that appears in every sentence boundary marker, e.g.
        "i" => "<i123>"
    :param first_sentence_num: Number we expect to see in the first sentence boundary
        marker in `target_text`.

    :returns: Self-describing dictionary of lists.
    """
    begins = []
    ends = []
    texts = []

    sentence_num = first_sentence_num

    # There is always at least one sentence if the text is encoded correctly.
    delimiter = sentence_delimiter(tag, sentence_num)
    delimiter_loc = target_text.find(delimiter, 0)
    if delimiter_loc == -1:
        raise ValueError(
            f"First sentence delimiter '{delimiter}' not found in "
            f"target string '{target_text}'"
        )
    begin = 0
    tagged_begin = delimiter_loc + len(delimiter)

    while True:
        # Loop invariant: We are looking for the end of sentence <sentence_num>.
        #   <begin> is positioned at the beginning of <sentence_num>,
        #   immediately after the delimiter for that sentence.

        # Delimiter string occurs at the BEGINNING of every sentence.
        # Check for delimiter of next sentence.
        delimiter = sentence_delimiter(tag, sentence_num + 1)
        delimiter_loc = target_text.find(delimiter, tagged_begin)
        if delimiter_loc == -1:
            # No more sentence markers
            begins.append(begin)
            # Begin + (remaining characters in tagged text)
            ends.append(begin + (len(target_text) - tagged_begin))
            texts.append(target_text[tagged_begin:])
            break
        begins.append(begin)
        begin += delimiter_loc - tagged_begin
        ends.append(begin)
        new_tagged_begin = delimiter_loc + len(delimiter)
        texts.append(target_text[tagged_begin:delimiter_loc])
        tagged_begin = new_tagged_begin
        sentence_num += 1

    return {"begins": begins, "ends": ends, "texts": texts}


class DecodeSentences(AddFieldsTransformation):
    """Transformation rule that decodes sentence refs into begin, end, text tuples.

    Args:
        config (dict): Configuration of the parent output processor, as parsed YAML.
        input_path_expr (list[str | int | None]): Path expression matching all
            instances of the field that this rule transforms.
        source (str): Name of the location to look for sentences; must be
            `"last_message"` or `"documents"`.
        output_names (dict): Mapping from output role name (`"begin"`,
            `"end"`, `"text"`, `"document_id"`) to the name of the new
            field to add in the result JSON.

    Attributes:
        YAML_NAME (str): YAML configuration key for this rule; always
            `"decode_sentences"`.
        begin_name (str | None): Name of the output field that receives the
            sentence begin offset; extracted from `output_names`, or `None`
            if not configured.
        end_name (str | None): Name of the output field that receives the
            sentence end offset; extracted from `output_names`, or `None`
            if not configured.
        text_name (str | None): Name of the output field that receives the
            sentence text; extracted from `output_names`, or `None` if not
            configured.
        document_id_name (str | None): Name of the output field that receives
            the document ID (only used when `source="documents"`); extracted
            from `output_names`, or `None` if not configured.
    """

    YAML_NAME = "decode_sentences"

    def __init__(
        self,
        config: dict,
        input_path_expr: list[str | int | None],
        /,
        source: str,
        output_names: dict,
    ):
        """Initialize DecodeSentences with a source location and output field name mapping.

        Raises:
            ValueError: If `source` is not `"last_message"` or
                `"documents"`, or if an unexpected key is found in
                `output_names`.
            TypeError: If `output_names` is not a dict.
        """
        super().__init__(config, input_path_expr)

        allowed_sources = ("last_message", "documents")
        if source not in allowed_sources:
            raise ValueError(
                f"'source' argument must be one of {allowed_sources}. "
                f"Received '{source}'"
            )
        self.source = source

        if not isinstance(output_names, dict):
            raise TypeError(
                f"Expected mapping for output_names, but received {output_names}"
            )
        for k in output_names:
            if source == "documents" and k == "document_id":
                continue
            if k not in ("begin", "end", "text"):
                raise ValueError(f"Unexpected key '{k}' in output_names")

        # Each of these attributes is set to None if not present in the YAML file
        self.begin_name = output_names.get("begin")
        self.end_name = output_names.get("end")
        self.text_name = output_names.get("text")
        self.document_id_name = output_names.get("document_id")

        if config["docs_as_message"] and config["docs_as_message"] not in [
            "json",
            "roles",
        ]:
            raise ValueError(
                f"Decoding sentences from message with document "
                f"encoding method '{config['docs_as_message']}' is "
                f"not yet supported."
            )

    def _prepare(
        self,
        parsed_json: Any,
        reparsed_json: Any,
        logprobs: ChatCompletionLogProbs | None,
        chat_completion: ChatCompletion | None,
    ) -> dict:
        if chat_completion is None:
            raise ValueError(
                f"No chat completion request object provided. "
                f"'{self.rule_name()}' rule requires this object."
            )

        if self.source == "documents":
            tag = self.config["sentence_boundaries"]["documents"]
            if tag is None:
                raise ValueError(
                    f"'{self.rule_name()}' attempting to decode document sentences, "
                    f"but 'sentence_boundaries' section of config file is missing "
                    f"the entry that tells how to tag document sentence boundaries."
                )

            documents: list[Document] = []
            if not self.config["docs_as_message"]:
                # Most common path: Documents from extra_body
                if chat_completion.extra_body is not None:
                    documents = chat_completion.extra_body.documents or []
            else:
                # Model requires documents in a user message. Decode the message.
                if self.config["docs_as_message"] == "json":
                    documents_json = json.loads(chat_completion.messages[0].content)
                    documents = [Document.model_validate(d) for d in documents_json]
                elif self.config["docs_as_message"] == "roles":
                    for message in chat_completion.messages:
                        if message.role.startswith("document "):
                            document = Document(
                                doc_id=message.role[len("document ") :],
                                text=message.content,
                            )
                            documents.append(document)
                else:
                    raise ValueError(
                        f"Unsupported doc type {self.config['docs_as_message']}"
                    )

            # De-split the sentences in each document in turn. Sentence numbers
            # start at zero on the first document and continue in subsequent documents.
            begins = []
            ends = []
            texts = []
            document_ids = []

            next_sentence_num = 0
            for d in documents:
                local_results = _desplit_sentences(d.text, tag, next_sentence_num)
                num_local_sentences = len(local_results["begins"])
                begins.extend(local_results["begins"])
                ends.extend(local_results["ends"])
                texts.extend(local_results["texts"])
                document_ids.extend([d.doc_id] * num_local_sentences)
                next_sentence_num += num_local_sentences

            return {
                "begins": begins,
                "ends": ends,
                "texts": texts,
                "document_ids": document_ids,
            }
        if self.source == "last_message":
            tag = self.config["sentence_boundaries"]["last_message"]
            if tag is None:
                raise ValueError(
                    f"'{self.rule_name()}' attempting to decode the last message, "
                    f"but 'sentence_boundaries' section of config file is missing "
                    f"the entry that tells how to tag message sentence boundaries."
                )

            # Use second-to-last turn if the input processing added an instruction turn
            message_ix = -2 if self.config["instruction"] else -1
            target_text = chat_completion.messages[message_ix].content

            return _desplit_sentences(target_text, tag, 0)

        raise ValueError(f"Unexpected source string '{self.source}'")

    def _transform(self, value: Any, path: tuple, prepare_output: dict) -> dict:
        # Unpack global values we set aside during the prepare phase
        begins = prepare_output["begins"]
        ends = prepare_output["ends"]
        texts = prepare_output["texts"]
        document_ids = prepare_output.get("document_ids")

        if not isinstance(value, int):
            raise TypeError(
                f"Expected integer sentence number at path {path}, but "
                f"found non-integer value {value} of type {type(value)}"
            )
        sentence_num = value

        result = {}
        if self.begin_name is not None:
            result[self.begin_name] = begins[sentence_num]
        if self.end_name is not None:
            result[self.end_name] = ends[sentence_num]
        if self.text_name is not None:
            result[self.text_name] = texts[sentence_num]
        if self.document_id_name is not None:
            result[self.document_id_name] = document_ids[sentence_num]  # type: ignore[index]
        return result


class Explode(InPlaceTransformation):
    """Expand list-valued attributes in a list of records.

    Turn each row in a list of records into zero or more rows by expanding
    the elements of a list-valued attribute.

    Attributes:
        YAML_NAME (str): YAML configuration key for this rule; always
            `"explode"`.
        target_field (str): Name of the list-valued field within each record
            to expand.
    """

    YAML_NAME = "explode"

    def __init__(self, config, input_path_expr, /, target_field):
        """Initialize Explode transformation rule.

        :param config: Parsed YAML config for IO processing
        :param input_path_expr: Path expression for the list of records to explode
        :param target: Name of list-valued field within each record.
        """
        super().__init__(config, input_path_expr)
        self.target_field = target_field

    def _transform(self, value, path, prepare_output):
        # Lots of error checking here because the input has no type constraints.
        if not isinstance(value, list):
            raise TypeError(
                f"Matching element at path {path} is not a list. "
                f"Matching paths of an '{self.rule_name()}' rule must be "
                f"lists."
            )
        result = []

        for i, element in enumerate(value):
            if not isinstance(element, dict):
                raise TypeError(
                    f"Element {i} of list at path {path} is not a record. "
                    f"'{self.rule_name()}' rule requires a list of records. "
                    f"Element is: {element}"
                )
            if self.target_field not in element:
                raise ValueError(
                    f"Element {i} of list at path {path} does not contain target field "
                    f"'{self.target_field}' for '{self.rule_name()}' rule. "
                    f"Element is: {element}"
                )
            to_explode = element[self.target_field]
            if not isinstance(to_explode, list):
                raise ValueError(
                    f"Target field '{self.target_field}' of element {i} of list at "
                    f"path {path} does not contain a list. "
                    f"'{self.rule_name()}' rule requires a list-valued attribute. "
                    f"Element is: {element}"
                )

            # Now that we've verified all the invariants, we can get down to the
            # business of constructing output records.
            for target_value in to_explode:
                # Use a single dictionary comprehension to ensure that field order
                # does not change.
                result.append(
                    {
                        k: target_value if k == self.target_field else v
                        for k, v in element.items()
                    }
                )
        return result


class DropDuplicates(InPlaceTransformation):
    """Remove duplicate records from a list of records.

    Attributes:
        YAML_NAME (str): YAML configuration key for this rule; always
            `"drop_duplicates"`.
        target_fields (list): Names of fields used to determine whether two
            records are considered duplicates.
    """

    YAML_NAME = "drop_duplicates"

    def __init__(self, config, input_path_expr, /, target_fields):
        """Initialize DropDuplicates transformation rule.

        :param config: Parsed YAML config for IO processing
        :param input_path_expr: Path expression for the list of records to explode
        :param target_fields: Names of fields to use for determining whether two records
            are duplicates.
        """
        super().__init__(config, input_path_expr)
        self.target_fields = target_fields

    def _transform(self, value, path, prepare_output):
        # Lots of error checking here because the input has no type constraints.
        if not isinstance(value, list):
            raise TypeError(
                f"Matching element at path {path} is not a list. "
                f"Matching paths of an '{self.rule_name()}' rule must be "
                f"lists."
            )

        # Build up a mapping from value of dedup key to last record with that key
        key_to_record = {}
        for i, element in enumerate(value):
            if not isinstance(element, dict):
                raise TypeError(
                    f"Element {i} of list at path {path} is not a record. "
                    f"'{self.rule_name()}' rule requires a list of records. "
                    f"Element is: {element}"
                )
            for t in self.target_fields:
                if t not in element:
                    raise TypeError(
                        f"Element {i} of list at path {path} is missing field '{t}'. "
                        f"'{self.rule_name()}' rule requires that the fields used for "
                        f"deduplication be present in all rows. "
                        f"Element is: {element}"
                    )
            key = tuple(element[t] for t in self.target_fields)
            key_to_record[key] = element
        return list(key_to_record.values())


class Project(InPlaceTransformation):
    """Project records down to a specified set of fields.

    Project records down to a specified set of fields.  Can also rename the
    retained fields.

    Attributes:
        YAML_NAME (str): YAML configuration key for this rule; always
            `"project"`.
        retained_fields (dict): Mapping from original field name to the
            (possibly renamed) output field name. Initialized from either a
            list of field names (identity mapping) or an explicit mapping.
    """

    YAML_NAME = "project"

    def __init__(self, config, input_path_expr, /, retained_fields):
        """Initialize Project transformation rule.

        :param config: Parsed YAML config for IO processing
        :param input_path_expr: Path expression for the list of records to explode
        :param retained_fields: Names of fields that remain after the projection. Can
            be either a list of fields or a mapping from retained field to new name of
            retained field.
        """
        super().__init__(config, input_path_expr)

        if not isinstance(retained_fields, list | dict):
            raise TypeError(
                f"'retained_fields' argument of '{self.rule_name()}' rule must be "
                f"either a list of fields or a mapping from retained field to new "
                f"name of retained field."
            )
        if isinstance(retained_fields, list):
            retained_fields = {f: f for f in retained_fields}
        self.retained_fields = retained_fields

    def _transform(self, value, path, prepare_output):
        # Lots of error checking here because the input has no type constraints.
        if isinstance(value, dict):
            # Single record
            return {self.retained_fields[k]: value.get(k) for k in self.retained_fields}
        if isinstance(value, list):
            return [
                {self.retained_fields[k]: element.get(k) for k in self.retained_fields}
                for element in value
            ]
        raise TypeError(
            f"Matching element at path {path} is not a list or a dictionary. "
            f"Matching paths of an '{self.rule_name()}' rule must contain "
            f"one or more records."
        )


class Nest(InPlaceTransformation):
    """Convert a value within a JSON structure into a record with a single field.

    Attributes:
        YAML_NAME (str): YAML configuration key for this rule; always
            `"nest"`.
        field_name (str): Name of the single field in the output JSON object
            that wraps each matching value.
    """

    YAML_NAME = "nest"

    def __init__(self, config, input_path_expr, /, field_name):
        """Initialize Nest transformation rule.

        :param config: Parsed YAML config for IO processing
        :param input_path_expr: Path expression for the values to nest
        :param field_name: name of the single field in the JSON object that this rule
            will wrap around each matching value.
        """
        super().__init__(config, input_path_expr)
        self._type_check("field_name", field_name, str)
        self.field_name = field_name

    def _transform(self, value, path, prepare_output):
        # Note that we don't check the type of the target value. This rule will happily
        # nest a collection of very different values. We may want to revisit this
        # design in the future. Or we might not.
        return {self.field_name: value}


class MergeSpans(InPlaceTransformation):
    """Merge adjacent spans into larger spans.

    Args:
        config: Parsed YAML config for IO processing.
        input_path_expr: Path expression for the list of records to merge.
        group_fields (list): List of fields used for grouping prior to merging
            spans.
        begin_field (str): Name of the field that holds the begin offset of
            spans.
        end_field (str): Name of the field that holds the end offset of spans.
        text_field (str | None): Optional field containing covered text strings
            that should be concatenated when spans are merged. Defaults to
            `None`.

    Attributes:
        YAML_NAME (str): YAML configuration key for this rule; always
            `"merge_spans"`.
    """

    YAML_NAME = "merge_spans"

    def __init__(
        self,
        config,
        input_path_expr,
        /,
        group_fields: list,
        begin_field: str,
        end_field: str,
        text_field: str | None = None,
    ):
        """Initialize MergeSpans with span field names and optional grouping fields."""
        super().__init__(config, input_path_expr)
        self._type_check("group_fields", group_fields, list)
        self._type_check("begin_field", begin_field, str)
        self._type_check("end_field", end_field, str)
        if text_field is not None:
            self._type_check("text_field", text_field, str)

        self.group_fields = group_fields
        self.begin_field = begin_field
        self.end_field = end_field
        self.text_field = text_field
        self.expected_fields = {*self.group_fields, self.begin_field, self.end_field}
        if self.text_field is not None:
            self.expected_fields.add(self.text_field)

    def _transform(self, value, path, prepare_output):
        # Lots of error checking here because the input has no type constraints.
        if not isinstance(value, list):
            raise TypeError(
                f"Matching element at path {path} is not a list. "
                f"Matching paths of an '{self.rule_name()}' rule must be "
                f"lists."
            )

        if len(value) == 0:
            # Save the code below from having to deal with this corner case
            return value

        # Extract keys for sorting and index the original records by sort key
        sort_key_to_records = collections.defaultdict(list)
        sort_keys = set()
        for i, record in enumerate(value):
            # Lots of checks here because input does not have a known schema
            if not isinstance(record, dict):
                raise TypeError(
                    f"Element {i} of list at path {path} is not a record. "
                    f"'{self.rule_name()}' rule requires a list of records. "
                    f"Element is: {record}"
                )
            if record.get(self.begin_field) is None:
                raise ValueError(
                    f"Record at position {i} of list at path {path} does not contain "
                    f"a begin offset in the '{self.begin_field}' field. "
                    f"'{self.rule_name()}' rule does not handle null spans. "
                    f"Record is: {record}"
                )
            if record.get(self.end_field) is None:
                raise ValueError(
                    f"Record at position {i} of list at path {path} does not contain "
                    f"an end offset in the '{self.end_field}' field. "
                    f"'{self.rule_name()}' rule does not handle null spans. "
                    f"Record is: {record}"
                )
            if self.text_field is not None and record.get(self.text_field) is None:
                raise ValueError(
                    f"Record at position {i} of list at path {path} does not contain "
                    f"span text field '{self.text_field}' field. "
                    f"'{self.rule_name()}' rule does not handle null spans. "
                    f"Record is: {record}"
                )
            for field in record:
                if field not in self.expected_fields:
                    raise ValueError(
                        f"Record at position {i} of list at path {path} contains "
                        f"unexpected field '{field}'. "
                        f"'{self.rule_name()}' rule does not support fields other than "
                        f"the grouping fields and the fields of the span. "
                        f"Record is: {record}"
                    )
            sort_key = tuple(
                record.get(f) for f in [*self.group_fields, self.begin_field]
            )
            sort_key_to_records[sort_key].append(record)
            sort_keys.add(sort_key)

        # Run through the sorted keys, merging spans
        result = []
        sorted_keys = sorted(sort_keys)
        key = sorted_keys[0]
        current_group = key[:-1]  # Last element is begin
        sorted_keys = sorted_keys[1:]
        records = sort_key_to_records[key]
        if len(records) != 1:
            raise ValueError(
                f"{len(records)} spans start at position "
                f"{records[0][self.begin_field]} of group {current_group}: {records}. "
                f"'{self.rule_name()}' rule only supports non-overlapping spans."
            )
        current_record = records[0].copy()  # Let's not modify the input, shall we?
        num_spans_merged = 1
        for key in sorted_keys:
            records = sort_key_to_records[key]
            if len(records) != 1:
                raise ValueError(
                    f"{len(records)} spans start at position "
                    f"{records[0][self.begin_field]} of group {current_group}: "
                    f"{records}. "
                    f"'{self.rule_name()}' rule only supports non-overlapping spans."
                )
            next_record = records[0]
            next_record_group = key[:-1]  # Last element is begin

            if (
                next_record_group != current_group
                or current_record[self.end_field] != next_record[self.begin_field]
            ):
                # Non-contiguous span.
                result.append(current_record)
                current_record = next_record.copy()
                current_group = next_record_group
                num_spans_merged = 1
            else:
                # Contiguous span. Extend current record.
                # Code above will have verified that only expected fields are present
                # and that begin, end, and text (if used) are always present.
                current_record[self.end_field] = next_record[self.end_field]
                if self.text_field is not None:
                    current_record[self.text_field] += next_record[self.text_field]
                num_spans_merged += 1

        # Don't forget the last output!
        result.append(current_record)
        return result


ALL_RULES = [
    # Try to keep these in alphabetical order, please
    DecodeSentences,
    DropDuplicates,
    Explode,
    MergeSpans,
    Nest,
    Project,
    TokenToFloat,
]
NAME_TO_RULE = {cls.YAML_NAME: cls for cls in ALL_RULES}  # type: ignore[attr-defined]

# END of rule implementations
############################################


_HARMONY_CHANNEL_TOKEN = "<|channel|>"
_HARMONY_MESSAGE_TOKEN = "<|message|>"
_HARMONY_END_TOKENS = frozenset({"<|end|>", "<|return|>", "<|end_of_text|>"})


def _find_final_channel_header(token_strings: list[str]) -> int | None:
    """Find the token index of the final `<|message|>` token in a token sequence.

    Find the token index of `<|message|>` that ends the last
    `<|channel|> final <|message|>` header in the token sequence.

    Matches are done on exact token values so that the single special token
    `<|channel|>` is never confused with regular tokens that happen to
    concatenate to the same string (e.g. `['<|', 'channel', '|>']`).

    :returns: Index of the `<|message|>` token, or `None`.
    """
    last_match = None
    i = 0
    while i < len(token_strings):
        if token_strings[i] == _HARMONY_CHANNEL_TOKEN:
            # Look ahead for "final" then <|message|>, skipping whitespace.
            j = i + 1
            while j < len(token_strings) and token_strings[j].strip() == "":
                j += 1
            if j < len(token_strings) and token_strings[j].strip() == "final":
                j += 1
                while j < len(token_strings) and token_strings[j].strip() == "":
                    j += 1
                if (
                    j < len(token_strings)
                    and token_strings[j] == _HARMONY_MESSAGE_TOKEN
                ):
                    last_match = j
                    i = j + 1
                    continue
        i += 1
    return last_match


def _logprobs_workaround(
    logprobs: ChatCompletionLogProbs,
) -> tuple[str, ChatCompletionLogProbs] | None:
    """Extract content and aligned logprobs from the final Harmony channel.

    Models using the `OpenAI Harmony response format
    <https://developers.openai.com/cookbook/articles/openai-harmony>`_ wrap
    output in channel tokens.  The final channel contains the user-facing
    payload::

        <|channel|> final <|message|> {payload} <|end|>

    This function walks the logprob token sequence, matching individual
    tokens (not concatenated strings) to locate the final channel header.
    This ensures that the single special token `<|channel|>` is never
    confused with regular tokens that concatenate to the same string.

    :param logprobs: Logprobs from a chat completion choice.
    :returns: `(content, trimmed_logprobs)` or `None` if no final channel
        is found.
    """
    if logprobs.content is None:
        return None

    token_strings = [lp.token for lp in logprobs.content]

    # Find the <|message|> token that ends the final channel header.
    message_ix = _find_final_channel_header(token_strings)
    if message_ix is None:
        return None

    # Payload starts at the token after <|message|>.
    start_ix = message_ix + 1
    if start_ix >= len(token_strings):
        return None

    # Find the end of the final channel by looking for an end token.
    end_ix = len(token_strings)
    for i in range(start_ix, len(token_strings)):
        if token_strings[i] in _HARMONY_END_TOKENS:
            end_ix = i
            break

    trimmed_tokens = logprobs.content[start_ix:end_ix]
    if not trimmed_tokens:
        return None

    content = "".join(lp.token for lp in trimmed_tokens)
    trimmed_logprobs = logprobs.model_copy(update={"content": trimmed_tokens})
    return content, trimmed_logprobs


class IntrinsicsResultProcessor(ChatCompletionResultProcessor):
    """General-purpose chat completion result processor for intrinsics.

    General-purpose chat completion result processor for use with the models that
    implement intrinsics. Reads parameters of the model's input and output formats
    from a YAML configuration file and edits the input chat completion appropriately.

    Attributes:
        config (dict): Parsed YAML configuration file for the target intrinsic.
        rules (list[TransformationRule]): Ordered list of transformation rules
            that this processor applies to each choice in a chat completion
            response.

    Args:
        config_file (str | pathlib.Path | None): Optional path to a YAML
            configuration file. Exactly one of `config_file` and
            `config_dict` must be provided.
        config_dict (dict | None): Optional pre-parsed YAML configuration dict.
            Exactly one of `config_file` and `config_dict` must be
            provided.
    """

    config: dict
    """Parsed YAML configuration file for the target intrinsic."""

    rules: list[TransformationRule]
    """Transformation rules that this object applies, in the order they are applied."""

    def __init__(
        self,
        /,
        config_file: str | pathlib.Path | None = None,
        config_dict: dict | None = None,
    ):
        """Initialize IntrinsicsResultProcessor from a YAML config file or dict."""
        config = make_config_dict(config_file, config_dict)
        if config is None:
            raise ValueError("config cannot be None")
        self.config = config

        # Set up transformation rules for the target model's JSON output
        self.rules = []
        if self.config["transformations"]:
            for transform_spec in self.config["transformations"]:
                if transform_spec["type"] not in NAME_TO_RULE:
                    raise ValueError(
                        f"Unknown transformation rule '{transform_spec['type']}'. "
                        f"Available rules are: {list(NAME_TO_RULE)}"
                    )
                rule_cls = NAME_TO_RULE[transform_spec["type"]]
                input_path = transform_spec["input_path"]
                rule_kwargs = {
                    k: v
                    for k, v in transform_spec.items()
                    if k not in ("type", "input_path")
                }
                self.rules.append(rule_cls(self.config, input_path, **rule_kwargs))

    # pylint: disable=unused-argument
    def _transform_impl(
        self,
        chat_completion_response: ChatCompletionResponse,
        chat_completion: ChatCompletion | None = None,
    ) -> ChatCompletionResponse:
        transformed_choices = [
            self._transform_choice(c, chat_completion)
            for c in chat_completion_response.choices
        ]
        return chat_completion_response.model_copy(
            update={"choices": transformed_choices}
        )

    def _transform_choice(
        self,
        choice: ChatCompletionResponseChoice,
        chat_completion: ChatCompletion | None,
    ) -> ChatCompletionResponseChoice:
        content = choice.message.content
        logprobs = choice.logprobs

        # When configured, use logprob token texts as the ground truth for
        # the model's output.  This handles cases where the inference server
        # modifies message.content (e.g. stripping or failing to strip
        # control tokens like gpt-oss channel tokens).
        if self.config.get("logprobs_workaround") and logprobs is not None:
            derived = _logprobs_workaround(logprobs)
            if derived is not None:
                derived_content, logprobs = derived
                if derived_content != content:
                    _logger.warning(
                        "logprobs_workaround: content derived from logprob "
                        "tokens differs from message.content. Using logprob "
                        "content as ground truth.\n"
                        "  message.content: %r\n"
                        "  logprob content: %r",
                        content,
                        derived_content,
                    )
                content = derived_content

        # Parse JSON output twice: Once to verify valid JSON and once to compute offsets
        # Note that we don't currently check schema, as that would require an additional
        # library dependency.
        parsed_json = json.loads(content)
        reparsed_json = json_util.reparse_json_with_offsets(content)
        for rule in self.rules:
            parsed_json = rule.apply(
                parsed_json, reparsed_json, logprobs, chat_completion
            )
        updated_message = choice.message.model_copy(
            update={"content": json.dumps(parsed_json)}
        )

        result = choice.model_copy(update={"message": updated_message})

        # Drop logprobs, since they should only be used by this function, and the tokens
        # referenced will no longer match the processed JSON value anyhow.
        # We may need to make this dropping configurable in the future.
        # Ok to modify in place because updated_message is a deep copy.
        if result.logprobs is not None:
            # Don't set the logprobs to None, unset it. There is a distinction in
            # Pydantic between these two states.
            del result.logprobs

        return result
