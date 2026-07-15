# SPDX-License-Identifier: Apache-2.0

"""JSON parsing utilities for Granite intrinsic formatters.

Provides a fast, position-aware JSON literal parser (`JsonLiteralWithPosition`) used
to extract and re-score tokens inside structured model outputs. The module also defines
compiled regular expressions for JSON structural characters, numbers, booleans, and
null values that are used throughout the Granite intrinsic formatting pipeline.
"""

# Standard
import bisect
import copy
import json
import re
from typing import Any

# Third Party
import pydantic

# First Party
from ..base.types import ChatCompletionLogProbs

##########################
# CONSTANTS

# Regexes for non-string JSON tokens that contain literal values.
# You shouldn't use regexes to parse JSON unless you know what you're doing.
# Fortunately we know what we're doing.
DELIM_REGEX = re.compile(r"[\{\}\[\]\,\:]")
NUMBER_REGEX = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")
BOOL_REGEX = re.compile(r"true|false")
NULL_REGEX = re.compile(r"null")


##########################
# CLASSES


class JsonLiteralWithPosition(pydantic.BaseModel):
    """JSON literal value with its position in the source string.

    Attributes:
        value (str | bool | int | float): The parsed Python value of the JSON
            literal (string, boolean, integer, or float).
        begin (int): Start offset (inclusive) of the literal within the source
            JSON string.
        end (int): End offset (exclusive) of the literal within the source
            JSON string.
    """

    value: str | bool | int | float
    begin: int
    end: int


##########################
# FUNCTIONS


def find_string_offsets(json_data: str) -> list[tuple[int, int, str]]:
    """Find the offsets of all strings in valid JSON data.

    Find the offsets of all strings in the input, assuming that this input
    contains valid JSON.

    Args:
        json_data: String containing valid JSON.

    Returns:
        Begin and end offsets of all strings in `json_data`, including
        the double quotes.
    """
    result = []
    position = 0
    while position < len(json_data):
        if json_data[position] == '"':
            begin = position
            decoded_str, end = json.decoder.scanstring(json_data, position + 1)  # type: ignore[attr-defined]
            result.append((begin, end, decoded_str))
            position = end
        position += 1
    return result


def non_string_offsets(json_str, compiled_regex, string_begins, string_ends):
    """Identify all matches of a regex that are not within string literals.

    Args:
        json_str: Original string of valid JSON data.
        compiled_regex: Compiled regex for the target token type.
        string_begins: Table of string begin offsets within `json_str`.
        string_ends: Table of string end offsets within `json_str`.

    Returns:
        List of `(begin, end, matched_string)` tuples.
    """
    offsets = []
    for match in compiled_regex.finditer(json_str):
        begin, end = match.span()
        delim_str = match.group()
        str_index = bisect.bisect(string_begins, match.span()[0]) - 1
        is_in_string = (
            str_index > 0
            and begin >= string_begins[str_index]
            and end < string_ends[str_index]
        )
        if not is_in_string:
            offsets.append((begin, end, delim_str))
    return offsets


def tokenize_json(json_str: str):
    """Lexer for parsing JSON.

    Args:
        json_str: String representation of valid JSON data.

    Returns:
        List of tuples of `(begin, end, value, type)`.
    """
    string_offsets = find_string_offsets(json_str)
    string_begins = [s[0] for s in string_offsets]
    string_ends = [s[1] for s in string_offsets]

    delim_offsets = non_string_offsets(
        json_str, DELIM_REGEX, string_begins, string_ends
    )
    number_offsets = non_string_offsets(
        json_str, NUMBER_REGEX, string_begins, string_ends
    )
    bool_offsets = non_string_offsets(json_str, BOOL_REGEX, string_begins, string_ends)
    null_offsets = non_string_offsets(json_str, NULL_REGEX, string_begins, string_ends)

    # n-way merge
    all_offsets = sorted(
        [(*t, "delim") for t in delim_offsets]
        + [(*t, "number") for t in number_offsets]
        + [(*t, "bool") for t in bool_offsets]
        + [(*t, "null") for t in null_offsets]
        + [(*t, "string") for t in string_offsets]
    )
    return all_offsets


def reparse_value(tokens, offset) -> tuple[Any, int]:
    """Parse JSON with offset generation using recursive-descent.

    Main entry point for a recursive-descent JSON parser with offset generation.
    Assumes valid JSON.

    Args:
        tokens: Token stream as produced by `tokenize_json()`.
        offset: Token offset at which to start parsing.

    Returns:
        Tuple of `(parsed_value, next_offset)`.

    Raises:
        ValueError: If an unexpected delimiter token or unknown token type is
            encountered at the current offset.
    """
    begin, end, value, type_ = tokens[offset]
    if type_ == "delim":
        if value == "{":
            return reparse_object(tokens, offset + 1)
        if value == "[":
            return reparse_list(tokens, offset + 1)
        raise ValueError(f"Unexpected token '{value}' found at {begin}")
    if type_ == "string":
        return JsonLiteralWithPosition(value=value, begin=begin, end=end), offset + 1
    if type_ in ("number", "bool", "null"):
        return JsonLiteralWithPosition(
            value=json.loads(value), begin=begin, end=end
        ), offset + 1
    raise ValueError(f"Unexpected type string {type_}")


def reparse_object(tokens, offset) -> tuple[dict, int]:
    """Parse a JSON object from the token stream, starting after the opening `{`.

    Subroutine called by :func:`reparse_value` when an opening curly brace is
    encountered. Consumes tokens until the matching closing `}` is found.

    Args:
        tokens: Token stream as produced by `tokenize_json()`.
        offset (int): Token offset immediately after the opening `{` delimiter.

    Returns:
        tuple[dict, int]: A tuple of `(parsed_dict, next_offset)` where
            `parsed_dict` maps string keys to parsed values (possibly
            :class:`JsonLiteralWithPosition` instances) and `next_offset`
            is the position of the next unconsumed token.

    Raises:
        ValueError: If the token stream does not conform to valid JSON object
            syntax (e.g. missing colon, unexpected delimiter, or non-string key).
    """
    result: dict[Any, Any] = {}
    while True:
        begin, _, value, type_ = tokens[offset]
        if type_ == "delim" and value == "}":
            # Closing curly brace
            return result, offset + 1

        # Attempt to consume a key: value pair
        # Key part
        if type_ != "string":
            raise ValueError(f"Expected string at {begin} but found '{value}'")
        next_key = value
        offset += 1
        begin, _, value, type_ = tokens[offset]

        # Colon part
        if type_ != "delim" or value != ":":
            raise ValueError(f"Expected ':' at {begin} but found '{value}'")
        offset += 1

        # Value part
        next_value, offset = reparse_value(tokens, offset)
        result[next_key] = next_value
        begin, _, value, type_ = tokens[offset]

        # Comma or closing curly brace
        if type_ != "delim":
            raise ValueError(f"Expected delimiter at {begin} but found '{value}'")
        if value == ",":
            offset += 1
        elif value != "}":
            raise ValueError(f"Expected comma or '}}' at {begin} but found '{value}'")


def reparse_list(tokens, offset) -> tuple[list, int]:
    """Parse a JSON array from the token stream, starting after the opening `[`.

    Subroutine called by :func:`reparse_value` when an opening square bracket is
    encountered. Consumes tokens until the matching closing `]` is found.

    Args:
        tokens: Token stream as produced by `tokenize_json()`.
        offset (int): Token offset immediately after the opening `[` delimiter.

    Returns:
        tuple[list, int]: A tuple of `(parsed_list, next_offset)` where
            `parsed_list` contains the parsed elements (possibly
            :class:`JsonLiteralWithPosition` instances) and `next_offset`
            is the position of the next unconsumed token.

    Raises:
        ValueError: If the token stream does not conform to valid JSON array
            syntax (e.g. unexpected delimiter between elements).
    """
    result: list[Any] = []
    while True:
        begin, _, value, type_ = tokens[offset]
        if type_ == "delim" and value == "]":
            # Closing square bracket
            return result, offset + 1

        # Attempt to consume a list element
        next_value, offset = reparse_value(tokens, offset)
        result.append(next_value)
        begin, _, value, type_ = tokens[offset]

        # Optional comma
        if type_ != "delim":
            raise ValueError(f"Expected delimiter at {begin} but found '{value}'")
        if value == ",":
            offset += 1


def reparse_json_with_offsets(json_str: str) -> Any:
    """Reparse a JSON string to compute the offsets of all literals.

    Args:
        json_str: String known to contain valid JSON data.

    Returns:
        Parsed representation of `json_str`, with literals at the leaf nodes of
        the parse tree replaced with `JsonLiteralWithPosition` instances containing
        position information.
    """
    tokens = tokenize_json(json_str)
    return reparse_value(tokens, 0)[0]


def scalar_paths(parsed_json) -> list[tuple]:
    """Get paths to all scalar values in parsed JSON.

    Args:
        parsed_json: JSON data parsed into native Python objects.

    Returns:
        A list of paths to scalar values within `parsed_json`, where each
        path is expressed as a tuple. The root element of a bare scalar is an empty
        tuple.
    """
    result = []
    if isinstance(parsed_json, dict):
        for key, value in parsed_json.items():
            result.extend([(key, *t) for t in scalar_paths(value)])
    elif isinstance(parsed_json, list):
        for i, value in enumerate(parsed_json):
            result.extend([(i, *t) for t in scalar_paths(value)])
    else:
        # Base case
        result.append(tuple())
    return result


def all_paths(parsed_json) -> list[tuple]:
    """Find all possible paths within a parsed JSON value.

    Args:
        parsed_json: JSON data parsed into native Python objects.

    Returns:
        A list of paths to all elements of the parse tree of `parsed_json`,
        where each path is expressed as a tuple. The root element of a bare scalar is
        an empty tuple.
    """
    result: list[tuple] = [tuple()]
    if isinstance(parsed_json, dict):
        for key, value in parsed_json.items():
            result.extend([(key, *t) for t in all_paths(value)])
    elif isinstance(parsed_json, list):
        for i, value in enumerate(parsed_json):
            result.extend([(i, *t) for t in all_paths(value)])
    return result


def fetch_path(json_value: Any, path: tuple):
    """Get the node at the indicated path in JSON.

    Args:
        json_value: Parsed JSON value.
        path: A tuple of names/numbers that indicates a path from root to a leaf
            or internal node of `json_value`.

    Returns:
        The node at the indicated path.

    Raises:
        TypeError: If `path` is not a tuple, if a path element is not a string
            or integer, or if an intermediate node is not a dict or list.
    """
    if not isinstance(path, tuple):
        raise TypeError(f"Expected tuple, but received '{type(path)}'")
    cur_json_value = json_value
    ix = 0
    while ix < len(path):
        cur_elem = path[ix]
        if not isinstance(cur_elem, str | int):
            raise TypeError(
                f"Found {type(cur_elem)} at element {ix} of path {path} "
                f"Expected string or integer"
            )
        if not isinstance(cur_json_value, dict | list):
            raise TypeError(
                f"Found {type(cur_json_value)} at path {path[:ix]} "
                f"of {json_value}. Was expecting dict or list."
            )
        # Both dicts and lists support indexing with appropriate types
        cur_json_value = cur_json_value[cur_elem]  # type: ignore[index,operator]
        ix += 1
    return cur_json_value


def replace_path(json_value: Any, path: tuple, new_value: Any) -> Any:
    """Modify a parsed JSON value in place by setting a particular path.

    Args:
        json_value: Parsed JSON value.
        path: A tuple of names/numbers indicating a path from root to the target node.
        new_value: New value to place at the indicated location.

    Returns:
        The modified input, or `new_value` itself if the root was replaced.

    Raises:
        TypeError: If `path` is not a tuple, or if any error propagated from
            :func:`fetch_path` during path traversal.
    """
    if not isinstance(path, tuple):
        raise TypeError(f"Expected tuple, but received '{type(path)}'")
    if len(path) == 0:
        # Root
        return new_value
    where_to_write = fetch_path(json_value, path[:-1])
    where_to_write[path[-1]] = new_value  # Modify in place
    return json_value


def parse_inline_json(json_response: dict) -> dict:
    """Replace the JSON strings in message contents with parsed JSON.

    Args:
        json_response: Parsed JSON representation of a `ChatCompletionResponse` object.

    Returns:
        Deep copy of the input with JSON message content strings replaced by parsed
        Python objects.
    """
    result = copy.deepcopy(json_response)

    for p in scalar_paths(json_response):
        if p[-1] == "content":
            # Found a content field. Parse the JSON string
            parsed_str = json.loads(fetch_path(result, p))
            replace_path(result, p, parsed_str)

    return result


def make_begin_to_token_table(logprobs: ChatCompletionLogProbs | None):
    """Create a table mapping token begin positions to token indices.

    Args:
        logprobs: The token log probabilities from the chat completion, or `None`
            if the chat completion request did not ask for logprobs.

    Returns:
        A dictionary mapping token begin positions to token indices,
        or `None` if `logprobs` is `None`.
    """
    if logprobs is None:
        return None
    content = logprobs.content
    if content is None:
        return None
    offset = 0
    result = {}
    for i, content_elem in enumerate(content):  # Linter prefers enumerate here
        result[offset] = i
        offset += len(content_elem.token)
    return result
