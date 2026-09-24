# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

r"""LLMON (LLM Object Notation) node model, parser, and serializer.

LLMON is an LLM-native markup for conveying the structure and semantic metadata
of text to a language model — for example, distinguishing instructions from data
to reduce prompt-injection risk. See arXiv:2603.22519.

This module implements the value layer only: a dedicated node hierarchy
(`LLMONNode` and subclasses), a `parse_llmon` reader (text to node tree), and a
`to_llmon` writer (node tree to text). It does not wire LLMON into the formatter,
backend, or `instruct` paths — those are separate follow-ups.

Syntax (from the grammar in the LLMON paper):

- Tags are backslash-delimited: open `\tag\`, close `/tag/`, self-close `\tag/`.
- Objects: `\object\ ... /object/`, holding zero or more items
  `\item\ "key" : value /item/`.
- Lists: `\list\ v1 , v2 /list/`.
- User types: `\name\ ... /name/` or self-closing `\name/`, where `name`
  matches `[_a-zA-Z][_a-zA-Z0-9.:]*`.
- Scalars: integer, float, double-quoted string, `true`, `false`, `null`.
- A document is a whitespace-separated sequence of items.

Two grammar readings are fixed here, because the published grammar is ambiguous
on them (it references an undefined `LMON` non-terminal and an undefined
`end_tag`):

- An object-item value and a list element are each a single item (one
  `LLMON_ITEM`), not a nested sequence.
- The list terminator is `/list/`.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

__all__ = [
    "LLMONBool",
    "LLMONDocument",
    "LLMONFloat",
    "LLMONInt",
    "LLMONItem",
    "LLMONList",
    "LLMONNode",
    "LLMONNull",
    "LLMONObject",
    "LLMONParseError",
    "LLMONString",
    "LLMONUserType",
    "parse_llmon",
    "to_llmon",
]

# A user-defined tag name: identifier that may include `.` and `:`.
_USER_TAG_RE = re.compile(r"[_a-zA-Z][_a-zA-Z0-9.:]*")
# The reserved structural tag names, which are never user types.
_RESERVED_TAGS = frozenset({"object", "item", "list"})


class LLMONParseError(ValueError):
    """Raised when a string cannot be parsed as valid LLMON."""


class LLMONNode(ABC):
    """Abstract base class for every node in a parsed LLMON tree.

    Subclasses are frozen dataclasses, so node trees compare by value and are
    hashable where their contents allow.
    """

    @abstractmethod
    def to_llmon(self) -> str:
        """Serialize this node to its LLMON text form.

        Returns:
            The LLMON representation of this node.
        """
        ...


@dataclass(frozen=True)
class LLMONInt(LLMONNode):
    """An LLMON integer scalar.

    Args:
        value: The integer value.
    """

    value: int

    def to_llmon(self) -> str:
        """Serialize the integer.

        Returns:
            The decimal representation of `value`.
        """
        return str(self.value)


@dataclass(frozen=True)
class LLMONFloat(LLMONNode):
    """An LLMON floating-point scalar.

    Args:
        value: The float value.
    """

    value: float

    def to_llmon(self) -> str:
        """Serialize the float.

        Returns:
            The `repr` of `value`, which round-trips through the parser.
        """
        return repr(self.value)


@dataclass(frozen=True)
class LLMONString(LLMONNode):
    """An LLMON string scalar.

    Args:
        value: The decoded string value (without surrounding quotes).
    """

    value: str

    def to_llmon(self) -> str:
        """Serialize the string with surrounding quotes and escaping.

        Returns:
            The double-quoted, backslash-escaped LLMON string literal.
        """
        escaped = self.value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'


@dataclass(frozen=True)
class LLMONBool(LLMONNode):
    """An LLMON boolean scalar.

    Args:
        value: The boolean value.
    """

    value: bool

    def to_llmon(self) -> str:
        """Serialize the boolean.

        Returns:
            `"true"` or `"false"`.
        """
        return "true" if self.value else "false"


@dataclass(frozen=True)
class LLMONNull(LLMONNode):
    """The LLMON null scalar, representing an absent value."""

    def to_llmon(self) -> str:
        """Serialize null.

        Returns:
            The literal `"null"`.
        """
        return "null"


@dataclass(frozen=True)
class LLMONItem(LLMONNode):
    """A single key/value entry inside an LLMON object.

    Args:
        key: The item key (an LLMON string in the source).
        value: The item value node.
    """

    key: str
    value: LLMONNode

    def to_llmon(self) -> str:
        r"""Serialize the item.

        Returns:
            The `\item\ "key" : value /item/` representation.
        """
        key_str = LLMONString(self.key).to_llmon()
        return f"\\item\\{key_str}:{self.value.to_llmon()}/item/"


@dataclass(frozen=True)
class LLMONObject(LLMONNode):
    """An LLMON object: an ordered collection of key/value items.

    Args:
        items: The object's items, in order.
    """

    items: list[LLMONItem] = field(default_factory=list)

    def to_llmon(self) -> str:
        r"""Serialize the object.

        Returns:
            The `\object\ ... /object/` representation.
        """
        body = "".join(item.to_llmon() for item in self.items)
        return f"\\object\\{body}/object/"


@dataclass(frozen=True)
class LLMONList(LLMONNode):
    """An LLMON list: an ordered collection of values.

    Args:
        items: The list's element nodes, in order.
    """

    items: list[LLMONNode] = field(default_factory=list)

    def to_llmon(self) -> str:
        r"""Serialize the list.

        Returns:
            The `\list\ v1 , v2 /list/` representation.
        """
        body = ",".join(item.to_llmon() for item in self.items)
        return f"\\list\\{body}/list/"


@dataclass(frozen=True)
class LLMONUserType(LLMONNode):
    r"""An LLMON user-defined type: a semantic tag wrapping a sequence of items.

    A user type with no children serializes to the self-closing form `\tag/`.

    Args:
        tag: The tag name; must match `[_a-zA-Z][_a-zA-Z0-9.:]*` and not be a
            reserved structural name (`object`, `item`, `list`).
        children: The wrapped item nodes, in order.
    """

    tag: str
    children: list[LLMONNode] = field(default_factory=list)

    def to_llmon(self) -> str:
        r"""Serialize the user type.

        Returns:
            `\tag/` when there are no children, otherwise
            `\tag\ ... /tag/`.
        """
        if not self.children:
            return f"\\{self.tag}/"
        body = "".join(child.to_llmon() for child in self.children)
        return f"\\{self.tag}\\{body}/{self.tag}/"


@dataclass(frozen=True)
class LLMONDocument(LLMONNode):
    """A top-level LLMON document: a sequence of items.

    Args:
        items: The document's top-level item nodes, in order.
    """

    items: list[LLMONNode] = field(default_factory=list)

    def to_llmon(self) -> str:
        """Serialize the document.

        Returns:
            The whitespace-separated LLMON representation of the items.
        """
        return " ".join(item.to_llmon() for item in self.items)


class _Parser:
    """Recursive-descent LLMON parser over a raw source string.

    This is a scanner-less parser: it advances an index over `src` directly,
    matching literal tag strings and scalar patterns as it goes.
    """

    # Scalar patterns anchored at the current position.
    _FLOAT_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+[eE][+-]?\d+)")
    _INT_RE = re.compile(r"[+-]?\d+")

    def __init__(self, src: str) -> None:
        """Initialize the parser at the start of `src`."""
        self.src = src
        self.pos = 0
        self.n = len(src)

    # -- low-level cursor helpers ------------------------------------------

    def _skip_ws(self) -> None:
        while self.pos < self.n and self.src[self.pos].isspace():
            self.pos += 1

    def _at_end(self) -> bool:
        return self.pos >= self.n

    def _looking_at(self, literal: str) -> bool:
        return self.src.startswith(literal, self.pos)

    def _consume(self, literal: str) -> bool:
        if self._looking_at(literal):
            self.pos += len(literal)
            return True
        return False

    def _error(self, message: str) -> LLMONParseError:
        return LLMONParseError(f"{message} (at position {self.pos})")

    # -- grammar ------------------------------------------------------------

    def parse_document(self) -> LLMONDocument:
        """Parse a whitespace-separated sequence of top-level items."""
        items: list[LLMONNode] = []
        self._skip_ws()
        while not self._at_end():
            items.append(self._parse_item())
            self._skip_ws()
        return LLMONDocument(items)

    def _parse_item(self) -> LLMONNode:
        """Parse a single LLMON_ITEM: object, list, user type, or scalar."""
        self._skip_ws()
        if self._at_end():
            raise self._error("unexpected end of input; expected a value")

        if self.src[self.pos] == "\\":
            return self._parse_tagged()
        if self.src[self.pos] == "/":
            raise self._error("unexpected closing tag; expected a value")
        return self._parse_scalar()

    def _parse_tagged(self) -> LLMONNode:
        """Parse a backslash-tagged construct: object, list, or user type."""
        if self._consume("\\object\\"):
            return self._parse_object_body()
        if self._consume("\\list\\"):
            return self._parse_list_body()
        # `\item\` is only valid inside an object body.
        if self._looking_at("\\item\\"):
            raise self._error("'\\item\\' is only valid inside an object")
        return self._parse_user_type()

    def _parse_object_body(self) -> LLMONObject:
        """Parse object items up to and including the `/object/` terminator."""
        items: list[LLMONItem] = []
        while True:
            self._skip_ws()
            if self._consume("/object/"):
                return LLMONObject(items)
            if self._at_end():
                raise self._error("unterminated object; expected '/object/'")
            if not self._looking_at("\\item\\"):
                raise self._error("expected '\\item\\' or '/object/' in object")
            items.append(self._parse_object_item())

    def _parse_object_item(self) -> LLMONItem:
        r"""Parse `\item\ "key" : value /item/`."""
        if not self._consume("\\item\\"):
            raise self._error("expected '\\item\\'")
        self._skip_ws()
        if self._at_end() or self.src[self.pos] != '"':
            raise self._error("expected a quoted key string in item")
        key = self._parse_string_literal()
        self._skip_ws()
        if not self._consume(":"):
            raise self._error("expected ':' after item key")
        value = self._parse_item()
        self._skip_ws()
        if not self._consume("/item/"):
            raise self._error("expected '/item/' to close item")
        return LLMONItem(key, value)

    def _parse_list_body(self) -> LLMONList:
        """Parse list elements up to and including the `/list/` terminator."""
        self._skip_ws()
        if self._consume("/list/"):
            return LLMONList([])
        items: list[LLMONNode] = [self._parse_item()]
        while True:
            self._skip_ws()
            if self._consume("/list/"):
                return LLMONList(items)
            if self._consume(","):
                items.append(self._parse_item())
                continue
            raise self._error("expected ',' or '/list/' in list")

    def _parse_user_type(self) -> LLMONUserType:
        r"""Parse `\tag\ ... /tag/` or the self-closing `\tag/`."""
        if not self._consume("\\"):
            raise self._error("expected '\\' to open a tag")
        match = _USER_TAG_RE.match(self.src, self.pos)
        if match is None or match.start() != self.pos:
            raise self._error("invalid user tag name")
        tag = match.group(0)
        if tag in _RESERVED_TAGS:
            raise self._error(f"'{tag}' is a reserved tag name")
        self.pos = match.end()

        if self._consume("/"):
            # Self-closing: `\tag/`.
            return LLMONUserType(tag, [])
        if not self._consume("\\"):
            raise self._error("expected '\\' or '/' after tag name")

        children: list[LLMONNode] = []
        close = f"/{tag}/"
        while True:
            self._skip_ws()
            if self._consume(close):
                return LLMONUserType(tag, children)
            if self._at_end():
                raise self._error(f"unterminated user type; expected '{close}'")
            children.append(self._parse_item())

    def _parse_scalar(self) -> LLMONNode:
        """Parse a scalar: string, keyword literal, float, or integer."""
        ch = self.src[self.pos]
        if ch == '"':
            return LLMONString(self._parse_string_literal())

        for keyword, node in (
            ("true", LLMONBool(True)),
            ("false", LLMONBool(False)),
            ("null", LLMONNull()),
        ):
            if self._looking_at(keyword) and self._keyword_ends(keyword):
                self.pos += len(keyword)
                return node

        float_match = self._FLOAT_RE.match(self.src, self.pos)
        if float_match is not None and float_match.start() == self.pos:
            self.pos = float_match.end()
            return LLMONFloat(float(float_match.group(0)))

        int_match = self._INT_RE.match(self.src, self.pos)
        if int_match is not None and int_match.start() == self.pos:
            self.pos = int_match.end()
            return LLMONInt(int(int_match.group(0)))

        raise self._error("expected a value")

    def _keyword_ends(self, keyword: str) -> bool:
        """True if `keyword` at the cursor is not a prefix of a longer token."""
        end = self.pos + len(keyword)
        if end >= self.n:
            return True
        nxt = self.src[end]
        return not (nxt.isalnum() or nxt in "_.:")

    def _parse_string_literal(self) -> str:
        """Parse a double-quoted string with backslash escapes."""
        if self.src[self.pos] != '"':
            raise self._error("expected '\"' to open a string")
        self.pos += 1
        out: list[str] = []
        while self.pos < self.n:
            ch = self.src[self.pos]
            if ch == "\\":
                if self.pos + 1 >= self.n:
                    raise self._error("dangling escape in string")
                out.append(self.src[self.pos + 1])
                self.pos += 2
                continue
            if ch == '"':
                self.pos += 1
                return "".join(out)
            out.append(ch)
            self.pos += 1
        raise self._error("unterminated string")


def parse_llmon(text: str) -> LLMONDocument:
    """Parse an LLMON string into a document node tree.

    Args:
        text: The LLMON source to parse. An empty or whitespace-only string
            yields an empty document.

    Returns:
        The parsed `LLMONDocument`.

    Raises:
        LLMONParseError: If `text` is not valid LLMON.
    """
    return _Parser(text).parse_document()


def to_llmon(node: LLMONNode) -> str:
    """Serialize an LLMON node tree back to its text form.

    Args:
        node: The node to serialize (commonly an `LLMONDocument`).

    Returns:
        The LLMON text representation of `node`.
    """
    return node.to_llmon()
