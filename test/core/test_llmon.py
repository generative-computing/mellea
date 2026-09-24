# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the LLMON (LLM Object Notation) parser and serializer.

LLMON uses backslash-delimited tags (``\\object\\``, ``/object/`` etc.) and
double-quoted strings. In these tests every literal is a normal (non-raw)
Python string: a single LLMON backslash is written ``\\\\`` and a literal
double quote inside an LLMON string is written ``\\"``.
"""

import pytest

from mellea.core.llmon import (
    LLMONBool,
    LLMONDocument,
    LLMONFloat,
    LLMONInt,
    LLMONItem,
    LLMONList,
    LLMONNull,
    LLMONObject,
    LLMONParseError,
    LLMONString,
    LLMONUserType,
    parse_llmon,
    to_llmon,
)


class TestParseScalars:
    """Scalars parse to the correct node type."""

    def test_integer(self):
        assert parse_llmon("42") == LLMONDocument([LLMONInt(42)])

    def test_negative_integer(self):
        assert parse_llmon("-7") == LLMONDocument([LLMONInt(-7)])

    def test_float(self):
        assert parse_llmon("3.14") == LLMONDocument([LLMONFloat(3.14)])

    def test_true(self):
        assert parse_llmon("true") == LLMONDocument([LLMONBool(True)])

    def test_false(self):
        assert parse_llmon("false") == LLMONDocument([LLMONBool(False)])

    def test_null(self):
        assert parse_llmon("null") == LLMONDocument([LLMONNull()])

    def test_string(self):
        assert parse_llmon('"hello world"') == LLMONDocument(
            [LLMONString("hello world")]
        )

    def test_string_with_escapes(self):
        assert parse_llmon('"he said \\"hi\\""') == LLMONDocument(
            [LLMONString('he said "hi"')]
        )


class TestParseObject:
    """OBJECT productions."""

    def test_empty_object(self):
        assert parse_llmon("\\object\\/object/") == LLMONDocument([LLMONObject([])])

    def test_object_single_item(self):
        src = '\\object\\\\item\\"name":"ada"/item//object/'
        assert parse_llmon(src) == LLMONDocument(
            [LLMONObject([LLMONItem("name", LLMONString("ada"))])]
        )

    def test_object_multiple_items(self):
        src = '\\object\\\\item\\"a":1/item/\\item\\"b":2/item//object/'
        assert parse_llmon(src) == LLMONDocument(
            [LLMONObject([LLMONItem("a", LLMONInt(1)), LLMONItem("b", LLMONInt(2))])]
        )

    def test_object_item_key_with_dot_and_colon(self):
        src = '\\object\\\\item\\"a.b:c":true/item//object/'
        assert parse_llmon(src) == LLMONDocument(
            [LLMONObject([LLMONItem("a.b:c", LLMONBool(True))])]
        )


class TestParseList:
    """LIST productions."""

    def test_empty_list(self):
        assert parse_llmon("\\list\\/list/") == LLMONDocument([LLMONList([])])

    def test_list_single_item(self):
        assert parse_llmon("\\list\\1/list/") == LLMONDocument(
            [LLMONList([LLMONInt(1)])]
        )

    def test_list_mixed_items(self):
        src = '\\list\\1,"two",true,null/list/'
        assert parse_llmon(src) == LLMONDocument(
            [LLMONList([LLMONInt(1), LLMONString("two"), LLMONBool(True), LLMONNull()])]
        )


class TestParseUserType:
    """USER_TYPE productions."""

    def test_user_type_with_children(self):
        assert parse_llmon("\\instruction\\42/instruction/") == LLMONDocument(
            [LLMONUserType("instruction", [LLMONInt(42)])]
        )

    def test_self_closing_user_type(self):
        assert parse_llmon("\\br/") == LLMONDocument([LLMONUserType("br", [])])

    def test_user_tag_with_dot_and_colon(self):
        assert parse_llmon("\\ns:type.name\\1/ns:type.name/") == LLMONDocument(
            [LLMONUserType("ns:type.name", [LLMONInt(1)])]
        )

    def test_user_type_multiple_children_is_sequence(self):
        assert parse_llmon("\\wrap\\1 2/wrap/") == LLMONDocument(
            [LLMONUserType("wrap", [LLMONInt(1), LLMONInt(2)])]
        )


class TestParseNested:
    """Structures nest arbitrarily."""

    def test_object_containing_list(self):
        src = '\\object\\\\item\\"xs":\\list\\1,2/list//item//object/'
        assert parse_llmon(src) == LLMONDocument(
            [LLMONObject([LLMONItem("xs", LLMONList([LLMONInt(1), LLMONInt(2)]))])]
        )

    def test_user_type_wrapping_object(self):
        # The prompt-injection boundary use case: untrusted data wrapped
        # in a semantic user tag.
        src = '\\user_data\\\\object\\\\item\\"q":"hi"/item//object//user_data/'
        assert parse_llmon(src) == LLMONDocument(
            [
                LLMONUserType(
                    "user_data", [LLMONObject([LLMONItem("q", LLMONString("hi"))])]
                )
            ]
        )


class TestParseDocumentSequence:
    """Top-level LLMON is a sequence of items."""

    def test_multiple_top_level_items(self):
        assert parse_llmon("1 2 3") == LLMONDocument(
            [LLMONInt(1), LLMONInt(2), LLMONInt(3)]
        )

    def test_empty_document(self):
        assert parse_llmon("") == LLMONDocument([])


class TestSerialize:
    """to_llmon renders nodes back to LLMON text."""

    def test_scalar_int(self):
        assert to_llmon(LLMONInt(42)) == "42"

    def test_scalar_string(self):
        assert to_llmon(LLMONString("hi")) == '"hi"'

    def test_string_escaped(self):
        assert to_llmon(LLMONString('a"b')) == '"a\\"b"'

    def test_null(self):
        assert to_llmon(LLMONNull()) == "null"

    def test_empty_object(self):
        assert to_llmon(LLMONObject([])) == "\\object\\/object/"

    def test_object_item(self):
        node = LLMONObject([LLMONItem("k", LLMONInt(1))])
        assert to_llmon(node) == '\\object\\\\item\\"k":1/item//object/'

    def test_empty_list(self):
        assert to_llmon(LLMONList([])) == "\\list\\/list/"

    def test_list_items(self):
        node = LLMONList([LLMONInt(1), LLMONInt(2)])
        assert to_llmon(node) == "\\list\\1,2/list/"

    def test_self_closing_user_type(self):
        assert to_llmon(LLMONUserType("br", [])) == "\\br/"

    def test_user_type_with_children(self):
        assert to_llmon(LLMONUserType("x", [LLMONInt(1)])) == "\\x\\1/x/"

    def test_document_sequence(self):
        doc = LLMONDocument([LLMONInt(1), LLMONInt(2)])
        assert to_llmon(doc) == "1 2"


class TestRoundTrip:
    """parse_llmon(to_llmon(node)) == node."""

    @pytest.mark.parametrize(
        "node",
        [
            LLMONInt(0),
            LLMONInt(-5),
            LLMONFloat(2.5),
            LLMONBool(True),
            LLMONBool(False),
            LLMONNull(),
            LLMONString("abc"),
            LLMONString('with "quotes"'),
            LLMONObject([]),
            LLMONObject([LLMONItem("k", LLMONInt(1))]),
            LLMONList([]),
            LLMONList([LLMONInt(1), LLMONString("x")]),
            LLMONUserType("tag", []),
            LLMONUserType("tag", [LLMONInt(1)]),
            LLMONObject([LLMONItem("xs", LLMONList([LLMONInt(1)]))]),
        ],
    )
    def test_node_round_trip(self, node):
        doc = LLMONDocument([node])
        assert parse_llmon(to_llmon(doc)) == doc


class TestParseErrors:
    """Malformed input raises LLMONParseError."""

    def test_unclosed_object(self):
        with pytest.raises(LLMONParseError):
            parse_llmon('\\object\\\\item\\"k":1/item/')

    def test_mismatched_close_tag(self):
        with pytest.raises(LLMONParseError):
            parse_llmon("\\object\\/list/")

    def test_invalid_user_tag_leading_digit(self):
        with pytest.raises(LLMONParseError):
            parse_llmon("\\1bad\\1/1bad/")

    def test_item_missing_colon(self):
        with pytest.raises(LLMONParseError):
            parse_llmon('\\object\\\\item\\"k"1/item//object/')

    def test_unterminated_string(self):
        with pytest.raises(LLMONParseError):
            parse_llmon('"no end')

    def test_item_outside_object(self):
        with pytest.raises(LLMONParseError):
            parse_llmon('\\item\\"k":1/item/')
