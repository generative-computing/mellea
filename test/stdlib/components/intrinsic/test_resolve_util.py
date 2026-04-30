"""Tests for _resolve_question and _resolve_response."""

import pytest

from mellea.core import CBlock, ModelOutputThunk
from mellea.stdlib.components import Document, Instruction, Message
from mellea.stdlib.components.intrinsic._util import (
    _resolve_question,
    _resolve_response,
)
from mellea.stdlib.context import ChatContext


class TestResolveQuestion:
    def test_explicit_string(self):
        ctx = ChatContext()
        text, returned_ctx = _resolve_question("hello", ctx)
        assert text == "hello"
        assert returned_ctx is ctx

    def test_from_context(self):
        ctx = ChatContext().add(Message("user", "What is 2+2?"))
        text, rewound = _resolve_question(None, ctx)
        assert text == "What is 2+2?"
        assert rewound.is_root_node  # type: ignore[union-attr]

    def test_context_with_prior_messages(self):
        ctx = (
            ChatContext()
            .add(Message("user", "first"))
            .add(Message("assistant", "reply"))
            .add(Message("user", "second"))
        )
        text, rewound = _resolve_question(None, ctx)
        assert text == "second"
        # Rewound context should end with the assistant reply
        last = rewound.last_turn()  # type: ignore[union-attr]
        assert last is not None
        assert isinstance(last.model_input, Message)
        assert last.model_input.content == "reply"

    def test_empty_context_raises(self):
        ctx = ChatContext()
        with pytest.raises(ValueError, match="no last turn"):
            _resolve_question(None, ctx)

    def test_from_cblock(self):
        ctx = ChatContext().add(CBlock("raw question"))
        text, rewound = _resolve_question(None, ctx)
        assert text == "raw question"
        assert rewound.is_root_node  # type: ignore[union-attr]

    def test_cblock_none_value_raises(self):
        ctx = ChatContext().add(CBlock(None))
        with pytest.raises(ValueError, match="no value"):
            _resolve_question(None, ctx)

    def test_from_component(self):
        ctx = ChatContext().add(Document("some document text"))
        text, rewound = _resolve_question(None, ctx)
        assert "some document text" in text
        assert rewound.is_root_node  # type: ignore[union-attr]

    def test_from_instruction_component(self):
        ctx = ChatContext().add(Instruction("Summarize the article"))
        text, rewound = _resolve_question(None, ctx)
        assert "Summarize the article" in text
        assert rewound.is_root_node  # type: ignore[union-attr]

    def test_from_component_uses_backend_formatter(self):
        from unittest.mock import MagicMock

        ctx = ChatContext().add(Document("some document text"))
        mock_backend = MagicMock()
        mock_backend.formatter.print.return_value = "custom formatted"

        text, rewound = _resolve_question(None, ctx, backend=mock_backend)
        assert text == "custom formatted"
        mock_backend.formatter.print.assert_called_once()
        assert rewound.is_root_node  # type: ignore[union-attr]


class TestResolveResponse:
    def test_explicit_string(self):
        ctx = ChatContext()
        text, returned_ctx = _resolve_response("answer", ctx)
        assert text == "answer"
        assert returned_ctx is ctx

    def test_from_context(self):
        ctx = (
            ChatContext()
            .add(Message("user", "question"))
            .add(ModelOutputThunk(value="The answer is 4."))
        )
        text, rewound = _resolve_response(None, ctx)
        assert text == "The answer is 4."
        # Rewound context should still have the user question
        last = rewound.last_turn()  # type: ignore[union-attr]
        assert last is not None
        assert isinstance(last.model_input, Message)
        assert last.model_input.content == "question"

    def test_empty_context_raises(self):
        ctx = ChatContext()
        with pytest.raises(ValueError, match="no last turn"):
            _resolve_response(None, ctx)

    def test_none_value_raises(self):
        ctx = ChatContext().add(ModelOutputThunk(value=None))
        with pytest.raises(ValueError, match="no value"):
            _resolve_response(None, ctx)
