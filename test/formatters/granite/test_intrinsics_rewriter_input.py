# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regressions for request rewriting with sentence boundaries."""

from mellea.formatters.granite import ChatCompletion, IntrinsicsRewriter


def test_sentence_boundary_rewrite_preserves_input_message():
    rewriter = IntrinsicsRewriter(
        config_dict={
            "model": None,
            "response_format": None,
            "transformations": None,
            "parameters": None,
            "sentence_boundaries": {"last_message": "r"},
            "instruction": None,
        }
    )
    request = ChatCompletion.model_validate(
        {"messages": [{"role": "user", "content": "First sentence. Second sentence."}]}
    )
    original = request.model_dump()

    rewritten = rewriter.transform(request)

    assert request.model_dump() == original
    assert rewritten.messages[-1].content != request.messages[-1].content
    assert rewriter.transform(request).model_dump() == rewritten.model_dump()
