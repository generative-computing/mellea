from mellea.stdlib.base import (
    CBlock,
    GenerateLog,
    ModelOutputThunk,
    ContextTurn,
    LinearContext,
    SimpleContext,
    LegacyContext,
)


def run_on_context_1(ctx: LegacyContext):
    ctx.insert(CBlock("abc"), generate_logs=[GenerateLog()])
    o, l = ctx.last_output_and_logs()
    assert (
        o is None
    ), "There is only a Cblock in the context, not an output (ModelOutputThunk). Shouldn't return anything"
    assert l is None, "If there is no output, there should be no corresponding log"


def run_on_context_2(ctx: LegacyContext):
    ctx.insert(ModelOutputThunk("def"), generate_logs=[GenerateLog(), GenerateLog()])
    o, l = ctx.last_output_and_logs(all_intermediate_results=True)
    assert o is not None
    assert isinstance(l, list)
    assert len(l) == 2
    assert isinstance(l[0], GenerateLog)


def run_on_context_3(ctx: LegacyContext):
    for is_final in (True, False):
        ctx.insert_turn(
            ContextTurn(None, ModelOutputThunk("def")),
            generate_logs=[GenerateLog(is_final_result=is_final)],
        )
        o, l = ctx.last_output_and_logs()
        print(f"o={o}, l={l}")
        assert o is not None
        assert isinstance(l, GenerateLog)


def test_ctx_single_log():
    ctx = SimpleContext()
    run_on_context_1(ctx)
    run_on_context_2(ctx)
    run_on_context_3(ctx)


def test_ctx_multi_log():
    ctx = LinearContext()
    run_on_context_1(ctx)
    run_on_context_2(ctx)
    run_on_context_3(ctx)


def test_ctx_overlap():
    ctx = SimpleContext()
    run_on_context_1(ctx)
    ctx = LinearContext()
    run_on_context_1(ctx)

    ctx2 = SimpleContext()
    last_logs = ctx.get_logs_by_index(-1)
    assert isinstance(last_logs, list)
    assert len(last_logs) == 1
    assert isinstance(last_logs[0], GenerateLog)
    run_on_context_1(ctx2)
