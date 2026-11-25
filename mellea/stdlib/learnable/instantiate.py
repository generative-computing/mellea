# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instantiate: learn a tree of learnable functions, inner functions first.

When a learnable function's body calls another (`F` calls `G`), `F`'s student
must learn from rows in which `G` was answered by `G`'s student, not by `G`'s
teacher body: that is what `F`'s student will meet when it is served. So `G` is
trained first, installed as its policy (`G.policy(student)`), and the episodes are
captured again before `F` is trained. `instantiate` owns that loop end to end:

1. **Pass 0:** `capture` the episodes with every body answering
   (`out_dir/pass_0.jsonl.gz`; resumable, since it depends on no student).
2. **Groups and levels,** read from pass 0's trees (`Record.calls`): functions
   that call each other, directly or through others, or themselves, form one
   group (a strongly connected component of the call graph). A group's level
   is 0 if it calls no other group, otherwise one more than the highest level
   of the groups it calls.
3. **For each level:** train each of its functions from the latest pass
   (`train(fn, path) -> impl`), and install the students. A group's functions
   are trained together, from the same pass: their calls to each other stay
   answered by their bodies in each other's rows, which is all recursion needs
   (every call is a row, with its context, whoever produced that context). If
   levels remain, capture the next pass (`pass_<L+1>.jsonl.gz`) with every
   student so far installed. That file is replaced, never resumed: it depends
   on the students just trained, and a resumed one could mix students.

Training is the caller's: `train` is given a function and the path of the rows
to learn it from (`read_rows`, `sft_rows`, `preference_rows` read them), and
returns its student, as `Learnable.policy` takes one. Mellea does the ordering
only.

The episodes must be re-runnable (a sequence of `(key, run)`, each `run()`
starting afresh), since every level after the first runs them again. A failed
episode stops the learning rather than training on part of the data. Students
are installed only while `instantiate` runs; it returns them, for the caller to
install. The order is read from pass 0 alone: a student that leads its caller's
body to call a function the teachers never did is not seen.

```python
students = instantiate(episodes, Path("learned/"), [SupportAgent.reply, Planner.next],
                       train=train_adapter, score=score)
with Planner.next.policy(students["Planner.next"]), ...:
    serve()
```

The name completes `capture`'s CUDA-graph analogy: a captured graph is
instantiated into something executable, and a graph whose nodes are child graphs
needs its children instantiated first.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from ...core import Backend
from .capture import Scorer, capture, read_rows, walk_rows
from .learnable import Learnable, Record, Student

Trainer = Callable[[Learnable, Path], Student]
"""`train(fn, rows_path)`: `fn`'s student, trained on the rows `capture` wrote there."""


def _call_graph(rows: Iterable[dict[str, Any]]) -> dict[str, set[str]]:
    """Which function calls which, read from written rows at every depth.

    Raises:
        ValueError: If a row has no `function` or `calls` (a `record=` that
            drops them leaves nothing to order the functions by).
    """
    graph: dict[str, set[str]] = {}
    for r in walk_rows(rows):
        if "function" not in r or "calls" not in r:
            raise ValueError(
                "instantiate reads the call tree from the rows it captured: a "
                "`record=` must keep each row's `function` and `calls`"
            )
        graph.setdefault(r["function"], set()).update(c["function"] for c in r["calls"])
    return graph


def _groups(graph: dict[str, set[str]]) -> list[frozenset[str]]:
    """The strongly connected components of `graph` (Tarjan's algorithm)."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    groups: list[frozenset[str]] = []

    def visit(v: str) -> None:
        index[v] = low[v] = len(index)
        stack.append(v)
        on_stack.add(v)
        for w in sorted(graph.get(v, ())):
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            group = set()
            while True:
                w = stack.pop()
                on_stack.discard(w)
                group.add(w)
                if w == v:
                    break
            groups.append(frozenset(group))

    for v in sorted(graph):
        if v not in index:
            visit(v)
    return groups


def levels(rows: Iterable[dict[str, Any]]) -> list[list[frozenset[str]]]:
    """The groups of the functions written in `rows`, by level, inner first.

    Args:
        rows: Rows `capture` wrote (with each record's `function` and `calls`).

    Returns:
        `levels[L]`: the groups at level `L`, each a set of qualified names.
        A group is functions that call each other, directly or not, or
        themselves; its level is 0 if it calls no other group, otherwise one
        more than the highest level among the groups it calls.

    Raises:
        ValueError: If a row has no `function` or `calls`.
    """
    graph = _call_graph(rows)
    for names in list(graph.values()):
        for c in names:
            graph.setdefault(c, set())
    groups = _groups(graph)
    group_of = {name: g for g in groups for name in g}
    level: dict[frozenset[str], int] = {}
    # Tarjan's algorithm yields a group only after every group it calls.
    for g in groups:
        called = {group_of[c] for v in g for c in graph[v]} - {g}
        level[g] = 1 + max((level[h] for h in called), default=-1)
    out: list[list[frozenset[str]]] = [
        [] for _ in range(max(level.values(), default=-1) + 1)
    ]
    for g in groups:
        out[level[g]].append(g)
    return out


def instantiate(
    episodes: Sequence[tuple[str, Callable[[], Any]]],
    out_dir: Path,
    functions: Iterable[Learnable],
    train: Trainer,
    *,
    score: Scorer | None = None,
    record: Callable[[Record], dict[str, Any]] | None = None,
    meta: dict[str, Any] | None = None,
    student: Backend | None = None,
    log: Callable[[str], None] | None = print,
) -> dict[str, Student]:
    """Learn every learnable function the episodes call, inner functions first.

    See the module's docstring for the loop.

    Args:
        episodes: ``(key, run)`` pairs, as ``capture`` takes them; run once per
            level, so each ``run()`` must start afresh.
        out_dir: Where each pass's rows go (``pass_<L>.jsonl.gz``).
        functions: Every learnable function the episodes call, at any depth.
        train: ``train(fn, rows_path)``: ``fn``'s student, trained on those
            rows.
        score: Passed to ``capture`` on every pass.
        record: Passed to ``capture`` on every pass; it must keep each row's
            ``function`` and ``calls``.
        meta: Passed to ``capture`` on every pass.
        student: The backend whose rendering defines every call's input,
            passed to ``capture`` on every pass (``train`` reads the rows'
            requests).
        log: Where progress goes (``None``: nowhere); passed to ``capture``.

    Returns:
        Each function's student, by qualified name. They are no longer
        installed, so install them (``policy``) to serve.

    Raises:
        ValueError: If the episodes call a learnable function not in
            ``functions``, if one in ``functions`` is never called, or if a row
            has no ``function`` or ``calls``.
        RuntimeError: If an episode fails on any pass.
    """
    episodes = list(episodes)
    by_name = {f.name: f for f in functions}
    out_dir.mkdir(parents=True, exist_ok=True)

    def run_pass(n: int) -> Path:
        path = out_dir / f"pass_{n}.jsonl.gz"
        if n > 0:  # depends on the students just installed: never resumed
            path.unlink(missing_ok=True)
        summary = capture(
            episodes,
            path,
            score=score,
            record=record,
            meta=meta,
            student=student,
            log=log,
        )
        if summary["failed"]:
            raise RuntimeError(
                f"pass {n}: episodes {summary['failed']} failed; instantiate does "
                f"not train on part of the data (run again: pass 0 resumes)"
            )
        return path

    latest = run_pass(0)
    rows = read_rows(latest)
    graph = _call_graph(rows)
    traced = set(graph) | {c for cs in graph.values() for c in cs}
    unknown = sorted(traced - set(by_name))
    if unknown:
        raise ValueError(f"the episodes call learnable functions not given: {unknown}")
    unused = sorted(set(by_name) - traced)
    if unused:
        raise ValueError(f"learnable functions given but never called: {unused}")

    order = levels(rows)
    students: dict[str, Student] = {}
    with ExitStack() as installed:
        for level, groups in enumerate(order):
            names = sorted(n for g in groups for n in g)
            if log is not None:
                log(f"instantiate: level {level}: training {names} from {latest.name}")
            trained = {n: train(by_name[n], latest) for n in names}
            for n, impl in trained.items():
                installed.enter_context(by_name[n].policy(impl))
            students.update(trained)
            if level + 1 < len(order):
                latest = run_pass(level + 1)
    return students
