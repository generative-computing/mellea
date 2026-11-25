# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture: run a driver over episodes, record every learnable call, score them, write them.

An *episode* is `(key, run)`: `run()` is plain driver code that calls
learnable functions (an environment's loop, a simulated user, recorded
traffic), so a conversation's next input can depend on the last output.
`capture` runs each episode under a `Recorder`, which keeps its calls as a tree:
the driver's own calls, each holding the calls its body made. Then, the episode
ended, it scores every call the body left unscored, with `score(record,
episode)`, which sees the whole tree, so what followed the call; and it writes
the episode's top-level calls together, one row each with its nested calls
inside, kept or not, each with its episode's key and number of top-level calls.
So a rerun skips the episodes already written and writes again one that was cut
short; an error fails only its episode, and the next run writes it:

```python
def run(ticket):
    agent = SupportAgent(ticket)
    for turn in ticket:
        reply = agent.reply(chat_at(turn), order=order_at(turn))
        ...

capture([(key, lambda t=t: run(t)) for key, t in tickets],
        Path("shard_0.jsonl.gz"), shard="0/8", score=score)
```

`sft_rows` turns the kept completions of one function, at any depth, into TRL's
conversational prompt-completion rows; `preference_rows` pairs a call's kept
completions with its others into TRL's preference rows, on the same stored
prompt. Every completion is written, kept or not: those not kept are the other
side of preferences, and later of RL.

The name is a CUDA graph's: run the real operations once, under a recording
mode, and keep a replayable record of them for a later step (`instantiate`,
then training) to learn from. Nothing is trained here.

An output path ending in `.gz` is written gzip-compressed (`read_rows` reads
either): every record holds its whole prompt, so the records of a long
conversation repeat it, and compress well.
"""

from __future__ import annotations

import gzip
import json
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from .learnable import Keep, Learnable, Record, Recorder, Score

Scorer = Callable[[Record, list[Record]], list[Score]]
"""`score(record, episode)`: one `Score` per completion of `record`, in order.
`episode` is the episode's top-level records, each holding its nested calls."""


def _open(path: Path, mode: str) -> Any:
    """`path` as text, gzip-compressed if its name ends in `.gz`."""
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def _read(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """`path`'s rows, and whether the file ends whole.

    Whole: no partial last row, no gzip stream cut short.
    """
    lines: list[str] = []
    whole = True
    try:
        with _open(path, "r") as f:
            for x in f:
                lines.append(x)
    except EOFError:  # a gzip stream cut short
        whole = False
    rows = []
    for i, x in enumerate(lines):
        if not x.strip():
            continue
        try:
            rows.append(json.loads(x))
        except json.JSONDecodeError:
            if i < len(lines) - 1:
                raise
            whole = False
    return rows, whole and (not lines or lines[-1].endswith("\n"))


def read_rows(path: Path | str) -> list[dict[str, Any]]:
    """The rows `capture` wrote to `path` (`.jsonl`, or `.jsonl.gz`).

    One row per top-level call, its nested calls inside it (`calls`). A run
    killed while it wrote can leave a partial last row, and a gzip stream
    without its end: what came before stands, the partial row is left out (as
    `capture`'s resume leaves out the episode it belonged to).

    Args:
        path: The file `capture` wrote.

    Returns:
        Its rows, in the order written. Rows written before records said who
        answered hold no `policy`: they read back with `policy: None`, as a
        call its body answered, at every depth.

    Raises:
        json.JSONDecodeError: If a row before the last is not JSON.
    """
    rows = _read(Path(path))[0]
    for r in walk_rows(rows):
        r.setdefault("policy", None)
    return rows


def _done(out: Path) -> set[str]:
    """The episodes `out` holds whole; it is rewritten without any cut short.

    It is rewritten whole too if the run before was killed while it wrote.
    """
    if not out.exists():
        return set()
    rows, whole = _read(out)
    count = Counter(r["episode"] for r in rows)
    want = {r["episode"]: r["episode_calls"] for r in rows}
    done = {k for k, c in count.items() if c == want[k]}
    kept = [r for r in rows if r["episode"] in done]
    if len(kept) < len(rows) or not whole:
        with _open(out, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in kept)
    return done


def walk(records: Iterable[Record]) -> Iterator[Record]:
    """Every record of a tree, each before the calls it made.

    Args:
        records: The tree's top-level records (an episode).

    Yields:
        Each record, in start order, at every depth.
    """
    for r in records:
        yield r
        yield from walk(r.calls)


def walk_rows(rows: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Every written record of rows `capture` wrote, each before its nested calls.

    Args:
        rows: Written rows (each a top-level record, its `calls` inside it).

    Yields:
        Each written record, in start order, at every depth.
    """
    for r in rows:
        yield r
        yield from walk_rows(r.get("calls") or ())


def _score(episode: list[Record], score: Scorer) -> None:
    """Score every record of `episode` its body left unscored, one whole list each.

    Raises:
        ValueError: If `score` returns a list whose length differs from the
            record's number of completions.
    """
    for r in walk(episode):
        if r.scores is None:
            scores = list(score(r, episode))
            if len(scores) != len(r.completions):
                raise ValueError(
                    f"the scorer gave {len(scores)} scores for "
                    f"{len(r.completions)} completions of {r.function}"
                )
            r.scores = scores


def capture(
    episodes: Iterable[tuple[str, Callable[[], Any]]],
    out: Path,
    formatter: Any,
    *,
    shard: str = "0/1",
    resume: bool = True,
    score: Scorer | None = None,
    record: Callable[[Record], dict[str, Any]] | None = None,
    meta: dict[str, Any] | None = None,
    keep: dict[Learnable, Keep] | None = None,
    log: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """Run the episodes, and write every learnable call they made to `out`.

    Scoring: `score` is called once for every record, at any depth, whose
    `scores` the body left `None`, in start order, and its whole list assigned
    at once; a record the body scored is not scored again. It receives the
    record and the episode's whole tree, so it can judge a nested call against
    what an enclosing or a sibling call did, and a call against what followed
    it. Without `score`, unscored records stay unscored, which training reads
    as every completion kept.

    Args:
        episodes: `(key, run)` pairs; `run()` is the episode's driver.
        out: The JSONL file the records go to (appended); gzip-compressed if
            its name ends in `.gz`.
        formatter: The backend's formatter (``TemplateFormatter``), so the
            learnable view is rendered through the same path the backend uses.
        shard: `"k/n"`: every n-th episode from the k-th.
        resume: Skip the episodes `out` already holds whole.
        score: `score(record, episode)`: one `Score` per completion of
            `record`, given the episode's top-level records.
        record: A top-level record's row, its nested calls included (default
            `Record.to_json`); the episode's key and number of top-level calls
            are added to it.
        meta: The run's own notes (the code's version, its settings), written
            with every row as `run`.
        keep: Which calls of each function are written (`Recorder`'s
            `keep`; `every(n)` writes one in n), counted per episode.
        log: Where a line per episode goes (`None`: nowhere).

    Returns:
        `{"episodes", "calls", "kept", "completions", "failed"}`: what this run
        wrote (`calls`: at every depth; `kept`: calls whose own completion, the
        first, was kept) and the keys of the episodes that failed (run again to
        write them). An episode fails if its driver raises, or if a scorer
        raises or returns a list of the wrong length.
    """
    k, n = map(int, shard.split("/"))
    mine = list(episodes)[k::n]
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done(out) if resume else set()
    row = record or Record.to_json
    summary: dict[str, Any] = {
        "episodes": 0,
        "calls": 0,
        "kept": 0,
        "completions": 0,
        "failed": [],
    }
    t0 = time.time()
    with _open(out, "a") as f:  # gzip: a new stream, after any before it
        for key, run in mine:
            if key in done:
                continue
            try:
                with Recorder(keep=keep, formatter=formatter) as rec:
                    run()
                episode = rec.records
                if score is not None:
                    _score(episode, score)
            except Exception as e:  # a server went away: the next run writes it
                summary["failed"].append(key)
                if log is not None:
                    log(f"episode {key} failed: {type(e).__name__}: {e}")
                continue
            extra = {"run": meta} if meta else {}
            rows = [
                {**row(r), "episode": key, "episode_calls": len(episode), **extra}
                for r in episode
            ]
            f.writelines(json.dumps(r) + "\n" for r in rows)  # the episode at once
            f.flush()
            every = list(walk(episode))
            summary["episodes"] += 1
            summary["calls"] += len(every)
            summary["completions"] += sum(len(r.completions) for r in every)
            summary["kept"] += sum(r.scores is None or r.scores[0].keep for r in every)
            if log is not None:
                log(
                    f"episode {key}: {len(every)} calls; so far {summary['calls']}, "
                    f"{100 * summary['kept'] / max(1, summary['calls']):.0f}% kept, "
                    f"{summary['calls'] / (time.time() - t0):.2f} calls/s"
                )
    return summary


def _kept(r: dict[str, Any]) -> list[bool]:
    """Which of a written record's completions training takes."""
    scores = r.get("scores")
    if scores is None:
        return [True] * len(r["completions"])
    return [bool(s["keep"]) for s in scores]


def _of(rows: Iterable[dict[str, Any]], function: Learnable | str) -> list[dict]:
    """The written records of `function`, at any depth, each before its nested calls."""
    name = function if isinstance(function, str) else function.name
    return [r for r in walk_rows(rows) if r["function"] == name]


def sft_rows(
    rows: Iterable[dict[str, Any]], function: Learnable | str
) -> list[dict[str, Any]]:
    """The kept completions of `function` as TRL's conversational prompt-completion rows.

    Each is `{"prompt": messages, "completion": [assistant message], "tools"}`.
    The calls of `function` are found at any depth.

    Args:
        rows: Written rows (as `Record.to_json` writes them: `function`,
            `prompt`, `completions`, `scores` and the nested `calls`).
        function: The learnable function, or its qualified name.

    Returns:
        One row per completion kept, in order.
    """
    return [
        {
            "prompt": r["prompt"]["messages"],
            "completion": [{"role": "assistant", "content": c["completion"]}],
            "tools": r["prompt"]["tools"],
        }
        for r in _of(rows, function)
        for c, kept in zip(r["completions"], _kept(r))
        if kept
    ]


def preference_rows(
    rows: Iterable[dict[str, Any]], function: Learnable | str
) -> list[dict[str, Any]]:
    """Each call's kept completions against its others, as TRL's preference rows.

    Each is `{"prompt": messages, "chosen": [assistant message], "rejected":
    [assistant message], "tools", "meta"}` (`meta`: the two completions' own,
    for analysis), on the call's stored prompt. The calls of `function` are
    found at any depth.

    Args:
        rows: Written rows, as for `sft_rows`.
        function: The learnable function, or its qualified name.

    Returns:
        One row per (kept, not kept) pair of a call's completions, in order.
    """
    out = []
    for r in _of(rows, function):
        kept = _kept(r)
        good = [c for c, k in zip(r["completions"], kept) if k]
        bad = [c for c, k in zip(r["completions"], kept) if not k]
        for g in good:
            for b in bad:
                out.append(
                    {
                        "prompt": r["prompt"]["messages"],
                        "chosen": [{"role": "assistant", "content": g["completion"]}],
                        "rejected": [{"role": "assistant", "content": b["completion"]}],
                        "tools": r["prompt"]["tools"],
                        "meta": {"chosen": g["meta"], "rejected": b["meta"]},
                    }
                )
    return out
