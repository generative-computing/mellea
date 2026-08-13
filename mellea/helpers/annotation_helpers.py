# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolution of postponed (PEP 563) annotations in callable signatures."""

import inspect
from collections.abc import Callable

from mellea.core.utils import MelleaLogger


def resolve_signature_annotations(func: Callable) -> inspect.Signature:
    """Return `func`'s signature with parameter annotations resolved where possible.

    Under `from __future__ import annotations` (PEP 563) every annotation is a
    string rather than a type object. `inspect.signature(func, eval_str=True)`
    resolves them, but it also evaluates the return annotation — so a return type
    imported only under `if TYPE_CHECKING:`, or a forward reference, raises
    `NameError` even when every parameter would resolve cleanly.

    This tries the whole signature first, and on any failure resolves each
    parameter annotation individually instead. The return annotation and any
    genuinely unresolvable parameter are left as their original strings, so this
    never resolves less than `inspect.signature(func)` alone would.

    Args:
        func: The callable to introspect.

    Returns:
        A signature whose parameter annotations are real type objects wherever
        they could be resolved. Annotations that could not be resolved remain
        strings; callers that build Pydantic models from them will surface that
        as `PydanticUserError`, and callers that render the signature as text
        will see the name quoted.
    """
    try:
        return inspect.signature(func, eval_str=True)
    except Exception as e:
        MelleaLogger.get_logger().debug(
            "Could not resolve the full signature of '%s' (%s); "
            "falling back to per-parameter annotation resolution: %s",
            getattr(func, "__name__", func),
            type(e).__name__,
            e,
        )

    sig = inspect.signature(func)
    # `inspect.signature` follows `__wrapped__`, so the annotations above may
    # come from a function in a different module than `func` itself. Take the
    # namespace from that same object, otherwise a `functools.wraps` decorator's
    # module supplies the wrong globals and a colliding type name resolves to the
    # wrong type. The `stop` predicate matches the one `inspect.signature` uses
    # to stop unwrapping.
    target = inspect.unwrap(func, stop=lambda f: hasattr(f, "__signature__"))
    # A module's globals already carry `__builtins__`, so no defensive copy is
    # needed.
    g = getattr(target, "__globals__", {})
    params = []
    for p in sig.parameters.values():
        if isinstance(p.annotation, str):
            try:
                # ast.literal_eval cannot evaluate type expressions (e.g.
                # `Decimal`, `Foo | None`); this mirrors what
                # `inspect.signature(..., eval_str=True)` does internally. The
                # input is an annotation from a callable the caller registered,
                # so it is no more attacker-controlled than the callable itself,
                # and the attempt above already evaluated a superset of these
                # strings.
                p = p.replace(annotation=eval(p.annotation, g))  # noqa: S307
            except Exception as e:
                # Leave as a string; the caller surfaces the unresolved name.
                MelleaLogger.get_logger().debug(
                    "Could not resolve annotation %r for parameter '%s' of '%s': %s",
                    p.annotation,
                    p.name,
                    getattr(func, "__name__", func),
                    e,
                )
        params.append(p)
    return sig.replace(parameters=params)
