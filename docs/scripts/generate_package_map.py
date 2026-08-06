#!/usr/bin/env python3
"""Generate the "Package map" reference pages under docs/docs/reference/package/.

A static, source-pinned inventory of mellea's public import surface:
one Markdown page per public module (second level), with deeper modules
folded into their subtree page. Every symbol links to the exact source
line at the pinned commit, and missing type annotations are reported
honestly instead of guessed (see issue #1177).

Standalone: Python 3.11+ stdlib only. Never imports the target package
(pure AST), so it has no side effects and no environment dependence.

Usage (from the repository root):

    python docs/scripts/generate_package_map.py --pin <full-commit-sha> \
        [--repo-root .] [--out docs/docs/reference/package]

Definitions (mechanical, no hand-picking):
- PUBLIC MODULE: a ``.py`` file (or a package via its ``__init__.py``)
  under ``mellea/`` whose dotted path has no component with a leading
  underscore (``__init__`` names its parent package).
- PUBLIC SYMBOL: a top-level ``class`` / ``def`` / ``async def`` whose
  name has no leading underscore, in a public module.
- PAGE UNIT: modules with at most three dotted components
  (``mellea.backends.ollama``) get their own page; deeper modules are
  folded as sections into the page of their three-component ancestor.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

REPO_BLOB = "https://github.com/generative-computing/mellea/blob"


# ---------------------------------------------------------------- discovery

def discover_modules(repo_root: Path) -> dict[str, Path]:
    pkg = repo_root / "mellea"
    out: dict[str, Path] = {}
    for py in sorted(pkg.rglob("*.py")):
        rel = py.relative_to(repo_root)
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if any(p.startswith("_") for p in parts):
            continue
        out[".".join(parts)] = py
    return out


# ---------------------------------------------------------------- extraction

def first_paragraph(doc: str | None) -> str:
    if not doc:
        return ""
    para = doc.strip().split("\n\n")[0]
    return " ".join(para.split())


def esc(text: str) -> str:
    """Escape ``<`` outside inline-code spans so CommonMark never sees raw HTML."""
    segs = text.split("`")
    return "`".join(s.replace("<", "\\<") if i % 2 == 0 else s for i, s in enumerate(segs))


def is_named(dec: ast.expr, name: str) -> bool:
    d = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(d, ast.Name):
        return d.id == name
    if isinstance(d, ast.Attribute):
        return d.attr == name
    return False


def signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, list[str], bool]:
    """Render the def's signature exactly as annotated in source.

    Returns (signature_text, unannotated_param_names, has_return_annotation).
    """
    a = fn.args
    parts: list[str] = []
    missing: list[str] = []
    pos = list(a.posonlyargs) + list(a.args)
    defaults = list(a.defaults)
    # align defaults to the tail of positional args
    pad = [None] * (len(pos) - len(defaults))
    pos_defaults = pad + defaults

    def one(arg: ast.arg, default: ast.expr | None) -> str:
        s = arg.arg
        if arg.annotation is not None:
            s += f": {ast.unparse(arg.annotation)}"
        elif arg.arg not in ("self", "cls"):
            missing.append(arg.arg)
        if default is not None:
            s += f" = {ast.unparse(default)}" if arg.annotation is not None else f"={ast.unparse(default)}"
        return s

    for arg, d in zip(pos, pos_defaults):
        parts.append(one(arg, d))
        if a.posonlyargs and arg is a.posonlyargs[-1]:
            parts.append("/")
    if a.vararg is not None:
        s = "*" + a.vararg.arg
        if a.vararg.annotation is not None:
            s += f": {ast.unparse(a.vararg.annotation)}"
        parts.append(s)
    elif a.kwonlyargs:
        parts.append("*")
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        parts.append(one(arg, d))
    if a.kwarg is not None:
        s = "**" + a.kwarg.arg
        if a.kwarg.annotation is not None:
            s += f": {ast.unparse(a.kwarg.annotation)}"
        parts.append(s)
    if parts and parts[0] in ("self", "cls"):
        parts = parts[1:]  # rendered signatures follow call-site convention
        if parts and parts[0] == "*" and len(parts) == 1:
            parts = []
    sig = f"{fn.name}({', '.join(parts)})"
    has_ret = fn.returns is not None
    if has_ret:
        sig += f" -> {ast.unparse(fn.returns)}"
    return sig, missing, has_ret


def extract(py: Path, repo_root: Path) -> dict:
    tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
    info: dict = {
        "file": str(py.relative_to(repo_root)).replace("\\", "/"),
        "doc": first_paragraph(ast.get_docstring(tree)),
        "all": None,
        "classes": [],
        "functions": [],
    }
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__all__":
                    try:
                        info["all"] = [str(e) for e in ast.literal_eval(node.value)]
                    except Exception:
                        info["all"] = None
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            info["classes"].append(extract_class(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            if any(is_named(d, "overload") for d in node.decorator_list):
                continue
            info["functions"].append(extract_fn(node))
    return info


def extract_fn(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict:
    sig, missing, has_ret = signature(node)
    return {
        "name": node.name,
        "line": node.lineno,
        "async": isinstance(node, ast.AsyncFunctionDef),
        "sig": sig,
        "doc": first_paragraph(ast.get_docstring(node)),
        "unannotated_params": missing,
        "return_annotated": has_ret,
        "deprecated": any(is_named(d, "deprecated") for d in node.decorator_list),
    }


def extract_class(node: ast.ClassDef) -> dict:
    methods, props = [], []
    ctor = None
    for item in node.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(is_named(d, "overload") for d in item.decorator_list):
            continue
        if item.name == "__init__":
            ctor = extract_fn(item)
            # a missing `-> None` on __init__ is conventional, not a doc gap
            ctor["return_annotated"] = True
        elif item.name.startswith("_"):
            continue
        elif any(is_named(d, "property") for d in item.decorator_list):
            props.append(extract_fn(item))
        else:
            m = extract_fn(item)
            m["static"] = any(is_named(d, "staticmethod") for d in item.decorator_list)
            m["classmethod"] = any(is_named(d, "classmethod") for d in item.decorator_list)
            methods.append(m)
    return {
        "name": node.name,
        "line": node.lineno,
        "bases": [ast.unparse(b) for b in node.bases],
        "doc": first_paragraph(ast.get_docstring(node)),
        "ctor": ctor,
        "methods": methods,
        "properties": props,
        "deprecated": any(is_named(d, "deprecated") for d in node.decorator_list),
    }


# ---------------------------------------------------------------- rendering

def src_link(pin: str, file: str, line: int | None = None) -> str:
    url = f"{REPO_BLOB}/{pin}/{file}"
    return url + (f"#L{line}" if line else "")


def gaps_of(fn: dict) -> str:
    notes = []
    if fn["unannotated_params"]:
        notes.append("params " + ", ".join(f"`{p}`" for p in fn["unannotated_params"]) + " unannotated")
    if not fn["return_annotated"]:
        notes.append("return type unannotated")
    return "; ".join(notes)


def fn_bullet(fn: dict, pin: str, file: str) -> str:
    flags = []
    if fn["async"]:
        flags.append("async")
    if fn.get("static"):
        flags.append("staticmethod")
    if fn.get("classmethod"):
        flags.append("classmethod")
    if fn.get("deprecated"):
        flags.append("deprecated in source")
    head = f"- `{fn['sig']}`"
    if flags:
        head += f" *({', '.join(flags)})*"
    head += f" — [line {fn['line']}]({src_link(pin, file, fn['line'])})"
    body = []
    if fn["doc"]:
        body.append(f"  {esc(fn['doc'])}")
    gaps = gaps_of(fn)
    if gaps:
        body.append(f"  *Annotation gaps in source: {gaps}.*")
    return "\n".join([head] + body)


def render_class(c: dict, pin: str, file: str, h: str) -> list[str]:
    out = [f"{h} `{c['name']}`", ""]
    bases = f"({', '.join(f'`{b}`' for b in c['bases'])})" if c["bases"] else ""
    dep = " **Deprecated in source.**" if c["deprecated"] else ""
    out.append(f"*class* — [line {c['line']}]({src_link(pin, file, c['line'])}) {bases}{dep}")
    out.append("")
    if c["doc"]:
        out += [esc(c["doc"]), ""]
    if c["ctor"]:
        out.append(f"Constructor: `{c['ctor']['sig'].replace('__init__', c['name'], 1)}`")
        gaps = gaps_of(c["ctor"])
        if gaps:
            out.append(f"*Annotation gaps in source: {gaps}.*")
        out.append("")
    if c["properties"]:
        out.append("Properties:")
        out.append("")
        for p in c["properties"]:
            ret = p["sig"].split(" -> ", 1)[1] if " -> " in p["sig"] else None
            line = f"- `{p['name']}`"
            if ret:
                line += f" → `{ret}`"
            else:
                line += " *(type not annotated in source)*"
            if p["doc"]:
                line += f" — {esc(p['doc'])}"
            out.append(line)
        out.append("")
    if c["methods"]:
        out.append("Methods (defined on this class; inherited members not listed):")
        out.append("")
        for m in c["methods"]:
            out.append(fn_bullet(m, pin, file))
        out.append("")
    if not (c["ctor"] or c["methods"] or c["properties"] or c["doc"]):
        out += ["No public methods or docstring in source.", ""]
    return out


def render_module_body(name: str, info: dict, pin: str, h: str) -> list[str]:
    out = []
    out.append(
        f"Source: [`{info['file']}`]({src_link(pin, info['file'])}) at commit `{pin[:12]}`."
    )
    out.append("")
    if info["doc"]:
        out += [esc(info["doc"]), ""]
    else:
        out += ["No module docstring in source.", ""]
    if info["all"]:
        out.append("Declared exports (`__all__`): " + ", ".join(f"`{s}`" for s in info["all"]))
        out.append("")
    for c in info["classes"]:
        out += render_class(c, pin, info["file"], h)
    for f in info["functions"]:
        out.append(f"{h} `{f['name']}()`")
        out.append("")
        out.append(f"*{'async ' if f['async'] else ''}function* — [line {f['line']}]({src_link(pin, info['file'], f['line'])})")
        out.append("")
        out.append(f"`{f['sig']}`")
        out.append("")
        if f["doc"]:
            out += [esc(f["doc"]), ""]
        gaps = gaps_of(f)
        if gaps:
            out += [f"*Annotation gaps in source: {gaps}.*", ""]
    return out


def front_matter(doc_id: str, title: str, label: str, desc: str, pos: int) -> str:
    desc = desc or f"Public API surface of {title}, generated from source."
    desc = desc.replace('"', "'")
    if len(desc) > 200:
        desc = desc[:197] + "..."
    return (
        "---\n"
        f"id: {doc_id}\n"
        f'title: "{title}"\n'
        f'sidebar_label: "{label}"\n'
        f"sidebar_position: {pos}\n"
        f'description: "{desc}"\n'
        "# diataxis: reference\n"
        "---\n\n"
    )


FOOTER = (
    "\n---\n\n"
    "*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; "
    "regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type "
    "annotations are absent in source, this page says so rather than guessing.*\n"
)


# ---------------------------------------------------------------- assembly

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", required=True, help="full commit SHA the pages are generated from")
    ap.add_argument("--repo-root", default=".", type=Path)
    ap.add_argument("--out", default=Path("docs/docs/reference/package"), type=Path)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="optional path for a JSON manifest of generated pages and byte sizes")
    args = ap.parse_args()
    repo_root = args.repo_root.resolve()
    pin = args.pin

    modules = discover_modules(repo_root)
    data = {name: extract(py, repo_root) for name, py in modules.items()}

    # page assignment: <=3 dotted components -> own page; deeper folds in
    pages: dict[str, list[str]] = {}
    for name in data:
        key = ".".join(name.split(".")[:3])
        pages.setdefault(key, []).append(name)

    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {"pin": pin, "pages": {}, "totals": {}}

    subpackages = sorted({n.split(".")[1] for n in data if "." in n})

    # subpackage _category_.json + index pages
    for i, sub in enumerate(subpackages, start=1):
        d = out_root / sub
        d.mkdir(exist_ok=True)
        (d / "_category_.json").write_text(
            json.dumps({"label": f"mellea.{sub}", "position": i, "collapsed": True}) + "\n",
            encoding="utf-8",
        )

    for key, members in sorted(pages.items()):
        parts = key.split(".")
        info = data[key]
        if len(parts) == 1:  # mellea -> handled by root index below
            continue
        if len(parts) == 2:  # subpackage index page
            path = out_root / parts[1] / "index.md"
            doc_id, label, pos = "index", "Overview", 0
        else:
            path = out_root / parts[1] / f"{parts[2]}.md"
            siblings = sorted(p.split(".")[2] for p in pages if p.startswith(f"{parts[0]}.{parts[1]}.") and len(p.split(".")) == 3)
            doc_id, label, pos = parts[2], parts[2], siblings.index(parts[2]) + 1
        body = [front_matter(doc_id, key, label, data[key]["doc"], pos)]
        body += render_module_body(key, info, pin, "##")
        # folded deeper modules
        for member in sorted(members):
            if member == key:
                continue
            body += ["---", "", f"## Module `{member}`", ""]
            body += render_module_body(member, data[member], pin, "###")
        if len(parts) == 2:
            # module listing on the subpackage index
            children = sorted(p for p in pages if p.startswith(key + ".") and len(p.split(".")) == 3)
            if children:
                body += ["## Modules", ""]
                for ch in children:
                    leaf = ch.split(".")[2]
                    summary = esc(data[ch]["doc"]) or "No module docstring in source."
                    body.append(f"- [`{ch}`]({leaf}.md) — {summary}")
                body.append("")
        text = "".join(body[0]) + "\n".join(body[1:]) + FOOTER
        path.write_text(text, encoding="utf-8", newline="\n")
        manifest["pages"][str(path.relative_to(out_root)).replace("\\", "/")] = {
            "modules": members,
            "bytes": path.stat().st_size,
        }

    # root index
    totals = {
        "public_modules": len(data),
        "public_classes": sum(len(m["classes"]) for m in data.values()),
        "public_functions": sum(len(m["functions"]) for m in data.values()),
    }
    totals["public_symbols"] = totals["public_classes"] + totals["public_functions"]
    root = [front_matter(
        "index", "Package map", "Overview",
        "A source-pinned, machine-generated map of mellea's public import surface: "
        "every public module, class, and function linked to its source line.", 0)]
    root += [
        "A machine-generated map of `mellea`'s **public import surface**, pinned to commit",
        f"[`{pin[:12]}`]({REPO_BLOB.rsplit('/', 1)[0]}/commit/{pin}):",
        f"**{totals['public_modules']} public modules** exposing **{totals['public_classes']} public classes**",
        f"and **{totals['public_functions']} public module-level functions**, every one linked to its",
        "source line. Where the source lacks type annotations, pages say so explicitly rather than",
        "inventing types (see issue",
        "[#1177](https://github.com/generative-computing/mellea/issues/1177)).",
        "",
        "Counting method: a module is public when no component of its dotted path starts with `_`;",
        "a symbol is public when it is a top-level `class`/`def` without a leading underscore.",
        "Everything below is derived from the AST of the pinned source — no imports, no inference.",
        "",
        "## Subpackages",
        "",
    ]
    for sub in subpackages:
        info = data.get(f"mellea.{sub}")
        summary = esc(info["doc"]) if info and info["doc"] else "No package docstring in source."
        n = sum(1 for m in data if m == f"mellea.{sub}" or m.startswith(f"mellea.{sub}."))
        root.append(f"- [`mellea.{sub}`]({sub}/index.md) — {summary} *({n} public modules)*")
    root.append("")
    if data["mellea"]["all"]:
        root += [
            "## Root exports",
            "",
            f"`mellea/__init__.py` declares `__all__` = "
            + ", ".join(f"`{s}`" for s in data["mellea"]["all"])
            + f" ([source]({src_link(pin, data['mellea']['file'])})).",
            "",
        ]
    idx = out_root / "index.md"
    idx.write_text("".join(root[0]) + "\n".join(root[1:]) + FOOTER, encoding="utf-8", newline="\n")
    manifest["pages"]["index.md"] = {"modules": ["mellea"], "bytes": idx.stat().st_size}
    manifest["totals"] = totals

    if args.manifest is not None:
        args.manifest.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(totals))
    print(f"pages: {len(manifest['pages'])} -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
