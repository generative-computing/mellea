import textwrap

def wrapped(text: str | Iterable[str], width:int=100, fn : Callable = print):
    """
    Print a text string in a reasonable screen width (100).
    Swapping fn with a custom function (e.g. yp, rp, ....) is handy.
    """
    if not isinstance(text, str):
        for elem in text:
            for subline in textwrap.wrap(str(elem), width=width, initial_indent="* ", subsequent_indent="  "):
                fn(subline)
    else:
        for line in text.split("\n"):
            for subline in textwrap.wrap(line, width=width):
                fn(subline)

import mellea
from mellea.stdlib.sampling import (
    RejectionSamplingStrategy,
    RepairSamplingStrategy,
)
from mellea.stdlib.requirement import (
    Requirement,
)
from mellea.stdlib.base import (
    LinearContext,
)

from mellea.backends import ModelOption

import matplotlib.colors as mcolors
import colorsys

def css_to_lightness(name: str):
    """Convert a CSS color name to HSL (h, s, l). h in [0, 360], s/l in [0, 100]."""
    # matplotlib gives us RGB in [0, 1]
    rgb = mcolors.to_rgb(name)  # tuple of floats
    # convert to HLS (colorsys uses h, l, s)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return l

def check_sorted(ctx):
    colors = [s.strip() for s in ctx.last_output().value.split(",")]
    lightnesses = map(css_to_lightness, colors)
    colors_sorted = list(map(lambda x: x[0], sorted(zip(colors, lightnesses), key=lambda x: x[1])))
    print("ground truth: ",colors_sorted)
    return colors == colors_sorted


def check_sorted12(ctx):
    colors = [s.strip() for s in ctx.last_output().value.split(",")]
    lightnesses = list(map(css_to_lightness, colors))
    return lightnesses[0] < lightnesses[1]


def check_sorted23(ctx):
    colors = [s.strip() for s in ctx.last_output().value.split(",")]
    lightnesses = list(map(css_to_lightness, colors))
    return lightnesses[1] < lightnesses[2]


def check_sorted34(ctx):
    colors = [s.strip() for s in ctx.last_output().value.split(",")]
    lightnesses = list(map(css_to_lightness, colors))
    return lightnesses[2] < lightnesses[3]



import re

def check_last_line(ctx) -> bool:
    text: str = ctx.last_output().value
    # split into lines, drop empty lines at the end
    lines = text.rstrip().splitlines()
    if not lines:
        return False
    last = lines[-1].strip()
    if not last:
        return False
    # regex: one or more "words" separated by commas (allowing spaces)
    pattern = r'^\s*[^,]+(\s*,\s*[^,]+)*\s*$'
    return bool(re.match(pattern, last))

def check_markdown(ctx) -> bool:
    text: str = ctx.last_output().value
    # Patterns for common markdown styles
    markdown_patterns = [
        r'\*{1,2}[^*]+\*{1,2}',   # *italic* or **bold**
        r'_{1,2}[^_]+_{1,2}',     # _italic_ or __bold__
        r'`[^`]+`',               # `inline code`
        r'~~[^~]+~~',             # ~~strikethrough~~
        r'^#+\s',                 # # heading
        r'^>\s',                  # > blockquote
        r'!\[[^\]]*\]\([^)]+\)',  # ![alt](url) image
        r'\[[^\]]+\]\([^)]+\)',   # [text](url) link
    ]

    for pat in markdown_patterns:
        if re.search(pat, text, re.MULTILINE):
            return False
    return True


class Test:
    m = mellea.start_session(
        backend_name="ollama",
        model_id="qwen3:1.7b",
        model_options={ModelOption.THINKING:False},
        ctx=LinearContext(),
    )

    @pytest.mark.xfail(reason="this task is difficult")
    def test_color_sort() -> str:
        ans = m.instruct(
            f"Sort these colors by the increasing order of lightness in HSL scale. colors: lavender, orange, purple, blue.",
            requirements=[
                "Be succinct and simply return the answer in the last line without explaining the steps.",
                Requirement("Do not use markdown or any other plain-text markup format.", validation_fn=check_markdown),
                Requirement("Format the last line as a comma separated list without spaces", validation_fn=check_last_line),
                Requirement("the first output has less lightness than the second output.", validation_fn=check_sorted12),
                Requirement("the second output has less lightness than the third output.", validation_fn=check_sorted23),
                Requirement("the third output has less lightness than the fourth output.", validation_fn=check_sorted34),
                Requirement("the output is sorted by the lightness in the hsl model.", validation_fn=check_sorted),
            ],
            strategy=RepairSamplingStrategy(loop_budget=3),
            return_sampling_results=True,

        )

        assert ans.success

if __name__ == "__main__":
    # import fattrace
    # fattrace.install()
    pytest.main([__file__])
