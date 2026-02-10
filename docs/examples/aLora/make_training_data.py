#!/usr/bin/env python3
import argparse
import json
import sys

try:
    import jsonschema
except ImportError as e:
    raise SystemExit(
        "Missing dependency: jsonschema\nInstall with: pip install jsonschema"
    ) from e


RESPONSE_SCHEMA: dict = {
    "title": "Stembolt Detective Part",
    "type": "object",
    "properties": {
        "defective_part": {
            "title": "name of defective part, or unknown",
            "type": "string",
        },
        "diag_likelihood": {
            "title": "likelihood of correct detective part identification",
            "type": "number",
        },
    },
    "required": ["defective_part", "diag_likelihood"],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert raw JSONL into {item, label(json-string)} format with schema validation."
    )
    p.add_argument("input_jsonl", help="Path to input JSONL")
    p.add_argument(
        "-o", "--output-jsonl", default="-", help="Output JSONL path (default: stdout)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_fh = (
        sys.stdout
        if args.output_jsonl == "-"
        else open(args.output_jsonl, "w", encoding="utf-8")
    )

    with open(args.input_jsonl, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)

            # Basic checks
            for key in ("item", "defective_part", "diag_likelihood"):
                if key not in obj:
                    raise ValueError(f"Line {lineno}: missing '{key}'")

            label_obj = {
                "defective_part": obj["defective_part"],
                "diag_likelihood": obj["diag_likelihood"],
            }

            # Validate against schema
            jsonschema.validate(instance=label_obj, schema=RESPONSE_SCHEMA)

            # Serialize label to JSON string
            label_str = json.dumps(label_obj, ensure_ascii=False, separators=(",", ":"))

            out_obj = {"item": obj["item"], "label": label_str}

            out_fh.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    if out_fh is not sys.stdout:
        out_fh.close()


if __name__ == "__main__":
    main()
