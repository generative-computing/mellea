"""Validate SKILL.md frontmatter for agent skills."""

import json
import os
import sys

import yaml


def validate_skill(skill_path: str) -> dict:
    """Check that a skill directory has valid SKILL.md with required frontmatter keys."""
    skill_file = os.path.join(skill_path, "SKILL.md")

    if not os.path.exists(skill_file):
        return {"status": "error", "message": "Missing SKILL.md"}

    try:
        with open(skill_file) as f:
            # safe_load_all handles the --- delimiters correctly and won't
            # break on markdown horizontal rules later in the file.
            frontmatter = next(yaml.safe_load_all(f))

        if not isinstance(frontmatter, dict):
            return {"status": "error", "message": "Frontmatter is not a YAML mapping"}

        # Root-level required keys
        for key in ("name", "description"):
            if key not in frontmatter:
                return {"status": "error", "message": f"Missing root key: {key}"}

        # version lives under metadata (per skill-author guide)
        meta = frontmatter.get("metadata")
        if not isinstance(meta, dict) or "version" not in meta:
            return {
                "status": "error",
                "message": "Missing nested key: metadata.version",
            }

        return {"status": "success", "data": frontmatter}

    except yaml.YAMLError as e:
        return {"status": "error", "message": f"Invalid YAML: {e}"}
    except StopIteration:
        return {"status": "error", "message": "No YAML frontmatter found"}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 validate_skill.py <skill-directory>", file=sys.stderr)
        sys.exit(1)
    result = validate_skill(sys.argv[1])
    print(json.dumps(result))
