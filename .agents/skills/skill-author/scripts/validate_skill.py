"""Validate SKILL.md frontmatter for agent skills."""

import json
import os
import sys

import yaml  # Ensure this is in the agent's environment


def validate_skill(skill_path):
    """Check that a skill directory has valid SKILL.md with required frontmatter keys."""
    skill_file = os.path.join(skill_path, "SKILL.md")

    if not os.path.exists(skill_file):
        return {"status": "error", "message": "Missing SKILL.md"}

    try:
        with open(skill_file) as f:
            content = f.read()
            # Split YAML frontmatter
            if content.startswith("---"):
                parts = content.split("---")
                metadata = yaml.safe_load(parts[1])

                # Validation Logic
                required_keys = ["name", "description", "version"]
                for key in required_keys:
                    if key not in metadata:
                        return {"status": "error", "message": f"Missing key: {key}"}

                return {"status": "success", "data": metadata}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Example usage: python3 validate_skill.py ./.agents/skills/new-skill
    result = validate_skill(sys.argv[1])
    print(json.dumps(result))
