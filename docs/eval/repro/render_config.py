#!/usr/bin/env python3
"""Render the judge-panel template from environment variables.

Reads ``panel3.template.yaml``, substitutes every ``${VAR}`` from the process
environment (with ``judges.env`` as a fallback for keys not already exported),
and writes ``system_panel3.yaml`` next to it.

Refuses to write when a variable is missing, so a half-filled environment cannot
silently produce a config with an empty endpoint.

Usage:
    python3 render_config.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "panel3.template.yaml"
ENV_FILE = HERE / "judges.env"
OUTPUT = HERE / "system_panel3.yaml"

PLACEHOLDER = re.compile(r"\$\{([A-Z0-9_]+)\}")


def load_environment() -> dict[str, str]:
    """Return the process environment, with judges.env filling in unset keys."""
    env = dict(os.environ)
    if not ENV_FILE.exists():
        return env
    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env.setdefault(key.strip(), value.strip().strip("\"'"))
    return env


def main() -> int:
    """Render the template or explain which variables are missing."""
    if not TEMPLATE.exists():
        print(f"template not found: {TEMPLATE}", file=sys.stderr)
        return 1

    text = TEMPLATE.read_text(encoding="utf-8")
    env = load_environment()

    missing = sorted(
        {match for match in PLACEHOLDER.findall(text) if not env.get(match)}
    )
    if missing:
        print("missing required variable(s):", ", ".join(missing), file=sys.stderr)
        print(f"check your environment or {ENV_FILE.name}", file=sys.stderr)
        return 1

    rendered = PLACEHOLDER.sub(lambda match: env[match.group(1)], text)
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUTPUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
