from __future__ import annotations

import json
import sys
from pathlib import Path


def sanitize_notebook(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False

    for cell in nb.get("cells", []):
        if "metadata" not in cell or cell["metadata"] is None:
            cell["metadata"] = {}
            changed = True

        for out in cell.get("outputs", []) or []:
            out_type = out.get("output_type")

            if out_type in {"display_data", "execute_result"}:
                if "metadata" not in out or out["metadata"] is None:
                    out["metadata"] = {}
                    changed = True

            if out_type == "stream":
                if "name" not in out or out["name"] is None:
                    out["name"] = "stdout"
                    changed = True

    if changed:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv[1:] if a.strip()]
    if not paths:
        print("Usage: sanitize_notebooks.py NOTEBOOK.ipynb [NOTEBOOK2.ipynb ...]", file=sys.stderr)
        return 2

    any_changed = False
    for p in paths:
        if not p.exists():
            print(f"Missing: {p}", file=sys.stderr)
            return 2
        changed = sanitize_notebook(p)
        any_changed = any_changed or changed
        print(f"{'Sanitized' if changed else 'OK'}: {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

