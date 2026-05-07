#!/usr/bin/env python3
"""Single-keystroke labeller for candidate-pair CSVs.

Reads a CSV (typically produced by joining ``nc.candidates()`` output
to the input names + any context columns), prompts for same / different
/ unsure on each row, writes the label back into the file in place.
Autosaves after every keystroke, supports undo, resumes by skipping
rows that already carry a label.

Usage
-----
    python scripts/label_pairs.py pairs.csv
    python scripts/label_pairs.py pairs.csv --hide-cols idx_a,idx_b
    python scripts/label_pairs.py pairs.csv --review   # visit labelled rows too

Keys
----
    s / 1   same
    d / 2   different
    u / 3   unsure
    b       undo previous label (restores the prior value, if any)
    q       save and quit
"""

from __future__ import annotations

import argparse
import csv
import sys
import termios
import tty
from pathlib import Path

KEYS = {
    "s": "same",
    "1": "same",
    "d": "different",
    "2": "different",
    "u": "unsure",
    "3": "unsure",
}

CTRL_C = "\x03"
CLEAR_SCREEN = "\033[2J\033[H"


def getch() -> str:
    """Read one keystroke from stdin in raw mode; always restores tty."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, old, termios.TCSADRAIN)


def atomic_save(rows: list[dict], path: Path, fields: list[str]) -> None:
    """Write rows via tmp + rename so a crash mid-write can't truncate the
    in-progress labelling."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def render(
    rows: list[dict], i: int, fields: list[str], args: argparse.Namespace
) -> None:
    sys.stdout.write(CLEAR_SCREEN)
    r = rows[i]
    done = sum(1 for x in rows if x.get(args.label_col))
    raw = r.get(args.score_col, "")
    try:
        score = f"{float(raw):.3f}"
    except (TypeError, ValueError):
        score = raw or "-"
    print(f"[{done + 1}/{len(rows)}]  {args.score_col}={score}")
    if r.get(args.label_col):
        print(
            f"  existing label: {r[args.label_col]}  "
            f"(s/d/u overwrites; b moves to previous row)"
        )
    print()
    print(f"  A: {r.get(args.name_a_col, '')}")
    print(f"  B: {r.get(args.name_b_col, '')}")
    print()
    hide = {c.strip() for c in args.hide_cols.split(",") if c.strip()}
    skip = {args.name_a_col, args.name_b_col, args.score_col, args.label_col} | hide
    for c in fields:
        if c in skip:
            continue
        v = r.get(c, "")
        if v not in ("", None):
            print(f"  {c}: {v}")
    print()
    print("  s/1=same  d/2=different  u/3=unsure  b=back  q=quit")
    sys.stdout.flush()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("path", type=Path, help="CSV file to label (modified in place)")
    p.add_argument("--name-a-col", default="name_a")
    p.add_argument("--name-b-col", default="name_b")
    p.add_argument("--score-col", default="score")
    p.add_argument("--label-col", default="label")
    p.add_argument(
        "--hide-cols",
        default="",
        help="comma-separated columns to hide from the context display",
    )
    p.add_argument(
        "--review",
        action="store_true",
        help="visit rows that already have a label (default: skip them)",
    )
    args = p.parse_args()

    if not args.path.exists():
        sys.exit(f"error: {args.path} not found")

    with args.path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"error: {args.path} has no rows")

    fields = list(rows[0].keys())
    if args.label_col not in fields:
        fields.append(args.label_col)
    for r in rows:
        r.setdefault(args.label_col, "")

    for col in (args.name_a_col, args.name_b_col):
        if col not in fields:
            sys.exit(
                f"error: column '{col}' not in {args.path}\n"
                f"  available: {', '.join(fields)}\n"
                f"  pass --name-a-col / --name-b-col to override"
            )

    # Defer the TTY check until after the file/column validation so a typo'd
    # path or missing column still gives the informative error first.
    if not sys.stdin.isatty():
        sys.exit("error: stdin is not a TTY — this tool needs a real terminal")

    # history entries: (row_idx, label_value_before_we_changed_it).
    # `b` restores prior state instead of just clearing — important in --review
    # mode so an accidental keystroke doesn't destroy the original label.
    history: list[tuple[int, str]] = []
    i = 0
    quit_requested = False

    while i < len(rows):
        if not args.review and rows[i].get(args.label_col):
            i += 1
            continue
        render(rows, i, fields, args)
        try:
            ch = getch()
        except KeyboardInterrupt:
            ch = "q"
        if ch in ("q", CTRL_C):
            quit_requested = True
            break
        if ch == "b":
            if history:
                prev_i, prev_label = history.pop()
                i = prev_i
                rows[i][args.label_col] = prev_label
                atomic_save(rows, args.path, fields)
            continue
        if ch in KEYS:
            old = rows[i].get(args.label_col, "") or ""
            rows[i][args.label_col] = KEYS[ch]
            history.append((i, old))
            atomic_save(rows, args.path, fields)
            i += 1
            continue
        # unknown key → re-render with no state change

    atomic_save(rows, args.path, fields)
    labeled = sum(1 for x in rows if x.get(args.label_col))
    suffix = "quit early" if quit_requested else "all done"
    print(f"\nsaved — {labeled}/{len(rows)} labeled ({suffix})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
