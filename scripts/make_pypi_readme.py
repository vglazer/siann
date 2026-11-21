"""Writes PyPI's copy of the README to python/PYPI_README.md (not committed).

PyPI does not create heading anchors and cannot follow links relative to the repository, so the copy points
in-page and relative links at GitHub. The release workflow runs this with --apply, which also points
pyproject.toml's readme at the copy, just before building. Run it without --apply to preview the text.

    python scripts/make_pypi_readme.py [--apply]
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "README.md"
TARGET = ROOT / "python" / "PYPI_README.md"
REPO = "https://github.com/vglazer/siann"


def pypi_readme(text: str) -> str:
    text = re.sub(r"\]\(#", f"]({REPO}#", text)
    text = re.sub(r"\]\((?!https?://|#|mailto:)([^)]+)\)", rf"]({REPO}/blob/main/\1)", text)
    return "<!-- Generated from README.md by scripts/make_pypi_readme.py; edit README.md instead. -->\n" + text


def main() -> None:
    args = sys.argv[1:]
    if args not in ([], ["--apply"]):
        raise SystemExit("usage: make_pypi_readme.py [--apply]")
    TARGET.write_text(pypi_readme(SOURCE.read_text(encoding="utf-8")), encoding="utf-8")
    print(TARGET)
    if args == ["--apply"]:
        pyproject = ROOT / "pyproject.toml"
        text = pyproject.read_text(encoding="utf-8")
        new = text.replace('readme = "README.md"', 'readme = "python/PYPI_README.md"')
        if new == text:
            raise SystemExit('pyproject.toml has no readme = "README.md" to replace')
        pyproject.write_text(new, encoding="utf-8")
        print("pyproject.toml now uses it as the readme")


if __name__ == "__main__":
    main()
