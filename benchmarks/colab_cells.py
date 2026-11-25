"""Cells for the notebooks the user runs on Colab or Kaggle (make_colab_*.py): every code cell says, when it
starts and when it ends, which step it is out of how many, about how long it takes, and what to run next.

    steps = Steps()
    steps.code("Install cuVS", 3, "...python...")
    steps.write("cuvs_probe.py", source)          # a script written to disk, reported (not %%writefile)
    steps.shell("Capabilities", 5, "python -u cuvs_probe.py --capabilities")
    cells = steps.cells()                          # markdown and code cells, numbered
"""
import json


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def _code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


class Steps:
    """The notebook's cells in order; code steps are numbered once all are known."""

    def __init__(self):
        self._items = []                   # ("md", text) or ("step", title, minutes, body, optional)

    def md(self, text):
        self._items.append(("md", text))

    def code(self, title, minutes, body, optional=False):
        """A Python step."""
        self._items.append(("step", title, minutes, body.strip(), optional))

    def shell(self, title, minutes, command, optional=False):
        """A step running one shell command (unbuffered Python, so its lines appear as they come)."""
        self._items.append(("step", title, minutes, "!" + command, optional))

    def write(self, name, source):
        """A step writing a script to disk, with its size reported."""
        body = (f"source = {json.dumps(source)}\n"
                f"with open({name!r}, 'w', encoding='utf-8') as f:\n"
                f"    f.write(source)\n"
                f"print(f\"wrote {name}: {{len(source.splitlines())}} lines\")")
        self._items.append(("step", f"Write {name}", 0, body, False))

    def cells(self):
        steps = [i for i in self._items if i[0] == "step"]
        n = len(steps)
        out, k = [], 0
        for item in self._items:
            if item[0] == "md":
                out.append(md(item[1]))
                continue
            _, title, minutes, body, optional = item
            k += 1
            nxt = next((s for s in steps[k:] if not s[4]), None)
            after = f"next: step {steps.index(nxt) + 1} of {n}, {nxt[1]}" if nxt else "that was the last step"
            took = "seconds" if minutes == 0 else f"about {minutes} min"
            tag = " (optional)" if optional else ""
            head = (f"# Step {k} of {n}{tag}: {title} ({took})\n"
                    f"_t0 = __import__(\"time\").time()\n"
                    f"print(\"step {k} of {n}{tag}: {title}, {took}: started\", flush=True)\n")
            tail = (f"\nprint(f\"step {k} of {n}: {title}: done in {{(__import__('time').time() - _t0) / 60:.1f}} min; "
                    f"{after}\", flush=True)")
            out.append(_code(head + body + tail))
        return out


def notebook(cells, accelerator):
    return {"cells": cells, "metadata": {"accelerator": accelerator,
                                         "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def numbered(cells, plan):
    """The same notebook with every code cell made a numbered step: ``plan`` gives, for each code cell in order,
    (title, minutes) or (title, minutes, "optional"). A %%writefile cell becomes a step that writes the file and
    reports it; a "!command" cell stays a shell step."""
    codes = [c for c in cells if c["cell_type"] == "code"]
    if len(codes) != len(plan):
        raise ValueError(f"{len(codes)} code cells, {len(plan)} entries in the plan")
    steps, it = Steps(), iter(plan)
    for c in cells:
        text = "".join(c["source"])
        if c["cell_type"] == "markdown":
            steps.md(text)
            continue
        title, minutes, *opt = next(it)
        optional = bool(opt)
        if text.startswith("%%writefile "):
            name, _, source = text.partition("\n")
            steps.write(name.split(maxsplit=1)[1].strip(), source if source.endswith("\n") else source + "\n")
            steps._items[-1] = ("step", title, minutes, steps._items[-1][3], optional)
        else:
            steps.code(title, minutes, text, optional)
    return steps.cells()
