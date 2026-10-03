"""Build arXiv submission packages: <project>/arxiv/<name>.tar.gz

Each package is self-contained: main.tex with figure paths rewritten to the package root,
numbers.tex, refs.bib, the compiled main.bbl (arXiv does not run BibTeX) and the figure PDFs.
The package is compiled once in a temporary copy to verify it builds on its own.
"""
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEX = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "TinyTeX", "bin", "windows")
PROJECTS = {"": "gulmez_snake_connectome"}  # this repository holds one project


def pdflatex(cwd, *args):
    exe = os.path.join(TEX, "pdflatex.exe") if os.path.exists(TEX) else "pdflatex"
    return subprocess.run([exe, "-interaction=nonstopmode", *args, "main.tex"], cwd=cwd,
                          capture_output=True, text=True)


def build(project, name):
    paper = os.path.join(REPO, project, "paper")
    out_dir = os.path.join(REPO, project, "arxiv")
    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp()
    tex = open(os.path.join(paper, "main.tex"), encoding="utf-8").read()
    figs = re.findall(r"\{\.\./figures/([^}]+)\}", tex)
    tex = tex.replace("../figures/", "")
    open(os.path.join(tmp, "main.tex"), "w", encoding="utf-8").write(tex)
    for f in ["numbers.tex", "refs.bib"]:
        shutil.copy(os.path.join(paper, f), tmp)
    for f in set(figs):
        shutil.copy(os.path.join(REPO, project, "figures", f), tmp)
    # compile inside the package so the shipped .bbl matches it
    pdflatex(tmp)
    bib = os.path.join(TEX, "bibtex.exe") if os.path.exists(TEX) else "bibtex"
    subprocess.run([bib, "main"], cwd=tmp, capture_output=True, text=True)
    pdflatex(tmp)
    res = pdflatex(tmp)
    pages = re.search(r"Output written on main\.pdf \((\d+) pages", res.stdout)
    bad = [l for l in res.stdout.splitlines() if l.startswith("!") or "undefined" in l.lower()]
    for junk in ("main.aux", "main.log", "main.blg", "main.out", "main.pdf"):  # keep figure PDFs
        path = os.path.join(tmp, junk)
        if os.path.exists(path):
            os.remove(path)
    tar_path = os.path.join(out_dir, name + ".tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in sorted(os.listdir(tmp)):
            tar.add(os.path.join(tmp, f), arcname=f)
    shutil.rmtree(tmp)
    size_kb = os.path.getsize(tar_path) / 1024
    print(f"{project}: {tar_path} ({size_kb:.0f} KB), {pages.group(1) if pages else '?'} pages, "
          f"{len(set(figs))} figures, problems: {bad if bad else 'none'}")
    return not bad


if __name__ == "__main__":
    ok = all(build(p, n) for p, n in PROJECTS.items())
    sys.exit(0 if ok else 1)
