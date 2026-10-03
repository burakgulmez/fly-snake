"""Verify every bib entry against Crossref (independent of OpenAlex).

For each entry in each paper's refs.bib: resolve the DOI through api.crossref.org and compare
title, year, first author and container (journal / proceedings). Prints a report; exits non-zero
if any entry fails to resolve or disagrees.
"""
import json
import re
import sys
import time
import os
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECTS = [""]  # this repository holds one project
MAILTO = "burak.gulmez@ucd.ie"


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def parse_bib(path):
    entries = []
    for block in open(path, encoding="utf-8").read().split("@")[1:]:
        kind, rest = block.split("{", 1)
        key = rest.split(",", 1)[0].strip()
        fields = dict(re.findall(r"(\w+)\s*=\s*\{(.*?)\}\s*(?:,\s*\n|\n\})", block, re.S))
        if "title" not in fields:  # title is wrapped in an extra brace pair
            m = re.search(r"title\s*=\s*\{\{(.*?)\}\}", block, re.S)
            if m:
                fields["title"] = m.group(1)
        entries.append((key, kind.strip(), fields))
    return entries


def datacite(doi):
    """arXiv and other DataCite DOIs are not in Crossref."""
    url = f"https://api.datacite.org/dois/{urllib.parse.quote(doi)}"
    req = urllib.request.Request(url, headers={"User-Agent": f"fly-connectome-games (mailto:{MAILTO})"})
    with urllib.request.urlopen(req, timeout=60) as r:
        a = json.load(r)["data"]["attributes"]
    return {"title": [a["titles"][0]["title"]], "issued": {"date-parts": [[a.get("publicationYear")]]},
            "container-title": [a.get("publisher") or ""],
            "author": [{"family": c["name"].split(",")[0]} for c in a.get("creators", [])]}


def crossref(doi):
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}?mailto={MAILTO}"
    req = urllib.request.Request(url, headers={"User-Agent": f"fly-connectome-games (mailto:{MAILTO})"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["message"]


import urllib.parse  # noqa: E402

problems = []
for proj in PROJECTS:
    path = os.path.join(REPO, proj, "paper", "refs.bib")
    print(f"\n=== {proj}")
    for key, kind, f in parse_bib(path):
        doi = f.get("doi")
        if not doi:
            problems.append((proj, key, "no DOI"))
            print(f"  [!] {key}: no DOI")
            continue
        try:
            m = datacite(doi) if doi.startswith("10.48550/") else crossref(doi)
        except Exception as e:
            problems.append((proj, key, f"DOI did not resolve: {e}"))
            print(f"  [!] {key}: DOI {doi} did not resolve ({e})")
            continue
        cr_title = (m.get("title") or [""])[0]
        cr_year = str((m.get("issued", {}).get("date-parts") or [[None]])[0][0])
        cr_container = (m.get("container-title") or [""])[0]
        cr_first = ((m.get("author") or [{}])[0].get("family") or "")
        first = f.get("author", "").split(" and ")[0]
        bib_first = first.split(",")[0].strip() if "," in first else (first.split()[-1] if first else "")
        issues = []
        if norm(cr_title) != norm(f.get("title")):
            issues.append(f"title: bib='{f.get('title')}' vs crossref='{cr_title}'")
        if cr_year != str(f.get("year")):
            issues.append(f"year: bib={f.get('year')} vs crossref={cr_year}")
        if norm(bib_first) not in norm(cr_first) and norm(cr_first) not in norm(bib_first):
            issues.append(f"first author: bib='{bib_first}' vs crossref='{cr_first}'")
        venue = f.get("journal") or f.get("booktitle") or ("arXiv" if f.get("howpublished") else "")
        if venue and norm(venue)[:12] not in norm(cr_container) and norm(cr_container)[:12] not in norm(venue):
            issues.append(f"venue: bib='{venue}' vs crossref='{cr_container}'")
        status = "OK " if not issues else "[!]"
        print(f"  {status} {key}: {cr_title[:60]} ({cr_year}, {cr_container[:35]})")
        for i in issues:
            print(f"        - {i}")
            problems.append((proj, key, i))
        time.sleep(0.2)

print("\n" + ("ALL ENTRIES VERIFIED" if not problems else f"{len(problems)} PROBLEM(S)"))
sys.exit(1 if problems else 0)
