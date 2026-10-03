"""Write each project's paper/refs.bib from publisher metadata.

Works were found through the OpenAlex API (tools/fetch_refs.py -> refs_cache.json). The bibliographic
fields written here (authors, title, year, journal, volume, issue, pages) are then taken from Crossref,
i.e. from the metadata the publisher deposited for that DOI, and from DataCite for arXiv DOIs.
Nothing is typed by hand. tools/verify_refs.py re-checks every entry against the same registries.
"""
import json
import os
import re
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
cache = json.load(open(os.path.join(HERE, "refs_cache.json"), encoding="utf-8"))
REG_CACHE = os.path.join(HERE, "registry_cache.json")
registry = json.load(open(REG_CACHE, encoding="utf-8")) if os.path.exists(REG_CACHE) else {}
MAILTO = "burak.gulmez@ucd.ie"
MAX_AUTHORS = 10

# project -> list of (bibtex key, cache key, candidate index)
SELECTION = {
    "fly-snake": [
        ("dorkenwald2024", "flywire_connectome", 0),
        ("schlegel2024", "flywire_annotation", 0),
        ("eckstein2024", "nt_prediction", 0),
        ("wu2016", "lc_types", 0),
        ("lappalainen2024", "connectome_constrained", 0),
        ("jaeger2004", "reservoir_esn", 0),
        ("hansen2001", "cmaes", 0),
        ("mnih2015", "dqn", 0),
        ("wei2018", "snake_rl", 0),
        ("yan2017", "elegans_dynamics", 0),
    ],
    "fly-flappy": [
        ("dorkenwald2024", "flywire_connectome", 0),
        ("eckstein2024", "nt_prediction", 0),
        ("matsliah2024", "optic_lobe", 0),
        ("klapoetke2017", "lplc2", 0),
        ("vonreyn2017", "giant_fiber", 0),
        ("shiu2024", "shiu_model", 0),
        ("maass2002", "liquid_state", 0),
        ("hansen2001", "cmaes", 0),
        ("salimans2017", "openai_es", 0),
        ("thurler2021", "flappy_rl", 1),
    ],
    "fly-pong": [
        ("dorkenwald2024", "flywire_connectome", 0),
        ("eckstein2024", "nt_prediction", 0),
        ("scheffer2020", "hemibrain", 0),
        ("hulse2021", "cx_hulse", 0),
        ("lyu2021", "pfn_vector", 0),
        ("westeinde2024", "pfl3_steering", 0),
        ("pires2024", "cx_navigation", 0),
        ("rayshubskiy2025", "dna02_steering", 2),
        ("hansen2001", "cmaes", 0),
        ("stanley2002", "neat", 0),
        ("wangchen2024", "flygym", 0),
    ],
}

SELECTION = {"": SELECTION["fly-snake"]}  # this repository holds one project


def fetch(doi):
    """Publisher metadata for a DOI: Crossref, or DataCite for arXiv DOIs."""
    if doi in registry:
        return registry[doi]
    headers = {"User-Agent": f"fly-connectome-games (mailto:{MAILTO})"}
    if doi.startswith("10.48550/"):
        req = urllib.request.Request("https://api.datacite.org/dois/" + urllib.parse.quote(doi), headers=headers)
        a = json.load(urllib.request.urlopen(req, timeout=60))["data"]["attributes"]
        rec = dict(source="datacite", title=a["titles"][0]["title"], year=a.get("publicationYear"),
                   authors=[[c.get("givenName") or c["name"].split(",")[-1].strip(),
                             c.get("familyName") or c["name"].split(",")[0].strip()] for c in a["creators"]],
                   container="", volume="", issue="", pages="", type="preprint")
    else:
        req = urllib.request.Request("https://api.crossref.org/works/" + urllib.parse.quote(doi) + "?mailto=" + MAILTO,
                                     headers=headers)
        m = json.load(urllib.request.urlopen(req, timeout=60))["message"]
        rec = dict(source="crossref", title=re.sub(r"\s+", " ", m["title"][0]).strip(),
                   year=(m.get("issued", {}).get("date-parts") or [[None]])[0][0],
                   authors=[[a.get("given", ""), a.get("family", a.get("name", ""))] for a in m.get("author", [])],
                   container=(m.get("container-title") or [""])[0], volume=m.get("volume", ""),
                   issue=m.get("issue", ""), pages=m.get("page", "") or m.get("article-number", ""), type=m.get("type", ""))
    registry[doi] = rec
    time.sleep(0.2)
    return rec


def bib_entry(key, doi):
    r = fetch(doi)
    names = [f"{fam}, {giv}".strip(", ") for giv, fam in r["authors"]]
    authors = " and ".join(names[:MAX_AUTHORS] + (["others"] if len(names) > MAX_AUTHORS else []))
    title = re.sub(r"<[^>]+>", "", r["title"])
    if r["type"] == "preprint":
        kind, fields = "misc", [("howpublished", "arXiv:" + doi.split("arxiv.")[1])]
    elif r["type"] in ("proceedings-article", "book-chapter"):
        kind, fields = "inproceedings", [("booktitle", r["container"])]
    else:
        kind, fields = "article", [("journal", r["container"])]
    fields = [("author", authors), ("title", "{" + title + "}"), ("year", r["year"])] + fields
    if r["volume"]:
        fields.append(("volume", r["volume"]))
    if r["issue"]:
        fields.append(("number", r["issue"]))
    if r["pages"]:
        fields.append(("pages", r["pages"].replace("-", "--")))
    fields.append(("doi", doi))
    body = ",\n  ".join(f"{k} = {{{v}}}" for k, v in fields if v)
    return f"@{kind}{{{key},\n  {body}\n}}\n"


for proj, sel in SELECTION.items():
    out = ["% Works found via the OpenAlex API; bibliographic fields from Crossref / DataCite (tools/make_bib.py).",
           "% Verified by tools/verify_refs.py. Do not edit by hand: re-run the scripts instead.", ""]
    for key, ckey, idx in sel:
        doi = cache[ckey]["candidates"][idx]["doi"].replace("https://doi.org/", "")
        out.append(bib_entry(key, doi))
    path = os.path.join(REPO, proj, "paper", "refs.bib")
    open(path, "w", encoding="utf-8").write("\n".join(out))
    print(path, len(sel), "entries")
json.dump(registry, open(REG_CACHE, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
