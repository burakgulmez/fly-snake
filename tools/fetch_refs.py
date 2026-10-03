"""Fetch reference metadata and abstracts from the OpenAlex API.

usage: python tools/fetch_refs.py
Writes tools/refs_cache.json (metadata + abstract text) for the queries below.
Only works that OpenAlex actually returns are stored; nothing is invented.
"""
import json
import os
import re
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "refs_cache.json")
MAILTO = "burak.gulmez@ucd.ie"

QUERIES = {
    # key: (search string, optional filter on publication year)
    "flywire_connectome": "Neuronal wiring diagram of an adult brain",
    "flywire_annotation": "Whole-brain annotation and multi-connectome cell typing of Drosophila",
    "nt_prediction": "Neurotransmitter classification from electron microscopy images at synaptic sites in Drosophila",
    "optic_lobe": "Neuronal parts list and wiring diagram for a visual system",
    "hemibrain": "A connectome and analysis of the adult Drosophila central brain",
    "connectome_constrained": "Connectome-constrained networks predict neural activity across the fly visual system",
    "cmaes": "Completely derandomized self-adaptation in evolution strategies",
    "openai_es": "Evolution strategies as a scalable alternative to reinforcement learning",
    "reservoir_esn": "Harnessing nonlinearity: predicting chaotic systems and saving energy in wireless communication",
    "liquid_state": "Real-time computing without stable states: a new framework for neural computation based on perturbations",
    "dqn": "Human-level control through deep reinforcement learning",
    "lplc2": "Ultra-selective looming detection from radial motion opponency",
    "giant_fiber": "Feature integration drives probabilistic behavior in the Drosophila escape response",
    "lc_types": "Visual projection neurons in the Drosophila lobula link feature detection to distinct behavioral programs",
    "dnp01_escape": "Escape behavior in Drosophila",
    "cx_hulse": "A connectome of the Drosophila central complex reveals network motifs suitable for flexible navigation",
    "pfl3_steering": "Transforming a head direction signal into a goal-oriented steering command",
    "pfn_wind": "Neural evidence for a vector representation of goal direction in the Drosophila central complex",
    "dna02": "Descending neurons coordinate turning in Drosophila",
    "cx_navigation": "Converting an allocentric goal into an egocentric steering signal",
    "worm_connectome_model": "Connectome-based model of the Caenorhabditis elegans",
    "neuromechfly": "NeuroMechFly, a neuromechanical model of adult Drosophila melanogaster",
    "flygym": "NeuroMechFly v2 simulating embodied sensorimotor control in adult Drosophila",
}


def openalex(search):
    url = ("https://api.openalex.org/works?" + urllib.parse.urlencode(
        {"search": search, "per-page": 5, "mailto": MAILTO}))
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)["results"]


def abstract_text(w):
    inv = w.get("abstract_inverted_index")
    if not inv:
        return ""
    pos = {}
    for word, places in inv.items():
        for p in places:
            pos[p] = word
    return " ".join(pos[i] for i in sorted(pos))


def clean(w):
    return dict(
        id=w["id"], doi=w.get("doi"), title=w.get("title"), year=w.get("publication_year"),
        venue=(w.get("primary_location") or {}).get("source", {}).get("display_name") if (w.get("primary_location") or {}).get("source") else None,
        type=w.get("type"), cited_by=w.get("cited_by_count"),
        authors=[a["author"]["display_name"] for a in w.get("authorships", [])],
        volume=(w.get("biblio") or {}).get("volume"), issue=(w.get("biblio") or {}).get("issue"),
        first_page=(w.get("biblio") or {}).get("first_page"), last_page=(w.get("biblio") or {}).get("last_page"),
        abstract=abstract_text(w),
    )


def main():
    out = {}
    for key, q in QUERIES.items():
        try:
            res = openalex(q)
        except Exception as e:  # network hiccup: keep going, report at the end
            out[key] = {"query": q, "error": str(e)}
            continue
        out[key] = {"query": q, "candidates": [clean(w) for w in res]}
        print(f"{key}: {len(res)} hits -> {res[0]['title'][:80] if res else 'NONE'}")
        time.sleep(0.3)
    json.dump(out, open(CACHE, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("written", CACHE)


if __name__ == "__main__":
    main()
