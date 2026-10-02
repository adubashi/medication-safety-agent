"""Public health data sources, each reached through a Trase OS connection.

On the platform every call goes through the governance gateway:

    ${TRASE_EGRESS_GATEWAY_URL}/<connection handle>/<path>
    Authorization: Bearer ${TRASE_RUN_CREDENTIAL}

so the connection handles below must match the connections created in Trase OS.
Outside a sandbox (TRASE_EGRESS_GATEWAY_URL unset) the same paths are sent
straight to the public upstream, which is how the agent is tested locally.

All three upstreams are free US government APIs that need no API key.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

log = logging.getLogger(__name__)

# connection handle -> public upstream it fronts
CONNECTIONS = {
    "rxnav": "https://rxnav.nlm.nih.gov",  # NLM RxNorm: drug-name normalization
    "openfda": "https://api.fda.gov",  # FDA labels, adverse events, recalls
    "medlineplus": "https://connect.medlineplus.gov",  # NLM plain-language drug pages
}

USER_AGENT = "trase-os-medication-safety-agent (demo)"


def _get(handle: str, path: str, params: dict[str, Any] | None = None) -> Any:
    gateway = os.environ.get("TRASE_EGRESS_GATEWAY_URL")
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if gateway:
        url = f"{gateway.rstrip('/')}/{handle}{path}"
        headers["Authorization"] = f"Bearer {os.environ['TRASE_RUN_CREDENTIAL']}"
    else:
        url = f"{CONNECTIONS[handle]}{path}"
    log.info("GET %s%s via connection %r", handle, path, handle)
    response = httpx.get(url, params=params, headers=headers, timeout=30)
    if response.status_code == 404:
        return None  # openFDA answers 404 for "no matches"
    response.raise_for_status()
    return response.json()


# --- RxNorm -------------------------------------------------------------------


def normalize_drug(name: str) -> dict[str, str] | None:
    """Map a brand or misspelled name to its RxNorm ingredient, e.g. Advil -> ibuprofen."""
    match = _get("rxnav", "/REST/approximateTerm.json", {"term": name, "maxEntries": 1})
    candidates = (match or {}).get("approximateGroup", {}).get("candidate") or []
    if not candidates:
        return None
    rxcui = candidates[0]["rxcui"]
    related = _get("rxnav", f"/REST/rxcui/{rxcui}/related.json", {"tty": "IN"})
    groups = (related or {}).get("relatedGroup", {}).get("conceptGroup") or []
    ingredients = [p for g in groups for p in g.get("conceptProperties", [])]
    if not ingredients:
        props = _get("rxnav", f"/REST/rxcui/{rxcui}/properties.json") or {}
        ingredient = props.get("properties", {})
        return {"input": name, "ingredient": ingredient.get("name", name), "rxcui": rxcui}
    return {"input": name, "ingredient": ingredients[0]["name"], "rxcui": ingredients[0]["rxcui"]}


# --- openFDA ------------------------------------------------------------------


def _clip(text: str, limit: int = 1200) -> str:
    return text if len(text) <= limit else text[:limit] + " …"


def label_warnings(ingredient: str) -> dict[str, str]:
    """Boxed warning and drug-interaction text from the FDA label of a single-ingredient product."""
    found = _get(
        "openfda",
        "/drug/label.json",
        {"search": f'openfda.generic_name.exact:"{ingredient.upper()}"', "limit": 5},
    )
    labels = (found or {}).get("results") or []
    # Prefer a label whose product contains only this ingredient (skip combination products).
    labels.sort(key=lambda r: len(r.get("openfda", {}).get("generic_name", [""])[0].split(" AND ")))
    if not labels:
        return {"ingredient": ingredient, "boxed_warning": "", "drug_interactions": ""}
    label = labels[0]
    return {
        "ingredient": ingredient,
        "boxed_warning": _clip(" ".join(label.get("boxed_warning", []))),
        "drug_interactions": _clip(" ".join(label.get("drug_interactions", [])), 2000),
    }


def co_reported_reactions(ingredients: list[str], top: int = 6) -> list[dict[str, Any]]:
    """Most common reactions in FAERS reports that list ALL of these drugs (co-reported, not causal)."""
    query = " AND ".join(f'patient.drug.openfda.generic_name:"{i}"' for i in ingredients)
    found = _get(
        "openfda",
        "/drug/event.json",
        {"search": query, "count": "patient.reaction.reactionmeddrapt.exact", "limit": top},
    )
    return [{"reaction": r["term"], "reports": r["count"]} for r in (found or {}).get("results", [])]


def ongoing_recalls(ingredient: str, limit: int = 3) -> dict[str, Any]:
    """Ongoing FDA drug recalls whose product description mentions the ingredient."""
    found = _get(
        "openfda",
        "/drug/enforcement.json",
        {"search": f'product_description:"{ingredient}" AND status:"Ongoing"', "limit": limit},
    )
    if not found:
        return {"ingredient": ingredient, "total": 0, "examples": []}
    return {
        "ingredient": ingredient,
        "total": found["meta"]["results"]["total"],
        "examples": [
            {"class": r["classification"], "reason": _clip(r["reason_for_recall"], 160)}
            for r in found["results"]
        ],
    }


# --- MedlinePlus --------------------------------------------------------------


def medlineplus_page(rxcui: str) -> str | None:
    """Plain-language MedlinePlus drug page for an RxNorm concept."""
    found = _get(
        "medlineplus",
        "/service",
        {
            "mainSearchCriteria.v.cs": "2.16.840.1.113883.6.88",  # RxNorm code system
            "mainSearchCriteria.v.c": rxcui,
            "knowledgeResponseType": "application/json",
        },
    )
    entries = (found or {}).get("feed", {}).get("entry") or []
    return entries[0]["link"][0]["href"].split("?")[0] if entries else None
