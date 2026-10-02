"""Medication safety check, as a LangGraph graph.

    extract -> normalize -> ┬ label        (FDA boxed warning + interactions) ┐
                            ├ events       (FAERS co-reported reactions)      ├-> brief
                            ├ recalls      (ongoing FDA recalls)              │
                            └ plain_lang   (MedlinePlus pages)                ┘

`extract` and `brief` call the model; the four middle branches call public
health APIs (see sources.py) and run in parallel.

The model runs through the platform's governance gateway too. With
AGENT_PROVIDER=openai (the default) it is OpenAI on the `openai` connection;
with AGENT_PROVIDER=vertex it is Gemini on the `vertex` connection. Either way
the sandbox holds no vendor key: the run credential is the key, and the gateway
swaps in the real one.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agent import sources

log = logging.getLogger(__name__)

PROVIDER = os.environ.get("AGENT_PROVIDER", "openai")
# Graphs are built at import, which the build-time topology inspector also does,
# without a sandbox environment; this placeholder lets the model be constructed there.
_INSPECTION_ONLY = "topology-inspection-only"

DISCLAIMER = (
    "This is general information from public FDA and NLM data, not medical advice. "
    "Talk to a pharmacist or doctor before changing any medication."
)


def _model() -> BaseChatModel:
    credential = os.environ.get("TRASE_RUN_CREDENTIAL", _INSPECTION_ONLY)
    if PROVIDER == "vertex":
        from google.oauth2.credentials import Credentials
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.environ.get("AGENT_MODEL", "gemini-2.5-flash"),
            vertexai=True,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT", _INSPECTION_ONLY),
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
            credentials=Credentials(token=credential),
            base_url=os.environ.get("TRASE_VERTEX_BASE_URL"),
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.environ.get("AGENT_MODEL", "gpt-4o-mini"),
        base_url=os.environ.get("TRASE_OPENAI_BASE_URL"),
        api_key=credential,
        temperature=0,
    )


model = _model()


class DrugList(BaseModel):
    drugs: list[str] = Field(description="Every medication named in the question, as written.")


class State(TypedDict, total=False):
    question: str
    drug_names: list[str]
    drugs: list[dict[str, str]]
    labels: list[dict[str, str]]
    reactions: list[dict[str, Any]]
    recalls: list[dict[str, Any]]
    pages: dict[str, str | None]
    brief: str


def extract(state: State) -> State:
    """Pull the medication names out of the question (skipped when the input lists them)."""
    if state.get("drug_names"):
        return {}
    found = model.with_structured_output(DrugList).invoke(
        "List every medication (brand or generic) named in this question:\n\n" + state["question"]
    )
    return {"drug_names": found.drugs}


def normalize(state: State) -> State:
    drugs = [d for name in state["drug_names"] if (d := sources.normalize_drug(name))]
    log.info("normalized %s -> %s", state["drug_names"], [d["ingredient"] for d in drugs])
    return {"drugs": drugs}


def label(state: State) -> State:
    return {"labels": [sources.label_warnings(d["ingredient"]) for d in state["drugs"]]}


def events(state: State) -> State:
    ingredients = [d["ingredient"] for d in state["drugs"]]
    return {"reactions": sources.co_reported_reactions(ingredients) if len(ingredients) > 1 else []}


def recalls(state: State) -> State:
    return {"recalls": [sources.ongoing_recalls(d["ingredient"]) for d in state["drugs"]]}


def plain_lang(state: State) -> State:
    return {"pages": {d["ingredient"]: sources.medlineplus_page(d["rxcui"]) for d in state["drugs"]}}


BRIEF_PROMPT = """You are a medication-safety assistant. Using ONLY the data below, answer the
user's question in a short brief with these sections:

1. Medications: brand/name given -> ingredient.
2. Key warnings: boxed warnings, and any interaction between these medications that the FDA
   label text mentions (quote the relevant point briefly).
3. Real-world reports: the most common reactions in FDA adverse-event reports that list these
   drugs together. Say clearly these are co-reported, not proven to be caused by the drugs.
   Point out if a reaction lines up with a label warning.
4. Recalls: number of ongoing FDA recalls per ingredient, with one example reason.
5. Learn more: the MedlinePlus links.

Do not add facts that are not in the data. End with this line exactly:
{disclaimer}

Question: {question}

Data:
{data}"""


def brief(state: State) -> State:
    data = {k: state.get(k) for k in ("drugs", "labels", "reactions", "recalls", "pages")}
    prompt = BRIEF_PROMPT.format(
        disclaimer=DISCLAIMER, question=state["question"], data=json.dumps(data, indent=1)
    )
    return {"brief": model.invoke(prompt).content}


def build() -> Any:
    graph = StateGraph(State)
    for name, fn in [
        ("extract", extract),
        ("normalize", normalize),
        ("label", label),
        ("events", events),
        ("recalls", recalls),
        ("plain_lang", plain_lang),
        ("brief", brief),
    ]:
        graph.add_node(name, fn)
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "normalize")
    for branch in ("label", "events", "recalls", "plain_lang"):
        graph.add_edge("normalize", branch)
    graph.add_edge(["label", "events", "recalls", "plain_lang"], "brief")
    graph.add_edge("brief", END)
    return graph.compile()


# Built at import: the topology inspector imports agent.main and reads this graph.
safety_graph = build()
