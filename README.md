# medication-safety-agent

A Trase OS third-party agent built with LangGraph. Ask about the medications you take, and it
checks real FDA and NLM data, then writes a short safety brief.

> "I take lisinopril for blood pressure and Advil for back pain. Anything I should know?"

```
extract -> normalize -> ┬ label        FDA label: boxed warning + drug interactions  ┐
                        ├ events       FDA adverse-event reports (co-reported)       ├-> brief
                        ├ recalls      ongoing FDA recalls                            │
                        └ plain_lang   MedlinePlus plain-language pages              ┘
```

`extract` and `brief` call the model. The four middle branches call public health APIs and run
in parallel. The graph is built at import, so Trase OS renders its topology.

## Data sources (free, no API key)

| Connection handle | Upstream | Used for |
| --- | --- | --- |
| `rxnav` | `https://rxnav.nlm.nih.gov` | NLM RxNorm: brand → ingredient (Advil → ibuprofen) |
| `openfda` | `https://api.fda.gov` | FDA drug labels, adverse-event reports (FAERS), recalls |
| `medlineplus` | `https://connect.medlineplus.gov` | NLM MedlinePlus drug pages |

## Model

`AGENT_PROVIDER=openai` (default): `gpt-4o-mini` on the platform's `openai` connection.
`AGENT_PROVIDER=vertex`: `gemini-2.5-flash` on the `vertex` connection. `AGENT_MODEL` overrides
the model name.

## Governance

Everything, including the model, goes through the Trase OS governance gateway:

```
${TRASE_EGRESS_GATEWAY_URL}/<handle>/<path>
Authorization: Bearer ${TRASE_RUN_CREDENTIAL}
```

The agent holds no vendor keys. Each connection must exist and be granted to this agent, with
policy activated, or the call is denied (`policy_deny`). The connection handles in the table
above must match exactly.

## Input

Read with `trase_os_sdk.sandbox.read_input()`:

```json
{"user_message": "I take lisinopril and Advil. Anything I should know?"}
```

Add `"drugs": ["lisinopril", "Advil"]` to skip the model's extraction step. With no input it
runs the built-in example question.

## Run locally

Without `TRASE_EGRESS_GATEWAY_URL`, the data sources call the public APIs directly:

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...        # local only; on Trase OS the gateway supplies the key
python -c "from agent.main import run; print(run())"
```

*Not medical advice: openFDA and MedlinePlus data are for information only.*
