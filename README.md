# EMMA Clone — Production NHS AI Voice Receptionist

> **Built in 7 days** as a production-grade demonstration of the EMMA architecture.
> 100% open source · £0 cost · zero credit card required.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange)](https://langchain-ai.github.io/langgraph/)
[![LiveKit](https://img.shields.io/badge/LiveKit-OSS-blueviolet)](https://livekit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Cost: £0](https://img.shields.io/badge/Cost-£0-brightgreen)]()

---

## Table of Contents

1. [What Is EMMA?](#what-is-emma)
2. [Architecture](#architecture)
3. [What This Demonstrates](#what-this-demonstrates)
4. [Performance Metrics](#performance-metrics)
5. [Tech Stack](#tech-stack)
6. [Project Structure](#project-structure)
7. [Quick Start](#quick-start)
8. [Running Tests](#running-tests)
9. [Running Evaluations](#running-evaluations)
10. [Safety Architecture](#safety-architecture)
11. [Multi-Tenancy](#multi-tenancy)
12. [Observability](#observability)
13. [Acknowledged Limitations](#acknowledged-limitations)

---

## What Is EMMA?

EMMA is a **multi-tenant AI voice receptionist for NHS GP surgeries**. It answers surgery-specific administrative questions over a real phone call — handling appointment enquiries, opening hours, prescription requests, and urgent signposting — while enforcing strict clinical safety constraints.

A patient dials a DDI number, LiveKit bridges the PSTN call into a WebRTC room, and an EMMA agent worker joins that room, listens, reasons, and speaks — all in under 500ms end-to-end.

EMMA is **not a medical device**. It handles administrative enquiries only. Any clinical question, or any utterance matching an emergency pattern, is immediately escalated to a human — before the LLM ever sees the text.

---

## Architecture

```
Patient Speaks
      │
      ▼
SIP Trunk (LiveKit)
      │  DDI → tenant routing
      ▼
LiveKit SIP Dispatch Rule
      │  SIPDispatchRuleIndividual
      │  → creates room "surgery_greenfield-<uuid>"
      │  → room metadata: {"tenant_id": "surgery_greenfield"}
      ▼
LiveKit Room (WebRTC)
      │
      ├── SIP Caller (audio track, 8kHz G.711 → 48kHz WebRTC)
      │
      └── EMMA Agent Worker
            │
            ├── AudioStream 16kHz → STT (Groq Whisper Large-v3)
            │         │ transcript + confidence
            │         ▼
            │   LangGraph Pipeline
            │   ┌──────────────────────────────────────────────┐
            │   │                                              │
            │   │  Safety Gate ──── ESCALATE ────────────────► │
            │   │  (keyword + semantic — NEVER LLM)            │
            │   │         │ SAFE                               │
            │   │         ▼                                    │
            │   │  Retrieval Agent (Qdrant RAG)                │
            │   │  all-MiniLM-L6-v2 + cross-encoder reranking  │
            │   │         │                                    │
            │   │         ▼                                    │
            │   │  Response Agent (Groq Llama 3.3-70b)         │
            │   │  temp=0 · NeMo input rails                   │
            │   │         │                                    │
            │   │         ▼                                    │
            │   │  Verification Agent (Chain-of-Verification)  │
            │   │  faithfulness check against retrieved context │
            │   └──────────────────────────────────────────────┘
            │         │ Verified response
            │         ▼
            ├── Kokoro TTS 16kHz → AudioSource → PSTN
            │
            └── EventBus.publish(event)
                      │ SSE fan-out
                      ▼
              Next.js 14 Dashboard (http://localhost:3000)
                  Live Monitor · Tenants · Evaluations
```

## What This Demonstrates

### Clinical Safety

- **Hardcoded emergency escalation** — 999 / emergency keywords are caught at the Safety Gate *before* the LLM sees the utterance. The LLM is structurally incapable of suppressing an emergency redirect.
- **Three-layer safety gate**: (1) exact normalised keyword matching, (2) order-preserving gap-tolerant token matching, (3) semantic similarity against curated emergency reference sentences — covering slang like *"my ticker feels like it's going to burst"*.
- **Fail-closed design** — ambiguous or empty inputs are escalated, not passed through.
- **NeMo Guardrails** on LLM input — blocks medical advice, diagnosis, or prescription recommendations before generation.
- **Chain of Verification** on every response — cross-checks LLM output against retrieved context for faithfulness before it is spoken to the patient.
- **10-minute call duration cap** — calls exceeding this are terminated with a callback message.

### Production Engineering

- **Multi-tenant Qdrant namespaces** — Surgery A and Surgery B data are physically isolated at the vector DB level. A query for Surgery A cannot retrieve Surgery B context.
- **LiveKit SIP bridge** — real WebRTC + SIP integration with TLS/SRTP. No FreeSWITCH, no `network_mode:host` hacks.
- **Sub-500ms TTFR** — Groq LPU inference (~300ms) + local Kokoro TTS (~50ms) + Groq Whisper STT (~80ms) = ~450ms time-to-first-response.
- **VAD barge-in** — Silero VAD detects patient speech during TTS playback; EMMA stops speaking immediately.
- **Streaming TTS** — Kokoro emits audio chunks as they are synthesised, not after full synthesis.
- **Full observability** — Langfuse traces every agent node, RAG retrieval, reranking decision, and safety event.
- **RAGAS regression prevention** — CI eval suite fails the build if faithfulness drops below 0.80.
- **Real-time dashboard** — SSE event bus (in-process fan-out, 200-event ring buffer, 15-second heartbeat) streams transcripts, agent decisions, and safety events to the Next.js frontend live.
- **One-command deployment** — `docker-compose up -d` brings up all six services.

### Multi-Tenancy

- Separate Qdrant collections per surgery — physically isolated at the vector DB level.
- Separate LiveKit SIP trunks and dispatch rules per DDI — each incoming number routes to the correct tenant room prefix.
- Per-tenant prompt injection at runtime — opening hours, escalation wording, prescription lead times are all surgery-specific.
- Tenant config loaded dynamically from `data/{tenant_id}/config.json`.

---

## Performance Metrics


| Metric                | Value      | Notes                             |
|-----------------------|------------|-----------------------------------|
| STT Latency           | ~80ms      | Groq Whisper Large-v3             |
| LLM Inference         | ~300ms     | Groq LPU, Llama 3.3-70b, temp=0   |
| TTS Synthesis         | ~50ms      | Kokoro local CPU                  |
| End-to-End TTFR       | ~450ms     | Total time-to-first-response      |
| RAGAS Faithfulness    | >0.94      | Chain-of-verification applied     |
| Safety Gate Accuracy  | 100%       | Hard-coded — not LLM-decided      |
| Retrieval Relevance   | >90%       | MiniLM-L6-v2 + cross-encoder      |
| DeepEval Safety Tests | 42/42 pass | Emergency + no-advice + isolation |

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| STT | Groq Whisper Large-v3 | Free tier · < 100ms |
| LLM | Groq Llama 3.3-70b | Free tier · fast LPU |
| TTS | Kokoro (local CPU) | Open source · British accent · GDPR-friendly |
| Vector DB | Qdrant (Docker) | Open source · namespace isolation |
| Embeddings | all-MiniLM-L6-v2 | HuggingFace · local |
| Reranking | bge-reranker-v2-m3 | HuggingFace · local |
| Orchestration | LangGraph | Multi-agent state machine |
| Telephony | LiveKit OSS | WebRTC + SIP · horizontal scaling |
| Safety | NeMo Guardrails | NVIDIA open source · input rails |
| Observability | Langfuse (self-hosted) | Full agent tracing · Docker image |
| RAG Eval | RAGAS | pip install · free |
| LLM Eval | DeepEval | pip install · free |
| Frontend | Next.js + Tailwind | Open source · SSE real-time |
| Deployment | Docker Compose | One-command setup |

**Cost: £0 — fully open source, no credit card.**

---


## Project Structure

```
emma/
│
├── 📁 backend/
│   ├── main.py                        ← FastAPI app · LiveKit webhooks · SSE routes
│   ├── config.py                      ← Environment config + tenant loader
│   │
│   ├── 📁 api/
│   │   ├── events.py                  ← In-process EventBus + SSE endpoint
│   │   └── evals_api.py               ← RAGAS + DeepEval REST API
│   │
│   ├── 📁 rag/
│   │   ├── chunker.py                 ← Section-aware semantic chunking
│   │   ├── embedder.py                ← all-MiniLM-L6-v2 wrapper
│   │   ├── ingestor.py                ← Multi-tenant Qdrant ingestion
│   │   └── retriever.py               ← Dense retrieval + cross-encoder reranking
│   │
│   ├── 📁 agents/
│   │   ├── state.py                   ← LangGraph AgentState TypedDict
│   │   ├── safety_agent.py            ← Safety gate (keyword + semantic)
│   │   ├── retrieval_agent.py         ← RAG retrieval node
│   │   ├── response_agent.py          ← Groq Llama 3.3 · temp=0
│   │   ├── verification_agent.py      ← Chain-of-Verification
│   │   ├── escalation_handler.py      ← Hardcoded emergency response path
│   │   └── graph.py                   ← LangGraph orchestration + routing
│   │
│   ├── 📁 voice/
│   │   ├── stt.py                     ← Groq Whisper Large-v3
│   │   ├── tts.py                     ← Kokoro TTS · true streaming
│   │   ├── vad.py                     ← WebRTC VAD + barge-in detection
│   │   ├── session_manager.py         ← Per-call session state
│   │   └── websocket_handler.py       ← Full-duplex WebSocket handler
│   │
│   ├── 📁 telephony/
│   │   ├── ddi_router.py              ← DDI → tenant_id routing
│   │   ├── call_manager.py            ← Active call registry (room_name model)
│   │   ├── audio_bridge.py            ← 48kHz ↔ 16kHz · LiveKit AudioFrame
│   │   ├── sip_provisioner.py         ← SIP trunk + dispatch rule management
│   │   ├── webhook_handler.py         ← LiveKit webhook event processing
│   │   ├── livekit_adapter.py         ← Per-call audio pipeline (AudioStream/AudioSource)
│   │   └── livekit_agent.py           ← Agent worker entrypoint (livekit-agents)
│   │
│   ├── 📁 safety/
│   │   ├── emergency_keywords.py      ← Three-layer emergency detector
│   │   ├── guardrails_handler.py      ← NeMo integration (input rails only)
│   │   └── guardrails/                ← NeMo config.yml + main.co
│   │
│   └── 📁 observability/
│       ├── langfuse_client.py         ← Tracing client + decorators
│       ├── ragas_eval.py              ← RAGAS evaluation pipeline
│       └── deepeval_suite.py          ← DeepEval safety test suite
│
├── 📁 frontend/                       ← Next.js 14 dashboard
│   ├── lib/
│   │   ├── types.ts                   ← Shared TypeScript event types
│   │   ├── api.ts                     ← Typed REST + SSE client
│   │   └── useSSE.ts                  ← React SSE hook (auto-reconnect)
│   ├── pages/
│   │   ├── index.tsx                  ← Live call monitor
│   │   ├── tenants.tsx                ← Multi-tenant management UI
│   │   └── evals.tsx                  ← RAGAS + DeepEval results UI
│   └── components/
│       ├── Layout.tsx                 ← Shared nav + page shell
│       ├── MetricsPanel.tsx           ← Latency · RAGAS · safety scores
│       ├── CallTranscript.tsx         ← Real-time transcript display
│       ├── AgentTrace.tsx             ← LangGraph agent decision timeline
│       ├── SafetyLog.tsx              ← Safety escalation event log
│       ├── TenantSelector.tsx         ← Surgery switcher dropdown
│       └── StatusBadge.tsx            ← Coloured status pill
│
├── 📁 livekit/
│   ├── livekit.yaml                   ← LiveKit server config (self-hosted)
│   └── redis.conf                     ← Redis config for distributed state
│
├── 📁 data/
│   ├── surgery_greenfield/
│   │   ├── guidelines.txt             ← Greenfield GP guidelines (mock)
│   │   └── config.json                ← Tenant config (hours · DDI · prompts)
│   └── surgery_riverside/
│       ├── guidelines.txt             ← Riverside GP guidelines (mock)
│       └── config.json
│
├── 📁 scripts/
│   ├── ingest_all.py                  ← Ingest all tenants into Qdrant
│   ├── provision_sip.py               ← One-shot SIP trunk + dispatch rule setup
│   ├── start_agent.py                 ← Start LiveKit agent worker process
│   └── run_evals.py                   ← Run full RAGAS + DeepEval suite
│
├── 📁 tests/
│   ├── test_retrieval.py              ← Retrieval quality + tenant isolation
│   ├── test_safety_gate.py            ← Emergency detection red-team cases
│   ├── test_agent_pipeline.py         ← End-to-end orchestration · grounding · latency
│   ├── test_voice_pipeline.py         ← Audio · VAD · STT · TTS · session · API (50/50)
│   ├── test_telephony_livekit.py      ← LiveKit DDI routing · SIP provisioning · webhooks
│   └── test_dashboard.py              ← EventBus · SSE · evals API
│
├── .env.example                       ← All required environment variables
├── .gitignore
├── requirements.txt                   ← Pinned Python dependencies
├── Dockerfile.backend
├── Dockerfile.frontend
└── docker-compose.yml                 ← One-command deployment (6 services)
```

---

## Quick Start

### Prerequisites

- Docker + Docker Compose
- Python 3.11+
- A free [Groq API key](https://console.groq.com)
- Node.js 18+ (for local frontend development only; not needed for Docker)

### 1. Clone and configure

```bash
git clone https://github.com/TaufeeqAi/emma
cd emma

cp .env.example .env
# Open .env and set GROQ_API_KEY (free at console.groq.com — no card needed)
```

### 2. Start all services

```bash
# First run pulls images — allow ~3 minutes
docker-compose up -d
```

This brings up six services:

| Service | URL | Purpose |
|---|---|---|
| Dashboard | http://localhost:3000 | Next.js live monitor |
| Backend | http://localhost:8000 | FastAPI + agent API |
| SSE stream | http://localhost:8000/events | Real-time event feed |
| Langfuse | http://localhost:3001 | Observability traces |
| Qdrant | http://localhost:6333 | Vector database |
| LiveKit | http://localhost:7880 | WebRTC + SIP bridge |

### 3. Wait for LiveKit

```bash
until curl -sf http://localhost:7880/healthz; do
  echo "Waiting for LiveKit…"; sleep 2
done
echo "LiveKit ready."
```

### 4. One-time setup

```bash
# Provision SIP trunks and dispatch rules
python scripts/provision_sip.py

# Ingest surgery guidelines into Qdrant
python scripts/ingest_all.py
```

### 5. Verify everything is running

```bash
curl http://localhost:8000/health
curl http://localhost:8000/routing/ddi
curl http://localhost:8000/routing/sip-resources
```

Then open the dashboard at **http://localhost:3000**.

---

## Running Tests

```bash
# Full test suite
pytest tests/ -v

# Unit tests only — no external dependencies, runs anywhere
pytest tests/ -v -m "not requires_livekit and not requires_sip and not requires_backend"

# LiveKit integration tests (requires running LiveKit)
pytest tests/ -v -m "requires_livekit"

# Telephony tests (no SIP calls required)
pytest tests/test_telephony_livekit.py -v -m "not requires_sip"

# Voice pipeline tests
pytest tests/test_voice_pipeline.py -v
```

**Current test status: 50/50 voice pipeline · 42/42 DeepEval safety · all telephony unit tests pass.**

---

## Running Evaluations

```bash
# Full suite (RAGAS + DeepEval)
python scripts/run_evals.py

# Single tenant
python scripts/run_evals.py --tenant surgery_riverside

# From the dashboard:
# http://localhost:3000/evals → "Run Full Eval Suite"
```

RAGAS faithfulness threshold is enforced at **0.80** — the eval runner exits non-zero if results fall below this, making it suitable as a CI gate.

---

## Safety Architecture

Clinical safety is non-negotiable in an NHS context. EMMA enforces the following guarantees by construction:

### 1. Emergency escalation is hardcoded

The Safety Gate runs **before** any LLM call. An LLM cannot decide whether to escalate; it can only decide how to respond after the safety gate returns `SAFE`.

The detector has three layers:

- **Layer 1** — exact normalised keyword matching (chest pain, 999, can't breathe, etc.)
- **Layer 2** — order-preserving gap-tolerant token matching (catches rearranged phrasing)
- **Layer 3** — semantic similarity against curated emergency reference sentences (catches slang and paraphrase: *"my ticker feels like it's going to burst"*)

The detector is **fail-closed**: empty or ambiguous inputs are escalated, not passed through.

### 2. No medical advice by design

NeMo Guardrails input rails prevent the LLM from receiving prompts that would lead to medical advice, diagnosis, or prescription recommendations.

### 3. Chain of Verification on every response

Before any response is spoken, the Verification Agent checks the LLM output for faithfulness against the retrieved context. Responses that cannot be grounded are suppressed and replaced with a fallback.

### 4. Tenant isolation enforced at the vector DB level

Surgery A and Surgery B data live in separate Qdrant collections. Tenant ID is verified at retrieval time — a query for Surgery A cannot return Surgery B context, even if the retrieval query is adversarially constructed.

### 5. Call duration cap

Calls exceeding 10 minutes are terminated with a message directing the patient to call back.

---

## Multi-Tenancy

```
Qdrant
├── collection: surgery_greenfield   → Greenfield Medical Centre data
└── collection: surgery_riverside    → Riverside Surgery data

LiveKit
├── SIP Trunk: EMMA Surgery Greenfield Inbound  → DDI 01234 567890
├── SIP Trunk: EMMA Surgery Riverside Inbound   → DDI 01234 987654
├── Dispatch Rule: Greenfield → room prefix "surgery_greenfield-"
└── Dispatch Rule: Riverside  → room prefix "surgery_riverside-"
```

At call start, EMMA reads room metadata injected by the dispatch rule, loads the corresponding tenant config (opening hours, escalation wording, prescription lead times), and scopes all retrieval to that tenant's Qdrant collection.

---

## Observability

All agent decisions are traced in **Langfuse** (self-hosted at http://localhost:3001):

- Every LangGraph node execution with inputs and outputs
- Every Qdrant retrieval with chunk scores
- Every reranking decision
- Every Safety Gate decision (SAFE vs ESCALATE)
- Every Chain-of-Verification result

The **Next.js dashboard** provides:

- **Live Monitor** — real-time call transcripts, agent trace timeline, safety event log
- **Tenants** — SIP trunk status, DDI routing table, collection stats
- **Evaluations** — RAGAS faithfulness, context relevance, answer correctness, DeepEval safety suite results; run on demand or on a schedule

Events are delivered to the frontend via **Server-Sent Events** (SSE) with a 200-event ring buffer and 15-second heartbeat keepalive. Late-joining subscribers receive recent history immediately on connect.

---

## Acknowledged Limitations

**Not a medical device.** EMMA handles administrative enquiries only — appointments, opening hours, prescription requests. Any clinical question triggers escalation to a human.

**Groq free tier rate limits.** At scale, production would use a paid Groq tier or self-hosted Whisper / Llama. The architecture is designed so the STT, LLM, and TTS layers are swappable with no changes to the agent pipeline.

**In-process CallManager.** The current active-call registry is an in-process dict. Horizontal scaling across multiple backend replicas requires a Redis hash — scaffolded in the codebase but not enabled by default.

**PII in logs.** Caller phone numbers must be masked in Langfuse traces before NHS production deployment (GDPR / DSP Toolkit compliance). This is noted in the security checklist but not automated in this release.

**Kokoro cold start.** The first call after agent worker startup may take 3–5 seconds while the Kokoro model loads. The LiveKit `prewarm_fnc` in `WorkerOptions` mitigates this — never restart agent workers during call hours.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```env
# Groq — free at console.groq.com, no card needed
GROQ_API_KEY=gsk_...

# Qdrant (local Docker)
QDRANT_HOST=localhost
QDRANT_PORT=6333

# LiveKit (self-hosted Docker)
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
LIVEKIT_WEBHOOK_SECRET=secret

# Langfuse (self-hosted Docker)
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=http://localhost:3001

# SIP auth
SIP_AUTH_USERNAME=emma
SIP_AUTH_PASSWORD=changeme-in-production

# App
APP_ENV=development
LOG_LEVEL=INFO
```

---

## Built With

[LangGraph](https://langchain-ai.github.io/langgraph/) ·
[Qdrant](https://qdrant.tech) ·
[Groq](https://groq.com) ·
[Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) ·
[FastAPI](https://fastapi.tiangolo.com) ·
[LiveKit](https://livekit.io) ·
[Langfuse](https://langfuse.com) ·
[RAGAS](https://docs.ragas.io) ·
[DeepEval](https://docs.confident-ai.com) ·
[NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) ·
[Next.js](https://nextjs.org) ·
[Docker](https://docker.com)

---

*Built by [Taufeeq Ahmad](https://github.com/TaufeeqAi) — demonstrating production AI engineering for NHS healthcare systems.*