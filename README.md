# Dynamic Agentic System

A dynamic multi-knowledge-base agentic AI system that intelligently routes
user queries to the appropriate capability, retrieves grounded information,
executes database and mathematical operations, dynamically selects AI
personas and LLMs, and provides transparent source citations and live tracing.

## 🚀 Project Overview

The Dynamic Agentic System is an agentic AI platform designed to handle
different types of user queries through a dynamic orchestration layer.

Instead of sending every query through the same pipeline, the system
determines what the query requires and routes it to the appropriate
capability.

The system supports:

- Multi-knowledge-base RAG
- Versioned RAG knowledge bases
- Dynamic query routing
- Source-grounded answers
- PostgreSQL database querying
- Mathematical computation
- AI personas
- Dynamic LLM selection
- Semantic LLM caching
- Stale RAG protection
- Suggested follow-up queries
- Live agent tracing
- Source citations

## 🏗️ Architecture

```
                    ┌─────────────────────┐
                    │      Next.js        │
                    │      Frontend       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       FastAPI       │
                    │     API Layer       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      LangGraph      │
                    │  Agent Orchestrator │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
          RAG Engine       Database          Math
              │             Engine           Engine
              │
              ▼
        Versioned KBs
              │
              ▼
           Pinecone

        PostgreSQL → System State
        Redis      → Semantic Cache
