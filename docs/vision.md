# Vision

Reyleight is a personal AI knowledge assistant that runs entirely on the owner's machine.

## Goals

- Answer questions about my own notes and documents, with citations to the exact source.
- Stay private: embeddings and generation run locally, and documents never leave the machine.
- Be honest: when the documents don't contain the answer, say so instead of guessing.
- Grow deliberately: one small, tested slice at a time, as a way to learn RAG, vector databases and privacy-preserving design.

## Non-goals (for now)

Long-term memory, agents and tools, cloud connectors, browser automation, a full frontend and distribution packaging. These are future direction, tracked in [roadmap.md](roadmap.md), and not current work.

## Design stance

- Local by default. Any cloud option would be opt-in and clearly flagged.
- Least privilege. The app reads only folders the owner explicitly allows.
- Retrieved content is data, never instructions.
