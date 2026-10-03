# Frontend

The web interface prototype: React, TypeScript and Vite, in [frontend/](../frontend/). Architecture and plans are in [product-roadmap.md](product-roadmap.md).

## Run it

```
cd backend
python -m uvicorn app.main:app          # the API, on 127.0.0.1:8000

cd frontend
npm install                             # once
npm run dev                             # http://127.0.0.1:5173
```

The dev server forwards `/api` to the backend, so the backend needs no CORS rules. If the backend is not running, the start-up screen says so and offers **Demo mode**: simulated notes, clearly labelled, never your real data.

Other commands: `npm test` (unit tests), `npm run typecheck`, `npm run build` (static files in `frontend/dist`).

## What works today

| Screen | Uses | Notes |
| --- | --- | --- |
| Start-up | `GET /health` | Shows the real link state; never fakes a connection |
| Console | `POST /ask` | Cited answers; `[n]` markers open the source; refusals shown as "insufficient data" |
| Knowledge | `POST /search`, `GET /ingestion/file-types` | Raw retrieval with file-type filters, no model involved |
| System panel | `GET /health` | Version, environment, ping, this session's queries |
| Modules, Privacy | none | Roadmap and policy, written from the docs |
| Settings | local storage | Theme (Arc Reactor, Mark III), reduced motion, demo mode |

## Not built yet

Library manager, ingestion from the UI, streaming answers, saved conversations, authentication, voice (the button is disabled on purpose). Each needs a backend slice first: see section 3 of the product roadmap.

## Rules for changing it

- No external requests: no CDN fonts, scripts or analytics. The page's Content-Security-Policy enforces this.
- Do not show information the app does not have. If the data is not available, say so.
- Conversations live in memory only until the conversation storage slice exists.
- Colours and fonts come from `src/styles/tokens.css`.
