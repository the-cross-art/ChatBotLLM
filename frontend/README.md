# ChatBot Frontend (Vite + React + TS)

This frontend mirrors the FastAPI + WebSocket integration pattern from `LangGraphPy-x-ReactJS`, connecting to your backend at `ws://localhost:8000/ws`.

## Dev Run (Create React App)

```bash
cd frontend
npm install
npm start
```

Open `http://localhost:3000`.

## Build + Serve via FastAPI (Optional)

```bash
cd frontend
npm install
npm run build
cd ..
uv run uvicorn server:app --reload --port 8000
```

If `frontend/build` exists, `server.py` mounts it at `/` automatically.

## WebSocket Protocol

- Init (sent by client on connect):
  - `{ user_id: string, thread_id: string, init: true }`
- Chat (sent by client):
  - `{ message: string }`
- Server response:
  - `{ type: "assistant_message", content: string, citations?: Array<{ text, source?, doc_id?, title?, section? }> }`

## Notes
- Ensure your backend (`server.py`) is running on port 8000.
- Adjust `useWebSocket` URL if deploying behind HTTPS or a proxy.
