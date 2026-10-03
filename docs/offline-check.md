# Checking that it works offline

Reyleight promises that your notes, your questions and the answers never leave your computer. This page shows how to **verify** that instead of trusting it. There are two checks, and the second is stronger.

Both assume you have already done the one-time setup that needs the internet: `python -m app download-model` for the embedding model, and `ollama pull <model>` for the answering model.

## Check 1: the built-in check (software proof)

```powershell
cd C:\dev\rayleigh\backend
.\.venv\Scripts\python.exe -m app offline-check
```

It runs the whole pipeline (load the embedding model, ingest and index the test notes, search, and ask the local model) while a guard blocks every network connection that is not to this computer. A good result looks like this:

```
OFFLINE CHECK
  network guard self-test: a connection to 203.0.113.1 was refused (OK)
  embedding model loaded from disk (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1)
  corpus ingested and indexed: 12 documents, 27 chunks
  searched 37 questions (right note in the top 5 for 100%)
  answered 5 questions with qwen3.5:4b (5 correct, informational only)
  outbound connection attempts blocked: 0
  local connections used: 5 (127.0.0.1:11434)
  RESULT: PASS - nothing tried to reach outside this computer.
```

How to read it:

- **Self-test.** The guard first proves it works, by trying to reach a public address and confirming it was refused. If this fails, the check stops, because a guard that does nothing would prove nothing.
- **Blocked: 0** is the point: nothing tried to reach the internet. If something did, the check fails and names the destination (only the address, never your text).
- **Local connections** shows what *was* allowed: your own Ollama server on `127.0.0.1`. That is traffic that stays on your computer.

### Checking any command

Put `--offline` before any command to run it under the same guard on your own notes:

```powershell
.\.venv\Scripts\python.exe -m app --offline ask how long should I simmer oats
```

The answer prints as usual, and a line on stderr reports `outbound connection attempts blocked: 0`. A command that tried to reach outside would fail and name the address. For example, `download-model` run this way would be refused if it needed to download anything.

### What this check cannot see

The guard watches Python's networking, which is how every library in this project connects (including the Hugging Face tools). A library that opened a network connection from its own native code, without going through Python, would not be seen. That is why Check 2 exists.

## Check 2: really disconnected (physical proof)

This is the check that needs no trust in any software.

1. Make sure Ollama is running (open the app, or run `ollama serve`) and that `python -m app status` shows `running, model installed`.
2. **Turn the network off.** Either use airplane mode, or disable Wi-Fi and unplug any network cable. Confirm it is off, for example by trying to open a website.
3. Run:
   ```powershell
   cd C:\dev\rayleigh\backend
   .\.venv\Scripts\python.exe -m app offline-check
   .\.venv\Scripts\python.exe -m app ingest
   .\.venv\Scripts\python.exe -m app ask <a question about your notes>
   ```
4. Everything should work exactly as before: answers, citations, refusals.
5. Turn the network back on.

If something fails while offline, the error message tells you what was missing. The usual cause is a model that was never downloaded.

## Optional: a stricter test with the Windows firewall

If you want to prevent *any* program on the machine from leaking, you can add a Windows Firewall rule that blocks `python.exe` from this project's virtual environment from reaching the network, and then run the commands above. Create it in Windows Security → Firewall → Advanced settings → Outbound Rules, pointing at `backend\.venv\Scripts\python.exe`. Loopback traffic to Ollama is not affected by outbound blocks for internet addresses. Remove the rule afterwards if you also use that Python for downloads (`download-model`).

## What was verified

On 2026-10-03:

- `python -m app offline-check` passed with the real embedding model and the real `qwen3.5:4b` model: 0 outbound attempts, and the only connections went to `127.0.0.1:11434` (Ollama). It took under two minutes.
- `python -m app --offline ingest` and `--offline ask` on a small test library also showed 0 blocked attempts, and `ask` made one local connection to Ollama.
- The guard itself is covered by automated tests: it refuses outside addresses, name lookups, UDP and web requests, allows `127.0.0.1`, and is restored afterwards. Check 2 (turning the network off) requires you to do it yourself, because it cannot be done from inside this project; the steps above are the exact procedure.
