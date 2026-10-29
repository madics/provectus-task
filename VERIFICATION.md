# Verification

- Python 3.13.2; dependency versions pinned in `pyproject.toml`.
- Automated acceptance suite: **36 passed**.
- Command from the workspace: `work/.venv/Scripts/python.exe -m pytest outputs/stock-price-agent/tests -q --basetemp=work/pytest-run-3`.
- Actual LangChain graph verifies error status, call-ID preservation, model observation, explicit corrected arguments, and success within one user request.
- All 15 fixture records processed within recursion limit 50: 4 validated, 11 unrecoverable sources.
- The model-driven AAPL scenario uses a scripted model, not a live OpenAI response.
- Live OpenAI AAPL smoke test: **passed on 2026-10-29**, with the environment's configured model and credentials. Both raw strings produced errors and the model generated explicit integer retries within the same request. See `LIVE_SMOKE_TRACE.txt`.

Use a workspace-local pytest `--basetemp` in restricted environments where the
system temporary directory is inaccessible.
