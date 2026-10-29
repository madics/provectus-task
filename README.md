# Stock-price agent

A small Python CLI demonstrating model-generated correction of tool arguments.
The application never edits the source stock data. No external stock API is used.

## Setup (PowerShell, Python 3.11+)

Run these commands from this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e '.[test]'
Copy-Item .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY` and `OPENAI_MODEL` to a tool-capable OpenAI
model available to your account. The application uses `ChatOpenAI` with
`use_responses_api=True`; model selection is intentionally explicit.

By default, the source is your home directory plus
`OneDrive/Desktop/provectus-task/synthetic_messy_stock_prices.json`.
To use the included unchanged copy of the generated fixture instead, set:

```dotenv
STOCK_DATA_PATH=data/synthetic_messy_stock_prices.json
```

Relative paths resolve from the process working directory. Then run:

```powershell
stock-price-agent
```

Ask `Get the stock price of AAPL.` or `Get all stock prices.`. `/new` creates an
isolated conversation; `/exit` quits. History persists only while this process
runs. Every user request gets fresh source snapshots and retry budgets. Within a
request, the first fetch pins the complete dataset, so later fetches cannot change
the source being verified or reset attempts.

## What the demonstration proves

`fetch_stock_prices` returns raw prices, preserves duplicate symbols and converts
IDs to strings. `validate_stock_price` deliberately exposes `price: Any` so values
such as `"$150.00"` reach its internal Pydantic `StrictInt` model. The error is
converted to `ToolException`; `handle_tool_error=True` produces an error
`ToolMessage` with the original call ID. The next model invocation sees that error.

Only the model constructs corrected tool-call arguments. The private source check
does not return its computed number or issue any retry. An integer candidate must
equal the source's lossless Decimal value, even if the model skipped the instructed
raw-price attempt. The system prompt requires raw-first attempts; the actual trace
lets an interviewer verify model compliance.

Allowed source formatting: outer whitespace, optional leading `$`, optional sign,
ASCII digits, optional correctly grouped thousands commas, optional decimal digits.
All fractional digits must represent zero. Thus `$150.00`, `150`, ` 150 ` and JSON
`150.0` permit integer `150`. `$150.75`, `true`, `null`, `€150`, `175,50`, `1e2`
(as a string), malformed grouping and typo substitutions are unrecoverable.
Finite JSON numbers are parsed with Decimal, preserving precision. Negative
integers are allowed; this exercise validates formatting rather than market rules.

Wrong candidates remain retryable if the source is representable. The third failed
attempt terminates that record. Calls to terminal records perform no further
validation. A request-local lock serializes tool bodies, and the live model is
configured with parallel tool calls disabled. There is no correction middleware.

The final model answer is JSON with `validated` and `failed` lists. The CLI prints
it as model output, followed by an authoritative `VERIFIED OUTCOME`. This checks
that every fetched record is terminal and that the answer exactly agrees with tool
results. An early stop, contradiction, malformed final answer, fetch failure or
execution limit is reported as incomplete. It does not trigger hidden model calls.
Mixed results can be complete when all records are either validated or explicitly
failed. On API/graph failure, the CLI automatically switches to a fresh conversation
to avoid resuming an unfinished tool-call protocol.

## Illustrative conversation (not a captured live run)

The actual fixture contains two AAPL records, so both must be processed. IDs below
are illustrative; runtime prints actual model-generated IDs and arguments.

```text
USER: Get the stock price of AAPL.
AGENT TOOL CALL [fetch]: fetch_stock_prices {"symbol":"AAPL"}
TOOL SUCCESS [fetch]: [{"id":"1",...,"price":"$150.00"},{"id":"13",...,"price":"150"}]
AGENT TOOL CALL [raw-1]: validate_stock_price {"record_id":"1","price":"$150.00"}
TOOL ERROR [raw-1]: {"code":"price_type_error",...,"recoverable":true,"attempts_remaining":2}
AGENT TOOL CALL [fixed-1]: validate_stock_price {"record_id":"1","price":150}
TOOL SUCCESS [fixed-1]: {"record_id":"1","price":150}
AGENT TOOL CALL [raw-13]: validate_stock_price {"record_id":"13","price":"150"}
TOOL ERROR [raw-13]: {"code":"price_type_error",...,"recoverable":true,"attempts_remaining":2}
AGENT TOOL CALL [fixed-13]: validate_stock_price {"record_id":"13","price":150}
TOOL SUCCESS [fixed-13]: {"record_id":"13","price":150}
ASSISTANT (model output): {"validated":[{"record_id":"1","price":150},{"record_id":"13","price":150}],"failed":[]}
VERIFIED OUTCOME: {"status":"complete",...}
```

MSFT's raw integer `312` succeeds immediately. AMZN's `175,50` is rejected as an
unrecoverable source; it must appear under `failed`, not as a fabricated integer.

## Automated acceptance tests (no credentials or API calls)

```powershell
python -m pytest -q --basetemp=.pytest-tmp
```

Tests inject a scripted `BaseChatModel` into the real `create_agent` graph. They
assert that the model receives an error observation and then emits an explicit
integer retry in the same request. They cover both AAPL records, source equivalence,
retry budgets, unrecoverable sources, source byte preservation, source errors,
history, `/new`, premature completion and graph exhaustion.

The full-fixture test processes 15 records with 19 tool executions and 20 model
decisions, within the 50 graph-step budget. The expected result is 4 validated
records (IDs 1, 2, 11 and 13) and 11 failures. ID 11 is the lossless integer -12;
its unrelated invalid date is outside price validation. Graph steps are not tool-call counts;
50 is a separate safeguard, not a promise that every possible three-attempt
trajectory over all records fits. Exhaustion produces an incomplete outcome.

## Live OpenAI smoke test

This is separate from deterministic tests and requires credentials and API usage.

1. Configure `.env`, run `stock-price-agent`, and request AAPL.
2. Inspect the real trace: fetch returns both raw strings; each raw validation call
   produces `TOOL ERROR`; subsequent model-generated calls contain integer `150`.
3. Confirm errors and responses retain their originating call IDs, retries occur
   without another user message, and `VERIFIED OUTCOME` reports complete with both IDs.
4. Request MSFT, then AMZN to inspect immediate success and graceful failure.
5. Use `/new` and confirm conversation isolation.

A scripted model proves orchestration, not live LLM interpretation. A separate live
OpenAI AAPL smoke test passed on 2026-10-29. See `LIVE_SMOKE_TRACE.txt` for actual
model-generated calls: both raw strings failed and both subsequent integer calls
succeeded within one user request. The example above remains illustrative.

## Tested mechanism

Dependencies in `pyproject.toml` pin the tested LangChain/Pydantic versions.
LangChain's installed `BaseTool` implementation handles `ToolException` and wraps
the result as `ToolMessage(status="error", tool_call_id=...)`; the graph acceptance
test verifies this behavior rather than relying on documentation alone.
