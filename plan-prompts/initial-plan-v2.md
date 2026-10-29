# Demonstrable stock-price agent with autonomous correction

## Implementation

Build a Python CLI in the current workspace using `create_agent`, `ChatOpenAI` with the Responses API, Pydantic v2, and `InMemorySaver`.

Require `OPENAI_API_KEY` and `OPENAI_MODEL`. Keep stable conversation thread IDs, `/new`, `/exit`, dependency configuration, `.env.example`, setup instructions, and the existing read-only JSON fixture with `STOCK_DATA_PATH` override.

### Fetch tool

`fetch_stock_prices(symbol: str | None = None)`:

- Return matching records with unchanged raw prices and string record IDs.
- Trim symbol whitespace, ignore case, and preserve duplicates.
- Handle missing files, malformed datasets, and unknown symbols.
- Retain fetched source records in request-local context for independent verification. Re-fetching does not reset validation budgets.

### Validation tool

`validate_stock_price(record_id: str, price: Any)` exposes permissive arguments so invalid values reach internal Pydantic validation. Its internal model requires `price: StrictInt`.

- Validate the submitted value without coercion.
- Independently verify integer candidates against the fetched source using `Decimal`.
- Accept lossless formatting corrections: surrounding whitespace, optional `$`, signed decimal strings, correctly grouped thousands separators, and mathematically integral numeric values.
- Reject fractional values, booleans, null, malformed grouping, ambiguous decimal-comma strings, unsupported currency markers, and typo substitutions.
- Never return a computed replacement value or modify the source.

## Error handling and retry behavior

- Inspect the installed LangChain version and use its supported tool-error mechanism. Verify that handled errors become `ToolMessage(status="error")` with the original tool-call ID.
- Structured errors include record ID, submitted value/type, error code, validation details, recoverability, and remaining attempts.
- Invalid model candidates remain retryable when the source is representable and attempts remain.
- Mark a record unrecoverable only when its source cannot represent an integer losslessly, or its three-attempt budget is exhausted.
- Enforce three validation attempts per record per user request programmatically. Re-fetching cannot reset the budget; serialize tool execution to avoid counter races.
- No middleware performs correction or retry. The model receives each error and generates its next call.

## Normal agent execution and observable results

System instructions require the model to:

1. Fetch the requested records.
2. Submit each raw price unchanged on its first validation call.
3. Observe validation errors.
4. Generate corrected integer arguments and retry autonomously.
5. Process every duplicate and distinguish validated results from failures.

Use the normal agent loop without middleware that forces extra model calls or dynamically disables tools.

After execution, deterministically compare fetched records against recorded validation outcomes. Display validated results, terminal failures, and unresolved records. If the model stops early or contradicts tool outcomes, mark the run incomplete; do not claim successful completion or silently initiate another model call.

Print actual tool calls, arguments, observations, retries, and the final response. Exclude private reasoning and fabricated traces.

Start with `recursion_limit=50`, since graph steps are not tool calls. Verify sufficient headroom for the complete 15-record fixture and graceful handling of exhaustion. Keep the independent three-attempt restriction.

## Verification and delivery

Automated tests use a scripted model through the real agent graph, without API credentials:

- Immediate integer success and observable strict-validation failures.
- Correct error status and original call ID reaching the next model invocation.
- A model-generated integer retry succeeding without another user message.
- Independent rejection of invented candidates.
- Retryable incorrect candidates versus unrecoverable source values.
- All duplicates processed and three-attempt limits enforced.
- Decimal precision preserved; no rounding or truncation accepted.
- Read-only source verification by comparing file bytes.
- Controlled file, dataset, symbol, and graph-limit failures.
- Persistent conversation history, fresh per-request budgets, and isolated `/new` conversations.
- Deterministic incomplete-run reporting when the model stops early.
- Complete fixture processing within the graph-step limit.

Document a separate live OpenAI smoke test for AAPL showing both raw validation errors and subsequent model-generated integer calls. Label documentation examples as illustrative; retain actual live output only when executed.

## Execution constraint

This is the complete replacement plan incorporating the final adjustments. Implementation can begin after this chat leaves Plan Mode.
