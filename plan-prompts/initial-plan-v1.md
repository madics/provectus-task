# Stock-price agent with observable validation errors

## Summary

Build a Python CLI project in the current workspace using LangChain, `langchain-openai`, and Pydantic v2. Preserve conversation state in memory for the lifetime of the process, as selected.

The loop will follow: **user → model → tool → tool observation → model → response**. LangChain supports this through [`create_agent`](https://docs.langchain.com/oss/python/langchain/agents); tool results must reach the model before its final response. [OpenAI documentation](https://developers.openai.com/api/docs/guides/function-calling)

## Implementation

- Use `ChatOpenAI` with the Responses API, configured through `OPENAI_API_KEY` and required `OPENAI_MODEL` environment variables.
- Create the agent with `InMemorySaver` and a stable thread ID per CLI conversation. Store user messages, model tool calls, successful results, and error observations. [LangChain memory documentation](https://docs.langchain.com/oss/python/langchain/short-term-memory)
- Provide an interactive CLI with visible tool calls, observations, and final answers. Support `/new` to start an isolated conversation and `/exit` to quit.
- Include dependency configuration, `.env.example`, setup instructions, and example success/error conversations. Bound each invocation to 20 graph steps to prevent endless loops.

## Fetch tool and error propagation

- Expose `fetch_stock_prices(symbol: str | None = None)`. No symbol fetches all records; a symbol matches case-insensitively after trimming whitespace. Return all matches, including duplicates.
- Read the existing Desktop `provectus-task/synthetic_messy_stock_prices.json` on every call. Allow overriding its path through `STOCK_DATA_PATH`.
- Validate selected records inside the tool using a Pydantic response model with `price: StrictInt`. Keep unrelated messy fields permissive so they do not obscure price-validation failures.
- Reject strings—including `"150"`—floats, booleans, and null prices without conversion. Accept integer prices without imposing additional financial rules.
- Aggregate validation failures with record ID, symbol, field, raw value, actual type, and Pydantic error details. Fail the entire selected batch if any price is invalid.
- Convert validation failures into a LangChain `ToolException`, handled as a `ToolMessage` with `status="error"` and the originating tool-call ID. Preserve this message in state and pass it to the next model invocation.
- Instruct the model to identify invalid records and ask the user to correct the source JSON. It must wait for a subsequent user turn before retrying, and must not invent or repair prices.
- Return explicit tool errors for missing files, malformed JSON, and unknown symbols.

## Verification

- Verify `MSFT` returns integer `312`, including whitespace and case variations.
- Verify `"$150.00"`, `"150"`, `150.0`, `true`, and `null` fail strict validation.
- Verify AAPL returns errors for both matching records; fetching all records reports every invalid price.
- Use a scripted model test to prove that the next model invocation receives the error `ToolMessage` with the correct call ID.
- Verify successful observations also reach the model, history persists across turns, and a new thread has isolated state.
- Verify editing a temporary source fixture allows a later retry to succeed.
- Run automated tests without API credentials; document a separate live smoke test using the configured OpenAI model.

## Assumptions

- Deliver a local CLI; no web interface or hosted service.
- State resets when the process exits.
- The source JSON remains unchanged by the application.
- Credentials stay in environment variables and are excluded from version control.
