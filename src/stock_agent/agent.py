import os
from pathlib import Path
from uuid import uuid4

import simplejson as json
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from .tools import RequestContext, encode, fetch_stock_prices, validate_stock_price


SYSTEM_PROMPT = """You demonstrate autonomous correction of stock-price tool arguments.
For every user request about prices, fetch the raw source records. For EACH returned
record (including duplicates), first call validate_stock_price with the original
raw price unchanged and its string record ID. Do not pre-normalize the first call.
Read each tool observation before deciding what to do next. On a recoverable error,
interpret it, construct the equivalent INTEGER argument yourself, and call validation
again within this same request. Never ask the user to correct formatting. Never
invent a price, round or truncate. No more than three attempts per record. A wrong
candidate is retryable while attempts remain; terminal sources cannot be fixed.
Process all fetched records. Never claim validation without a successful tool result.
On later user turns fetch again, even if you remember a previous result.
Return only JSON as your final answer, with exactly these keys:
{"validated": [{"record_id": "...", "price": 150}],
 "failed": [{"record_id": "...", "code": "..."}]}
Include every fetched record exactly once. For terminal failures, use the terminal
code from the tool (attempts_exhausted when the third recoverable attempt fails).
If fetching fails, return empty lists. Do not include reasoning or markdown fences.
"""


def configured_model():
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL"):
        if not os.getenv(name, "").strip():
            raise ValueError(f"Set {name} before starting the agent")
    return ChatOpenAI(model=os.environ["OPENAI_MODEL"], use_responses_api=True,
                      model_kwargs={"parallel_tool_calls": False}, max_retries=2, timeout=60)


def default_data_path() -> Path:
    return Path(os.getenv("STOCK_DATA_PATH", str(Path.home() /
        "OneDrive/Desktop/provectus-task/synthetic_messy_stock_prices.json")))


def final_outcome(ctx: RequestContext, answer: str, execution_error: str | None = None) -> dict:
    validated = sorted(({"record_id": rid, "price": row["price"]} for rid, row in ctx.outcomes.items()
                        if row["status"] == "validated"), key=lambda row: row["record_id"])
    failed = sorted(({"record_id": rid, "code": row["code"]} for rid, row in ctx.outcomes.items()
                     if row["status"] == "failed"), key=lambda row: row["record_id"])
    unresolved = sorted(set(ctx.fetched) - set(ctx.outcomes))
    matches = False
    try:
        proposed = json.loads(answer)
        # Exact JSON types prevent True or 150.0 masquerading as validated integers.
        matches = (set(proposed) == {"validated", "failed"}
                   and all(type(row["price"]) is int for row in proposed["validated"])
                   and sorted(proposed["validated"], key=lambda row: row["record_id"]) == validated
                   and sorted(proposed["failed"], key=lambda row: row["record_id"]) == failed)
    except (ValueError, TypeError, KeyError, AttributeError):
        pass
    complete = bool(ctx.fetched) and not unresolved and matches and not execution_error
    return {"status": "complete" if complete else "incomplete", "validated": validated,
            "failed": failed, "unresolved": unresolved, "answer_matches_tools": matches,
            "fetch_errors": ctx.fetch_errors, "execution_error": execution_error}


class StockAgent:
    def __init__(self, model, data_path: Path):
        self.data_path = data_path
        self.thread_id = str(uuid4())
        self.graph = create_agent(model=model, tools=[fetch_stock_prices, validate_stock_price],
                                  system_prompt=SYSTEM_PROMPT, context_schema=RequestContext,
                                  checkpointer=InMemorySaver())

    def new_conversation(self):
        self.thread_id = str(uuid4())

    def run(self, request: str, emit=print, recursion_limit: int = 50) -> dict:
        ctx = RequestContext(self.data_path)
        answer = ""
        error = None
        emit(f"USER: {request}")
        try:
            for update in self.graph.stream({"messages": [{"role": "user", "content": request}]},
                    config={"configurable": {"thread_id": self.thread_id}, "recursion_limit": recursion_limit},
                    context=ctx, stream_mode="updates"):
                for node in update.values():
                    if not isinstance(node, dict):
                        continue
                    for message in node.get("messages", []):
                        if isinstance(message, AIMessage):
                            for call in message.tool_calls:
                                emit(f"AGENT TOOL CALL [{call['id']}]: {call['name']} {encode(call['args'])}")
                            if not message.tool_calls:
                                answer = message.text
                        elif isinstance(message, ToolMessage):
                            emit(f"TOOL {message.status.upper()} [{message.tool_call_id}]: {message.content}")
        except GraphRecursionError:
            error = "Graph step limit reached; request incomplete. Start /new before continuing."
        except Exception as exc:
            # Keep the CLI usable on API failures without printing credentials or payloads.
            error = f"Execution failed ({type(exc).__name__}); start /new before continuing."
        report = final_outcome(ctx, answer, error)
        emit(f"ASSISTANT (model output): {answer or '[no final answer]'}")
        emit(f"VERIFIED OUTCOME: {encode(report)}")
        if error:
            self.new_conversation()
        return report
