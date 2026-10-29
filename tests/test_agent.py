from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import simplejson as json
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import ToolException
from pydantic import PrivateAttr

from stock_agent.agent import StockAgent, configured_model, final_outcome
from stock_agent.cli import interact
from stock_agent.tools import RequestContext, encode, fetch_stock_prices, source_number, validate_stock_price


FIXTURE = Path(__file__).parent.parent / "data/synthetic_messy_stock_prices.json"


class ScriptedModel(BaseChatModel):
    _decide: Any = PrivateAttr()
    _histories: list = PrivateAttr(default_factory=list)

    def __init__(self, decide):
        super().__init__()
        self._decide = decide

    @property
    def _llm_type(self):
        return "scripted-acceptance-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._histories.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self._decide(messages))])


def call(name, args, cid):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid}])


def fixture_context(tmp_path, price):
    path = tmp_path / "prices.json"
    path.write_text(encode([{"id": 1, "symbol": "TEST", "price": price}]), encoding="utf8")
    ctx = RequestContext(path)
    runtime = SimpleNamespace(context=ctx)
    fetch_stock_prices.func(runtime=runtime)
    return ctx, runtime


def validate(runtime, value):
    return json.loads(validate_stock_price.func(record_id="1", price=value, runtime=runtime))


def test_real_graph_error_correction_and_duplicate_processing():
    def decide(messages):
        observations = [m for m in messages if isinstance(m, ToolMessage)]
        if not observations:
            return call("fetch_stock_prices", {"symbol": " aapl "}, "fetch")
        last = observations[-1]
        if last.tool_call_id == "fetch":
            records = json.loads(last.content)
            assert [r["id"] for r in records] == ["1", "13"]
            assert [r["price"] for r in records] == ["$150.00", "150"]
            return call("validate_stock_price", {"record_id": "1", "price": "$150.00"}, "bad-1")
        if last.tool_call_id == "bad-1":
            assert last.status == "error"
            payload = json.loads(last.content)
            assert payload["code"] == "price_type_error"
            assert payload["validation_details"][0]["type"] == "int_type"
            assert payload["recoverable"]
            # This is the MODEL response, not a tool-generated corrected call.
            return call("validate_stock_price", {"record_id": "1", "price": 150}, "fixed-1")
        if last.tool_call_id == "fixed-1":
            assert last.status == "success"
            return call("validate_stock_price", {"record_id": "13", "price": "150"}, "bad-13")
        if last.tool_call_id == "bad-13":
            assert last.status == "error"
            return call("validate_stock_price", {"record_id": "13", "price": 150}, "fixed-13")
        return AIMessage(content=encode({"validated": [{"record_id": "1", "price": 150},
                                                     {"record_id": "13", "price": 150}], "failed": []}))

    model = ScriptedModel(decide)
    original = FIXTURE.read_bytes()
    trace = []
    report = StockAgent(model, FIXTURE).run("Get AAPL", emit=trace.append)
    assert report["status"] == "complete"
    assert len(model._histories) == 6
    assert sum(isinstance(m, HumanMessage) for m in model._histories[-1]) == 1
    assert any('"price": 150' in line and "fixed-1" in line for line in trace)
    assert any("TOOL ERROR [bad-1]" in line for line in trace)
    assert FIXTURE.read_bytes() == original


@pytest.mark.parametrize("raw", [150, "$150.00", "150", " 150 ", Decimal("150.0"), "$1,000.00", "-12.00"])
def test_lossless_candidates(tmp_path, raw):
    ctx, runtime = fixture_context(tmp_path, raw)
    candidate = int(source_number(raw))
    assert validate(runtime, candidate)["price"] == candidate
    assert ctx.attempts["1"] == 1


@pytest.mark.parametrize("raw", ["$150.75", True, None, "1,50", "€150", "1e2", "$32.4O", "", Decimal("150.00000000000000000001")])
def test_unrecoverable_source(tmp_path, raw):
    ctx, runtime = fixture_context(tmp_path, raw)
    with pytest.raises(ToolException) as caught:
        validate(runtime, raw)
    error = json.loads(str(caught.value))
    assert error["code"] == "unrecoverable_source"
    assert error["recoverable"] is False
    assert ctx.outcomes["1"]["status"] == "failed"


@pytest.mark.parametrize("candidate", ["150", 150.0, True, None, "$150.00"])
def test_invalid_runtime_types_are_observable(tmp_path, candidate):
    _, runtime = fixture_context(tmp_path, "$150.00")
    with pytest.raises(ToolException) as caught:
        validate(runtime, candidate)
    error = json.loads(str(caught.value))
    assert error["code"] == "price_type_error" and error["recoverable"]


def test_wrong_candidates_remain_retryable_and_limit_cannot_reset(tmp_path):
    ctx, runtime = fixture_context(tmp_path, "$150.00")
    original = ctx.data_path.read_bytes()
    for attempt in range(1, 4):
        with pytest.raises(ToolException) as caught:
            validate(runtime, 999)
        error = json.loads(str(caught.value))
        assert error["code"] == "source_value_mismatch"
        assert error["recoverable"] == (attempt < 3)
        fetch_stock_prices.func(runtime=runtime)
    with pytest.raises(ToolException):
        validate(runtime, 150)
    assert ctx.attempts["1"] == 3
    assert ctx.outcomes["1"]["code"] == "attempts_exhausted"
    assert ctx.data_path.read_bytes() == original


def test_wrong_candidate_then_correct_candidate(tmp_path):
    ctx, runtime = fixture_context(tmp_path, "$150.00")
    with pytest.raises(ToolException):
        validate(runtime, 151)
    assert "1" not in ctx.outcomes
    assert validate(runtime, 150)["price"] == 150


@pytest.mark.parametrize("contents", ["{", "{}", "[]", '[{"id":1}]',
    '[{"id":1,"price":1},{"id":1,"price":2}]', '[{"id":true,"price":1}]'])
def test_invalid_datasets(tmp_path, contents):
    path = tmp_path / "bad.json"
    path.write_text(contents)
    ctx = RequestContext(path)
    with pytest.raises(ToolException):
        fetch_stock_prices.func(runtime=SimpleNamespace(context=ctx))
    assert ctx.fetch_errors[0]["code"] == "invalid_dataset"


def test_missing_file_and_unknown_symbol(tmp_path):
    ctx = RequestContext(tmp_path / "missing.json")
    with pytest.raises(ToolException):
        fetch_stock_prices.func(runtime=SimpleNamespace(context=ctx))
    assert ctx.fetch_errors[0]["code"] == "source_file_error"
    ctx = RequestContext(FIXTURE)
    with pytest.raises(ToolException):
        fetch_stock_prices.func(runtime=SimpleNamespace(context=ctx), symbol="unknown")
    assert ctx.fetch_errors[0]["code"] == "unknown_symbol"


def test_snapshot_is_pinned_and_unfetched_id_rejected(tmp_path):
    ctx, runtime = fixture_context(tmp_path, "$150.00")
    ctx.data_path.write_text('[{"id":1,"symbol":"TEST","price":999}]')
    fetch_stock_prices.func(runtime=runtime)
    assert validate(runtime, 150)["price"] == 150
    with pytest.raises(ToolException):
        validate_stock_price.func(record_id="not-fetched", price=1, runtime=runtime)


def test_state_history_new_thread_and_fresh_attempts():
    def decide(messages):
        # Only consider observations since this user request.
        last_user = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        tools = [m for m in messages[last_user:] if isinstance(m, ToolMessage)]
        if not tools:
            return call("fetch_stock_prices", {"symbol": "MSFT"}, "fetch")
        if len(tools) == 1:
            return call("validate_stock_price", {"record_id": "2", "price": "312"}, "bad")
        if len(tools) == 2:
            assert json.loads(tools[-1].content)["attempt"] == 1
            return call("validate_stock_price", {"record_id": "2", "price": 312}, "good")
        return AIMessage(content='{"validated":[{"record_id":"2","price":312}],"failed":[]}')
    model = ScriptedModel(decide)
    agent = StockAgent(model, FIXTURE)
    for _ in range(2):
        assert agent.run("MSFT", emit=lambda _: None)["status"] == "complete"
    assert sum(isinstance(m, HumanMessage) for m in model._histories[-1]) == 2
    old = agent.thread_id
    commands = iter(["/new", "MSFT", "/exit"])
    interact(agent, read=lambda _: next(commands), emit=lambda _: None)
    assert agent.thread_id != old
    assert sum(isinstance(m, HumanMessage) for m in model._histories[-1]) == 1


def test_early_stop_contradiction_and_recursion():
    ctx = RequestContext(FIXTURE)
    ctx.fetched["1"] = {"price": "$150.00"}
    assert final_outcome(ctx, '{"validated":[],"failed":[]}')["unresolved"] == ["1"]
    ctx.outcomes["1"] = {"status": "validated", "price": 150}
    assert final_outcome(ctx, '{"validated":[{"record_id":"1","price":999}],"failed":[]}')["status"] == "incomplete"
    model = ScriptedModel(lambda _: call("fetch_stock_prices", {"symbol": "AAPL"}, "again"))
    report = StockAgent(model, FIXTURE).run("AAPL", emit=lambda _: None, recursion_limit=4)
    assert report["execution_error"].startswith("Graph step limit")


def test_full_fixture_under_50_graph_steps():
    def decide(messages):
        observations = [m for m in messages if isinstance(m, ToolMessage)]
        if not observations:
            return call("fetch_stock_prices", {}, "fetch")
        records = json.loads(observations[0].content, use_decimal=True)
        outcomes = {}
        counts = {}
        for msg in observations[1:]:
            payload = json.loads(msg.content)
            rid = payload["record_id"]
            counts[rid] = counts.get(rid, 0) + 1
            if msg.status == "success":
                outcomes[rid] = {"record_id": rid, "price": payload["price"]}
            elif not payload["recoverable"]:
                outcomes[rid] = {"record_id": rid, "code": payload["code"]}
        for record in records:
            rid = record["id"]
            if rid in outcomes:
                continue
            raw = record["price"]
            if counts.get(rid, 0):
                # Scripted model emits an explicit new argument, independently of tools.
                assert rid in ("1", "11", "13")
                raw = -12 if rid == "11" else 150
            return call("validate_stock_price", {"record_id": rid, "price": raw}, f"{rid}-{counts.get(rid, 0)}")
        return AIMessage(content=encode({"validated": [v for v in outcomes.values() if "price" in v],
                                         "failed": [v for v in outcomes.values() if "code" in v]}))
    model = ScriptedModel(decide)
    report = StockAgent(model, FIXTURE).run("Get every stock", emit=lambda _: None)
    assert report["status"] == "complete", report
    assert len(report["validated"]) == 4
    assert len(report["failed"]) == 11
    assert len(model._histories) == 20  # Fetch + 18 validation calls + final model decision.


def test_no_credentials_required_and_startup_checks(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        configured_model()
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(ValueError, match="OPENAI_MODEL"):
        configured_model()
