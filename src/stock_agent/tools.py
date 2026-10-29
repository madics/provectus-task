from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
import re
from threading import RLock
from typing import Any

import simplejson as json
from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException
from pydantic import BaseModel, StrictInt, StrictStr, ValidationError


def encode(value: Any) -> str:
    return json.dumps(value, use_decimal=True, ensure_ascii=True, allow_nan=False)


class ValidatedStockPrice(BaseModel):
    record_id: StrictStr
    price: StrictInt


@dataclass
class RequestContext:
    data_path: Path
    snapshot: dict[str, dict] | None = None
    fetched: dict[str, dict] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    outcomes: dict[str, dict] = field(default_factory=dict)
    fetch_errors: list[dict] = field(default_factory=list)
    lock: Any = field(default_factory=RLock)


def source_number(raw: Any) -> Decimal:
    """Private equivalence check. Never expose its computed value to the model."""
    if isinstance(raw, bool) or raw is None:
        raise ValueError("Source is boolean or null")
    if isinstance(raw, int):
        number = Decimal(raw)
    elif isinstance(raw, (Decimal, float)):
        number = Decimal(str(raw))
    elif isinstance(raw, str):
        text = raw.strip()
        if text.startswith("$"):
            text = text[1:]
        if not re.fullmatch(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?", text):
            raise ValueError("Unsupported or ambiguous source formatting")
        number = Decimal(text.replace(",", ""))
    else:
        raise ValueError("Unsupported source type")
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError("Source cannot be represented losslessly as an integer")
    return number


def fail(code: str, **details: Any) -> None:
    raise ToolException(encode({"code": code, **details}))


@tool
def fetch_stock_prices(runtime: ToolRuntime[RequestContext], symbol: str | None = None) -> str:
    """Fetch raw stock records, preserving duplicate symbols and original prices.

    Omit symbol for all records. Symbol matching ignores case and outer whitespace.
    Validate every returned record separately; do not change its raw initial price.
    """
    ctx = runtime.context
    with ctx.lock:
        try:
            if ctx.snapshot is None:
                with ctx.data_path.open(encoding="utf-8-sig") as stream:
                    rows = json.load(stream, use_decimal=True, allow_nan=False)
                if not isinstance(rows, list) or not rows:
                    raise ValueError("Expected a nonempty JSON array")
                snapshot = {}
                for row in rows:
                    if not isinstance(row, dict) or "price" not in row:
                        raise ValueError("Every record requires id and price")
                    rid = row.get("id")
                    if isinstance(rid, bool) or not isinstance(rid, (str, int)) or not str(rid).strip():
                        raise ValueError("Record IDs must be nonempty strings or integers")
                    rid = str(rid)
                    if rid in snapshot:
                        raise ValueError("Record IDs must be unique")
                    if row.get("symbol") is not None and not isinstance(row["symbol"], str):
                        raise ValueError("Symbol must be a string or null")
                    snapshot[rid] = {**row, "id": rid}
                ctx.snapshot = snapshot
            selected = [row for row in ctx.snapshot.values() if symbol is None or
                        (isinstance(row.get("symbol"), str) and row["symbol"].strip().casefold() == symbol.strip().casefold())]
            if not selected:
                fail("unknown_symbol", symbol=symbol)
            ctx.fetched.update({row["id"]: row for row in selected})
            return encode(selected)
        except ToolException as exc:
            ctx.fetch_errors.append(json.loads(str(exc)))
            raise
        except (OSError, ValueError) as exc:
            code = "source_file_error" if isinstance(exc, OSError) else "invalid_dataset"
            error = {"code": code, "message": str(exc)}
            ctx.fetch_errors.append(error)
            fail(**error)


@tool
def validate_stock_price(record_id: str, price: Any, runtime: ToolRuntime[RequestContext]) -> str:
    """Validate an integer price against a fetched source record.

    Initially submit the exact raw price, even when it is a string or null.
    After an error, YOU must generate corrected integer arguments and call again.
    Formatting corrections permit outer whitespace, optional leading $, signed
    decimal notation and properly grouped thousands commas. Fractional values
    cannot be rounded. Boolean/null/unsupported currencies and typos cannot be fixed.
    There are at most three attempts per record. No source data is ever modified.
    """
    ctx = runtime.context
    with ctx.lock:
        if record_id not in ctx.fetched:
            fail("record_not_fetched", record_id=record_id, recoverable=True)
        if record_id in ctx.outcomes:
            fail("record_terminal", record_id=record_id, outcome=ctx.outcomes[record_id], recoverable=False)
        attempt = ctx.attempts.get(record_id, 0) + 1
        ctx.attempts[record_id] = attempt
        remaining = 3 - attempt
        raw = ctx.fetched[record_id]["price"]
        source_error = None
        expected = None
        try:
            expected = source_number(raw)
        except ValueError as exc:
            source_error = str(exc)
        validation_errors = []
        try:
            validated = ValidatedStockPrice(record_id=record_id, price=price)
        except ValidationError as exc:
            validation_errors = exc.errors(include_url=False, include_context=False)
        code = None
        if source_error:
            code = "unrecoverable_source"
        elif validation_errors:
            code = "price_type_error"
        elif Decimal(validated.price) != expected:
            code = "source_value_mismatch"
        if code:
            recoverable = source_error is None and remaining > 0
            error = {
                "code": code, "record_id": record_id, "submitted_value": price,
                "submitted_type": type(price).__name__, "expected_type": "integer (StrictInt)",
                "validation_details": validation_errors, "source_issue": source_error,
                "recoverable": recoverable, "attempt": attempt, "attempts_remaining": remaining,
            }
            if not recoverable:
                ctx.outcomes[record_id] = {"record_id": record_id, "status": "failed",
                                           "code": "attempts_exhausted" if remaining == 0 and not source_error else code}
            fail(**error)
        result = {"record_id": record_id, "price": validated.price}
        ctx.outcomes[record_id] = {**result, "status": "validated"}
        return encode(result)


# Verified with tool-call invocations and the actual create_agent graph in tests.
# LangChain wraps handled ToolException content as status=error with the call ID.
fetch_stock_prices.handle_tool_error = True
validate_stock_price.handle_tool_error = True
