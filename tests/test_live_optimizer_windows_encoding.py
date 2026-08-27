from datetime import date, time
from pathlib import Path

from same_day_1445.live_optimizer.engine.signal_adapter import NormalizedSignal
from same_day_1445.live_optimizer.jobs.run_1445 import run_1445_cycle
from same_day_1445.live_optimizer.providers.file_provider import FileMinuteProvider


def _raw(path: Path, symbol: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "date,symbol,name,open,high,low,close,volume,amount,pct_change,turnover_rate,source_file"
    ]
    for i in range(1, 23):
        rows.append(f"2026-07-{i:02d},{symbol},x,1,1,1,1,1,1,,,old")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8-sig")


def _minute(root: Path, symbol: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{symbol}.csv").write_text(
        "timestamp,open,high,low,close,volume,amount\n"
        "2026-08-24 14:45:00,1.00,1.02,0.99,1.01,100,101\n"
        "2026-08-24 15:00:00,1.01,1.03,1.00,1.02,100,102\n",
        encoding="utf-8",
    )


def test_c_delay1_pending_json_is_read_explicitly_as_utf8(tmp_path: Path, monkeypatch) -> None:
    runtime = tmp_path / "runtime"
    optimizer = tmp_path / "optimizer"
    feed = tmp_path / "feed"

    for symbol in ("510880", "510500"):
        _raw(runtime / "C/raw/etf" / f"{symbol}.csv", symbol)
        _raw(runtime / "S/raw/etf" / f"{symbol}.csv", symbol)
        _minute(feed, symbol)

    (runtime / "R/formal").mkdir(parents=True, exist_ok=True)
    (runtime / "R/formal/r_single_yearly_choice.csv").write_text(
        'year,policy_id,policy\n2026,base_CS,"{""policy_id"":""base_CS""}"\n',
        encoding="utf-8-sig",
    )

    pending = optimizer / "state/c_delay1_pending.json"
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.write_text(
        '{"queued_date":"2026-08-23","action":"BUY","symbol":"510880","name":"红利ETF"}',
        encoding="utf-8",
    )

    original_read_text = Path.read_text

    def cp950_default_guard(self: Path, encoding=None, errors=None):
        if self == pending and encoding is None:
            raise UnicodeDecodeError("cp950", b"\x88", 0, 1, "illegal multibyte sequence")
        return original_read_text(self, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", cp950_default_guard)

    def runner(root, target, dpos):
        return {
            "C": NormalizedSignal(
                "C", target, "BUY", "510880", "红利ETF", True, "OK", "same_day_preclose_buy", "", None, {}
            ),
            "S": NormalizedSignal(
                "S", target, "HOLD", "", "", True, "OK", "wait_idle", "", None, {}
            ),
            "D": NormalizedSignal(
                "D", target, "HOLD", "", "", True, "OK", "空仓等待", "", None, {}
            ),
        }, []

    result = run_1445_cycle(
        optimizer,
        runtime,
        FileMinuteProvider(feed),
        date(2026, 8, 24),
        time(14, 45),
        1,
        adapter_runner=runner,
        shadow_config={"C_DELAY1": {"enabled": True}, "D_STRICT": {"enabled": False}},
    )

    assert result["status"] == "OK"
    assert result["shadow_count"] == 1
