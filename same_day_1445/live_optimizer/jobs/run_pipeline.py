from __future__ import annotations
import argparse
import json
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Callable
from ..engine.orchestrator import run_optimizer
from ..ops.health import build_health_report
from ..ops.dashboard import write_dashboard
from ..providers.factory import build_provider
from .run_1445 import run_1445_cycle
from .run_close import run_close_cycle


def _load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _provider(cfg: dict[str, Any]):
    return build_provider(cfg.get('provider', {}))


def _write_run_status(root: Path, command: str, observed_at: str) -> None:
    path = Path(root) / 'ledger' / 'run_status.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _load(path) if path.exists() and path.read_text(encoding='utf-8').strip() else {}
    payload[str(command)] = observed_at
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def _has_unreconciled_close(root: Path) -> bool:
    path = Path(root) / 'state' / 'unreconciled_close.json'
    if not path.exists() or not path.read_text(encoding='utf-8').strip():
        return False
    try:
        value = _load(path)
    except (json.JSONDecodeError, OSError):
        return True
    return bool(value.get('symbols'))


def run_pipeline(command: str, optimizer_root: Path, workspace_config: Path, trade_date: str, observed_at: str | None=None, handlers: dict[str, Callable[..., Any]] | None=None) -> dict[str, Any]:
    optimizer_root = Path(optimizer_root)
    cfg = _load(Path(workspace_config))
    runtime_root = Path(cfg['runtime_root'])
    day = date.fromisoformat(str(trade_date)[:10])
    now = observed_at or datetime.now().astimezone().isoformat()
    custom = handlers or {}
    if command == 'optimizer' and _has_unreconciled_close(optimizer_root):
        return {'status': 'BLOCKED', 'reason': 'UNRECONCILED_CLOSE', 'trade_date': day.isoformat()}
    if command == 'preclose':
        handler = custom.get('preclose')
        if handler is None:
            health = build_health_report(optimizer_root, now)
            if health.get('status') == 'BLOCKED':
                return {'status': 'BLOCKED', 'reason': 'HEALTH_PREFLIGHT', 'trade_date': day.isoformat(), 'health': health}
        kwargs = {'optimizer_root': optimizer_root, 'runtime_root': runtime_root, 'provider': _provider(cfg) if handler is None else None, 'trade_date': day, 'cutoff': time.fromisoformat(cfg.get('cutoff', '14:45')), 'max_staleness_minutes': int(cfg.get('max_staleness_minutes', 2)), 'shadow_config': cfg.get('shadow', {}), 'market_proxy': cfg.get('market_proxy', '510500'), 'd_account_mode': cfg.get('d_account_mode', 'model'), 'snapshot_workers': int(cfg.get('snapshot_workers', 8))}
        result = handler(**kwargs) if handler else run_1445_cycle(**kwargs)
    elif command == 'close':
        handler = custom.get('close')
        kwargs = {'optimizer_root': optimizer_root, 'runtime_root': runtime_root, 'provider': _provider(cfg) if handler is None else None, 'trade_date': day, 'cutoff': time.fromisoformat(cfg.get('close_cutoff', '15:00')), 'max_staleness_minutes': int(cfg.get('close_staleness_minutes', 5)), 'optimizer_callback': None, 'snapshot_workers': int(cfg.get('close_workers', 8))}
        result = handler(**kwargs) if handler else run_close_cycle(**kwargs)
    elif command == 'optimizer':
        handler = custom.get('optimizer')
        result = handler(root=optimizer_root, as_of=now) if handler else run_optimizer(optimizer_root, now)
        if result is None:
            result = {'status': 'OK'}
        write_dashboard(optimizer_root)
    elif command == 'health':
        handler = custom.get('health')
        result = handler(root=optimizer_root, now=now) if handler else build_health_report(optimizer_root, now)
    else:
        raise ValueError(f'unknown pipeline command: {command}')
    if not isinstance(result, dict):
        result = {'status': 'OK', 'result': result}
    result.setdefault('status', 'OK')
    if result.get('status') != 'BLOCKED':
        _write_run_status(optimizer_root, command, now)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('preclose', 'close', 'optimizer', 'health'))
    parser.add_argument('--optimizer-root', type=Path, required=True)
    parser.add_argument('--workspace-config', type=Path, required=True)
    parser.add_argument('--date', default=date.today().isoformat())
    parser.add_argument('--observed-at')
    args = parser.parse_args()
    result = run_pipeline(args.command, args.optimizer_root, args.workspace_config, args.date, args.observed_at)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, default=str))
    raise SystemExit(2 if result.get('status') == 'BLOCKED' else 0)


if __name__ == '__main__':
    main()
