"""
Deployment Pre-Flight Check Script (Phase 20 & Audit Update).
Validates runtime environment, Python runtime, worker configuration,
storage volume permissions, secret redaction, database/Redis reachability,
frontend bundle presence, and execution mode readiness (PAPER vs LIVE).
Strictly returns exit code 0 on success, and exit code 1 on ANY validation failure.
"""

import os
from pathlib import Path
import socket
import sys
from urllib.parse import urlparse

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.config.settings import EnvironmentEnum, ExecutionModeEnum, Settings, get_settings


def _check_tcp_port(host: str, port: int, timeout: float = 1.0) -> bool:
    """Attempt TCP socket connection with bounded timeout."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def _probe_service_url(url_str: str) -> tuple[bool, str]:
    """Parse URL and probe host/port reachability."""
    try:
        parsed = urlparse(url_str)
        host = parsed.hostname or "localhost"
        port = parsed.port
        if not port:
            if "postgres" in parsed.scheme:
                port = 5432
            elif "redis" in parsed.scheme:
                port = 6379
            else:
                port = 80
        reachable = _check_tcp_port(host, port, timeout=1.0)
        return reachable, f"{host}:{port}"
    except Exception as e:
        return False, str(e)


def run_preflight_checks(settings_override: Settings = None) -> bool:
    print("=" * 65)
    print("S&P 500 AI TRADING PLATFORM — DEPLOYMENT PRE-FLIGHT AUDIT")
    print("=" * 65)

    all_passed = True
    settings = settings_override or get_settings()
    is_prod = (settings.ENVIRONMENT == EnvironmentEnum.PRODUCTION)

    # 1. Python Version Check
    py_ver = sys.version_info
    print(f"[1] Python Runtime: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    if py_ver.major != 3 or py_ver.minor < 11:
        print("    ❌ FAILED: Python 3.11+ required.")
        all_passed = False
    else:
        print("    ✅ PASSED: Compatible Python runtime.")

    # 2. Settings & Single-Worker Policy
    print(f"[2] Configuration Environment: {settings.ENVIRONMENT.value.upper()}")
    print(f"    Authoritative Execution Mode: {settings.EXECUTION_MODE.value}")
    print(f"    Single-Worker Policy: {settings.WORKERS_COUNT} worker(s)")

    if settings.WORKERS_COUNT != 1:
        print("    ❌ FAILED: WORKERS_COUNT must be 1 to preserve single-process autonomous loop & watchdog.")
        all_passed = False
    else:
        print("    ✅ PASSED: Single-worker stateful architecture verified.")

    # 3. Secret Redaction Verification
    safe_cfg = settings.to_safe_dict()
    sensitive_keys = ["secret_key", "broker_secret_key", "alpaca_api_secret"]
    redaction_ok = True
    for k in sensitive_keys:
        if k in safe_cfg and safe_cfg[k] not in ("REDACTED", None):
            print(f"    ❌ FAILED: Secret key '{k}' not properly redacted in to_safe_dict.")
            redaction_ok = False
            all_passed = False
    if redaction_ok:
        print("    ✅ PASSED: Configuration secret redaction verified.")

    # 4. Storage Directories & Persistent Volumes Check
    storage_dirs = [
        Path("data"),
        Path("models"),
        Path("backtests"),
        Path("models/monitoring"),
        Path("models/paper_trading"),
        Path("models/autonomous"),
    ]
    print("[3] Persistent Storage Volumes:")
    storage_ok = True
    for d in storage_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            test_file = d / ".write_test.tmp"
            test_file.write_text("ok")
            test_file.unlink()
            print(f"    ✅ Accessible & Writable: {d}")
        except Exception as e:
            print(f"    ❌ FAILED to write to {d}: {e}")
            storage_ok = False
            all_passed = False

    # 5. Database & Redis Reachability
    print("[4] Infrastructure Services Reachability:")
    # PostgreSQL probe
    db_reachable, db_target = _probe_service_url(settings.DATABASE_URL)
    if db_reachable:
        print(f"    ✅ PostgreSQL Reachable: {db_target}")
    else:
        if is_prod:
            print(f"    ❌ FAILED: PostgreSQL unreachable at {db_target} in PRODUCTION mode.")
            all_passed = False
        else:
            print(f"    ℹ️ PostgreSQL Offline ({db_target}) — Running in local filesystem/paper storage mode.")

    # Redis probe
    redis_reachable, redis_target = _probe_service_url(settings.REDIS_URL)
    if redis_reachable:
        print(f"    ✅ Redis Reachable: {redis_target}")
    else:
        if is_prod:
            print(f"    ❌ FAILED: Redis unreachable at {redis_target} in PRODUCTION mode.")
            all_passed = False
        else:
            print(f"    ℹ️ Redis Offline ({redis_target}) — Running in in-memory state fallback mode.")

    # 6. Frontend Build Verification
    frontend_dist = Path("frontend/dist")
    print(f"[5] Static Frontend Assets ({frontend_dist}):")
    if frontend_dist.exists() and (frontend_dist / "index.html").exists():
        print("    ✅ PASSED: Frontend production bundle compiled and ready.")
    else:
        if is_prod or settings.SERVE_STATIC_FRONTEND:
            print("    ❌ FAILED: Frontend dist bundle missing (frontend/dist/index.html). Run 'npm run build' in frontend/.")
            all_passed = False
        else:
            print("    ⚠️ Notice: Frontend dist bundle not found (SERVE_STATIC_FRONTEND is false).")

    # 7. Production Configuration & Execution Mode Readiness
    print("[6] Production Configuration & Broker Readiness:")
    is_valid_prod, prod_issues = settings.validate_production_settings()
    if is_prod:
        if not is_valid_prod:
            print("    ❌ FAILED: Production configuration validation errors:")
            for issue in prod_issues:
                print(f"       - {issue}")
            all_passed = False
        else:
            print("    ✅ PASSED: Production settings validated.")

    # 8. Execution Mode Classification
    print("[7] Execution Mode Safety Gate:")
    if settings.EXECUTION_MODE == ExecutionModeEnum.LIVE:
        has_live_creds = (
            settings.ALPACA_API_KEY
            and "dev" not in settings.ALPACA_API_KEY.lower()
            and settings.ALPACA_API_SECRET
            and "dev" not in settings.ALPACA_API_SECRET.lower()
        )
        if not has_live_creds:
            print("    ❌ FAILED: System set to LIVE execution mode without verified production broker credentials.")
            all_passed = False
        else:
            print("    ⚠️ LIVE-READY: Live broker execution mode enabled with configured credentials.")
    else:
        print("    ✅ PAPER-READY: System operating in simulated PAPER execution mode (Safety Enforced).")

    print("=" * 65)
    if all_passed:
        print("PRE-FLIGHT STATUS: ALL CRITICAL CHECKS PASSED ✅")
    else:
        print("PRE-FLIGHT STATUS: FAILED ❌ (Action required on failed checks above)")
    print("=" * 65)

    return all_passed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Deployment Pre-Flight Checks")
    parser.add_argument("--env", choices=["development", "staging", "production"], help="Override environment setting")
    args = parser.parse_args()

    settings_to_check = None
    if args.env:
        # Load settings with overridden environment
        settings_to_check = Settings(ENVIRONMENT=EnvironmentEnum(args.env))

    success = run_preflight_checks(settings_override=settings_to_check)
    sys.exit(0 if success else 1)
