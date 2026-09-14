"""
Deployment Smoke Test Script (Phase 20).
Executes end-to-end smoke verification against running server at API_URL (default http://localhost:8000).
Tests liveness, readiness, aggregate health, deployment info, and all 17 dashboard API routes.
"""

import sys
import json
import urllib.request
import urllib.error


def smoke_test(base_url: str = "http://localhost:8000") -> bool:
    print("=" * 60)
    print(f"DEPLOYMENT SMOKE TEST: {base_url}")
    print("=" * 60)

    endpoints = [
        ("/health/live", "Liveness Probe"),
        ("/health/ready", "Readiness Probe"),
        ("/health", "Aggregate Health"),
        ("/info", "Deployment Info"),
        ("/api/dashboard/overview", "Overview"),
        ("/api/dashboard/health", "System Health"),
        ("/api/dashboard/market-data", "Market Data"),
        ("/api/dashboard/models", "Models"),
        ("/api/dashboard/signals", "Signals"),
        ("/api/dashboard/portfolio", "Portfolio"),
        ("/api/dashboard/risk", "Risk Engine"),
        ("/api/dashboard/orders", "Orders"),
        ("/api/dashboard/executions", "Executions"),
        ("/api/dashboard/broker", "Broker Status"),
        ("/api/dashboard/autonomous-loop", "Autonomous Loop"),
        ("/api/dashboard/alerts", "Alerts"),
        ("/api/dashboard/incidents", "Incidents"),
        ("/api/dashboard/kill-switch", "Kill Switch"),
        ("/api/dashboard/reconciliation", "Reconciliation"),
        ("/api/dashboard/audit", "Audit Trail"),
        ("/api/dashboard/backtest", "Backtest Summary"),
    ]

    all_passed = True
    for path, name in endpoints:
        url = f"{base_url}{path}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SmokeTest/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                status_code = resp.getcode()
                raw_body = resp.read().decode("utf-8")
                data = json.loads(raw_body)
                print(f"  ✅ {status_code} OK | {name:<22} ({path})")
        except urllib.error.HTTPError as e:
            print(f"  ❌ HTTP {e.code} | {name:<22} ({path}): {e.reason}")
            all_passed = False
        except Exception as e:
            print(f"  ❌ FAILED   | {name:<22} ({path}): {e}")
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("SMOKE TEST RESULT: 100% ENDPOINTS OPERATIONAL ✅")
    else:
        print("SMOKE TEST RESULT: FAILURES DETECTED ❌")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    success = smoke_test(target)
    sys.exit(0 if success else 1)
