# Getting Started Guide

## Prerequisites
- **Python**: 3.13+
- **Node.js**: 20+ & npm
- **Git**: 2.30+
- **Docker & Docker Compose** (optional, for containerized deployment)

---

## 1. Local Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/example/sp500-autonomous-trading-platform.git
cd sp500-autonomous-trading-platform
```

### Step 2: Python Virtual Environment
```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### Step 3: Frontend Dependencies & Production Build
```bash
cd frontend
npm install
npm run build
cd ..
```

### Step 4: Environment Configuration
```bash
cp .env.example .env
```
*Review `.env` to ensure `EXECUTION_MODE=PAPER` and `ENVIRONMENT=development` (or `production`).*

---

## 2. Pre-Flight Verification & Testing

### Run Pre-Flight Deployment Check
```bash
python backend/scripts/deploy_check.py
```

### Run Automated Pytest Suite
```bash
# Full regression suite (366 tests)
pytest backend/tests/ -v

# Cross-phase system validation suite (20 tests)
pytest backend/tests/test_phase21_validation.py -v
```

### Run Frontend Linter
```bash
cd frontend && npx oxlint && cd ..
```

---

## 3. Running the Application

### Option A: Local Development Server
Start the unified FastAPI server (which automatically serves the compiled frontend static dashboard):
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Navigate to `http://localhost:8000` to access the Web Dashboard.

### Option B: Production Container Deployment
```bash
docker-compose -f docker-compose.prod.yml up -d --build
```
Verify container health:
```bash
docker ps
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
```

---

## 4. End-to-End Smoke Test
Run the deployment smoke test script against a running server:
```bash
python backend/scripts/smoke_test.py --base-url http://localhost:8000
```
