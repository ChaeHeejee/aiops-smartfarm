# tom_aiops — Smart Farm AIOps Platform

AI-powered growth forecasting and anomaly monitoring system for tomato smart farms.  
Built with LSTM, FastAPI, and LLM-based report generation.

---

## Architecture

```text
CSV Upload (growth + environment + cultivation data)
    ↓
TomatoAIOpsService (model_service.py)
├── LSTM inference  →  prediction vs. actual plot
├── RMSE drift detection  →  auto-retrain trigger
└── 2×2 quad metrics (growth / yield / temp / humidity)
    ↓
FastAPI (app.py)  →  JSON response + base64 chart
    ↓
AIOps Dashboard (index_farm_improved.html)
    ↓
LLM Report (report_service.py, GPT-4o)
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r tom_aiops/requirements.txt

# 2. Set OpenAI API key
export OPENAI_API_KEY=sk-...

# 3. Run (from 팜타스틱4_실습코드/ directory)
./tom_aiops/run.sh
```

Browser: `http://localhost:8010`

Or with Docker:

```bash
docker build -t tom-aiops .
docker run -p 8010:8010 -e OPENAI_API_KEY=sk-... tom-aiops
```

---

## API Endpoints

| Method | Path            | Description                                                  |
| ------ | --------------- | ------------------------------------------------------------ |
| `GET`  | `/api/targets`  | Available prediction targets from base dataset               |
| `POST` | `/api/analyze`  | Upload CSV + select target → prediction + drift status       |
| `POST` | `/api/retrain`  | Manually trigger model retraining for a target               |
| `POST` | `/api/report`   | Generate LLM operational report from analysis result         |

### `POST /api/analyze`

```bash
curl -X POST http://localhost:8010/api/analyze \
  -F "file=@sample_upload_one_row.csv" \
  -F "target_col=초장"
```

Response:

```json
{
  "target_col": "초장",
  "plot": {
    "plot_base64": "data:image/png;base64,...",
    "rmse": 2.34,
    "actual_series": [...],
    "pred_series": [...]
  },
  "retrain_status": {
    "triggered": false,
    "policy": "current_rmse_lower_than_previous_rmse",
    "current_rmse": 2.34,
    "previous_rmse": 3.10
  },
  "quad_metrics": {
    "생장길이": {"actual": 12.5, "predicted": 11.8, "delta": 0.7, "direction": "up"},
    ...
  },
  "rmse_history": [...]
}
```

---

## Key Implementation Details

### RMSE-based Drift Detection & Auto-Retraining

Every inference call compares the current RMSE against the previous logged value.  
If RMSE improves (current < previous), retraining is triggered automatically using the base dataset.

```python
# model_service.py
def _apply_rmse_retrain_policy(self, target_col, current_rmse):
    previous_rmse = self._latest_logged_rmse(target_col)
    if previous_rmse and current_rmse < previous_rmse:
        self.train_target(target_col)   # auto-retrain
    self._save_rmse_log(target_col, current_rmse)
```

### LSTM Model Architecture

```text
LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16) → Dense(1)
```

- Sequence length: 4 weeks
- Groups: 시군 / 농가명 / 작기 / 개체번호
- MinMaxScaler normalization per-group, with median fallback for missing values

### LLM Report Generation

Calls OpenAI GPT-4o with structured analysis payload (RMSE history, retrain status, target statistics).  
Returns a Korean markdown report with sections: ML 성능 / 히스토리 데이터 요약 / 운영 판단 및 권고.

---

## Project Structure

```text
팜타스틱4_실습코드/
├── data/
│   └── 분석.csv              # Merged training dataset (growth + env + cultivation)
└── tom_aiops/
    ├── app.py                # FastAPI app & routes
    ├── model_service.py      # LSTM training, inference, drift detection
    ├── report_service.py     # LLM-based report generation
    ├── config.py             # Config (reads OPENAI_API_KEY from env)
    ├── index_farm_improved.html  # AIOps dashboard frontend
    ├── Dockerfile
    ├── requirements.txt
    ├── run.sh
    ├── sample_upload_one_row.csv           # Single-row upload sample
    ├── sample_upload_one_farm_one_entity.csv  # Full farm entity sample
    └── artifacts/            # Trained model artifacts (auto-generated)
        └── <target_col>/
            ├── model.keras
            ├── feature_scaler.pkl
            └── metadata.json
```
