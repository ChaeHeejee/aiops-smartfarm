from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


CSV_ENCODINGS = ("utf-8", "utf-8-sig", "euc-kr", "cp949", "latin-1")


class TomatoReportService:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key.strip()
        self.model = model

    def generate_report(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("OpenAI API key is not configured. Set OPENAI_API_KEY in tom_aiops/config.py.")

        payload = self._build_report_payload(analysis_result)
        client = self._build_client()
        response = client.responses.create(
            model=self.model,
            instructions=(
                "You are an MLOps analyst for a tomato farm monitoring system. "
                "Write the report in Korean. "
                "Use short, clear markdown with exactly these sections: "
                "1. ML 성능, 2. 히스토리 데이터 요약, 3. 운영 판단 및 권고. "
                "In section 3, decide whether retraining is appropriate now or whether model methodology changes are needed. "
                "Base your answer only on the provided analysis summary. "
                "If evidence is weak, say so explicitly."
            ),
            input=json.dumps(payload, ensure_ascii=False, indent=2),
        )

        return {
            "model": self.model,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_markdown": response.output_text,
        }

    def _build_client(self):
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError("The 'openai' package is not installed. Install dependencies from tom_aiops/requirements.txt.") from exc

        return OpenAI(api_key=self.api_key)

    def _build_report_payload(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        target_col = str(analysis_result.get("target_col", ""))
        plot = analysis_result.get("plot") or {}
        retrain_status = analysis_result.get("retrain_status") or {}
        rmse_history = analysis_result.get("rmse_history") or []
        quad_metrics = analysis_result.get("quad_metrics") or {}
        final_dataset_path = analysis_result.get("final_dataset_path")

        historical_data = self._summarize_historical_data(
            csv_path=Path(final_dataset_path) if final_dataset_path else None,
            target_col=target_col,
        )

        return {
            "target_col": target_col,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "ml_performance": {
                "series_rmse": plot.get("rmse"),
                "series_key": plot.get("key"),
                "actual_series_points": len(plot.get("actual_series") or []),
                "predicted_series_points": len(plot.get("pred_series") or []),
                "quad_metrics": quad_metrics,
            },
            "rmse_history": self._summarize_rmse_history(rmse_history),
            "retrain_status": {
                "triggered": retrain_status.get("triggered"),
                "reason": retrain_status.get("reason"),
                "previous_rmse": retrain_status.get("previous_rmse"),
                "current_rmse": retrain_status.get("current_rmse"),
                "retrained_model_valid_rmse": (retrain_status.get("retrained_model") or {}).get("valid_rmse"),
                "retrained_model_trained_at_utc": (retrain_status.get("retrained_model") or {}).get("trained_at_utc"),
            },
            "historical_data": historical_data,
        }

    def _summarize_rmse_history(self, rmse_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rmse_history:
            return {"count": 0, "recent_entries": []}

        recent_entries = []
        for item in rmse_history[-5:]:
            recent_entries.append(
                {
                    "timestamp": item.get("timestamp"),
                    "rmse": item.get("rmse"),
                }
            )

        return {
            "count": len(rmse_history),
            "recent_entries": recent_entries,
        }

    def _summarize_historical_data(self, csv_path: Path | None, target_col: str) -> Dict[str, Any]:
        if csv_path is None or not csv_path.exists():
            return {"available": False, "reason": "final_dataset_not_found"}

        df = self._read_csv_with_fallbacks(csv_path)
        summary: Dict[str, Any] = {
            "available": True,
            "dataset_path": str(csv_path),
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
        }

        if "조사일자" in df.columns:
            date_series = pd.to_datetime(df["조사일자"], errors="coerce").dropna()
            if not date_series.empty:
                summary["date_start"] = date_series.min().date().isoformat()
                summary["date_end"] = date_series.max().date().isoformat()

        if target_col and target_col in df.columns:
            numeric_target = pd.to_numeric(df[target_col], errors="coerce").dropna()
            if not numeric_target.empty:
                summary["target_statistics"] = {
                    "latest": float(numeric_target.iloc[-1]),
                    "mean": float(numeric_target.mean()),
                    "min": float(numeric_target.min()),
                    "max": float(numeric_target.max()),
                }

            recent_records = []
            recent_df = df.tail(5).copy()
            for _, row in recent_df.iterrows():
                recent_records.append(
                    {
                        "조사일자": str(row.get("조사일자", "")),
                        target_col: row.get(target_col),
                    }
                )
            summary["recent_records"] = recent_records

        return summary

    def _read_csv_with_fallbacks(self, csv_path: Path) -> pd.DataFrame:
        for encoding in CSV_ENCODINGS:
            try:
                return pd.read_csv(csv_path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Unable to read CSV with supported encodings: {csv_path}")
