from __future__ import annotations

import base64
import io
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import load_model


@dataclass
class ModelConfig:
    sequence_length: int = 4
    group_cols: Tuple[str, ...] = ("시군", "농가명", "작기", "개체번호")
    date_col: str = "조사일자"
    excluded_feature_cols: Tuple[str, ...] = (
        "품목",
        "작기",
        "개체번호",
        "줄기번호",
        "생장길이",
        "엽수",
        "화방별꽃수",
        "화방별착과수",
        "착과수",
        "비고",
    )


class TomatoAIOpsService:
    def __init__(
        self,
        base_dir: Path,
        base_analysis_csv: Path,
        train_epochs: int = 30,
        train_batch_size: int = 32,
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.base_analysis_csv = Path(base_analysis_csv)
        self.train_epochs = train_epochs
        self.train_batch_size = train_batch_size

        self.artifacts_dir = self.base_dir / "artifacts"
        self.pretrained_dir = self.base_dir.parent / "artifacts" / "tomato"
        self.uploads_dir = self.base_dir / "uploads"
        self.monitor_dir = self.base_dir / "monitoring"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.monitor_dir.mkdir(parents=True, exist_ok=True)

        self.cfg = ModelConfig()
        self._base_numeric_medians: Dict[str, float] = {}

    @staticmethod
    def _now_utc_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(math.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def _build_model(seq_len: int, feat_dim: int) -> Sequential:
        m = Sequential([
            LSTM(64, return_sequences=True, input_shape=(seq_len, feat_dim)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        m.compile(optimizer="adam", loss="mse")
        return m

    def _load_df(self, path: Path, target_col: str) -> pd.DataFrame:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df[self.cfg.date_col] = pd.to_datetime(df[self.cfg.date_col], errors="coerce")
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not found in csv")
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        for c in self.cfg.group_cols:
            if c not in df.columns:
                raise ValueError(f"필수 컬럼 누락: {c}")
        return df.dropna(subset=[*self.cfg.group_cols, self.cfg.date_col, target_col])

    def _resolve_features(self, df: pd.DataFrame, target_col: str) -> List[str]:
        blocked = set(self.cfg.excluded_feature_cols) | set(self.cfg.group_cols) | {self.cfg.date_col, target_col, "주차시작일"}
        candidates = [c for c in df.columns if c not in blocked]
        selected = []
        for c in candidates:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                selected.append(c)
        if not selected:
            raise ValueError("학습 가능한 numeric feature가 없습니다.")
        return selected

    def _prepare(self, df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> pd.DataFrame:
        out = df.copy()
        for c in feature_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out[c] = out.groupby(list(self.cfg.group_cols), dropna=False)[c].transform(lambda s: s.ffill().bfill())
        out = out.dropna(subset=[target_col, *feature_cols])
        return out

    def _make_sequences(self, df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        all_x, all_y = [], []
        for _, g in df.groupby(list(self.cfg.group_cols), dropna=False):
            g = g.sort_values(self.cfg.date_col)
            xv = g[feature_cols].to_numpy(dtype=np.float32)
            yv = g[target_col].to_numpy(dtype=np.float32)
            if len(g) <= self.cfg.sequence_length:
                continue
            for i in range(self.cfg.sequence_length, len(g)):
                all_x.append(xv[i - self.cfg.sequence_length:i])
                all_y.append(yv[i])
        if not all_x:
            return np.empty((0, self.cfg.sequence_length, len(feature_cols))), np.empty((0,))
        return np.array(all_x, dtype=np.float32), np.array(all_y, dtype=np.float32)

    def _target_dir(self, target_col: str) -> Path:
        safe = target_col.replace("/", "_")
        d = self.artifacts_dir / safe
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_supported_targets(self) -> List[str]:
        targets = set()

        # Per-target artifacts
        if self.artifacts_dir.exists():
            for p in self.artifacts_dir.glob("*/metadata.json"):
                try:
                    meta = json.loads(p.read_text(encoding="utf-8"))
                    t = meta.get("target_col") or p.parent.name
                    targets.add(str(t))
                except Exception:
                    continue

        # Notebook pretrained artifact
        p_meta = self.pretrained_dir / "metadata.json"
        if p_meta.exists():
            try:
                meta = json.loads(p_meta.read_text(encoding="utf-8"))
                t = meta.get("target_col")
                if t:
                    targets.add(str(t))
            except Exception:
                pass

        return sorted(targets)

    def _find_model_bundle(self, target_col: str) -> Tuple[Path, dict]:
        """Return (model_dir, metadata) from per-target artifacts or pretrained notebook artifacts."""
        tdir = self._target_dir(target_col)
        meta_path = tdir / "metadata.json"
        model_path = tdir / "model.keras"
        scaler_path = tdir / "feature_scaler.pkl"
        if meta_path.exists() and model_path.exists() and scaler_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return tdir, meta

        # Fallback to Tomato_LSTM_MLOps.ipynb artifact
        p_meta = self.pretrained_dir / "metadata.json"
        p_model = self.pretrained_dir / "model.keras"
        p_scaler = self.pretrained_dir / "feature_scaler.pkl"
        if p_meta.exists() and p_model.exists() and p_scaler.exists():
            meta = json.loads(p_meta.read_text(encoding="utf-8"))
            pretrained_target = meta.get("target_col")
            if pretrained_target and pretrained_target != target_col:
                raise ValueError(
                    f"사전학습 모델 타깃은 '{pretrained_target}' 입니다. "
                    f"현재 선택 타깃 '{target_col}'은 지원되지 않습니다."
                )
            return self.pretrained_dir, meta

        raise ValueError("사전학습 모델을 찾지 못했습니다. 먼저 Tomato_LSTM_MLOps.ipynb에서 모델을 저장하세요.")

    def _history_path(self) -> Path:
        return self.uploads_dir / "user_uploads.csv"

    def append_user_csv(self, csv_path: Path) -> None:
        user_df = pd.read_csv(csv_path, encoding="utf-8-sig")
        hist = self._history_path()
        if hist.exists():
            prev = pd.read_csv(hist, encoding="utf-8-sig")
            merged = pd.concat([prev, user_df], ignore_index=True)
            merged.to_csv(hist, index=False, encoding="utf-8-sig")
        else:
            user_df.to_csv(hist, index=False, encoding="utf-8-sig")

    def combined_training_df(self) -> pd.DataFrame:
        # 재학습도 기본 분석 데이터만 사용 (요청사항: 업로드 테스트 CSV는 학습에 사용하지 않음)
        return pd.read_csv(self.base_analysis_csv, encoding="utf-8-sig")

    def _save_rmse_log(self, target_col: str, rmse: float) -> None:
        p = self.monitor_dir / "rmse_history.csv"
        row = pd.DataFrame([
            {"timestamp": datetime.now().isoformat(timespec="seconds"), "target_col": target_col, "rmse": rmse}
        ])
        if p.exists():
            prev = pd.read_csv(p)
            pd.concat([prev, row], ignore_index=True).to_csv(p, index=False)
        else:
            row.to_csv(p, index=False)

    def load_rmse_history(self, target_col: str) -> List[Dict[str, object]]:
        p = self.monitor_dir / "rmse_history.csv"
        if not p.exists():
            return []
        df = pd.read_csv(p)
        df = df[df["target_col"] == target_col]
        return df.to_dict("records")

    def _latest_logged_rmse(self, target_col: str) -> float | None:
        history = self.load_rmse_history(target_col)
        if not history:
            return None

        latest = history[-1].get("rmse")
        try:
            rmse = float(latest)
        except (TypeError, ValueError):
            return None

        if np.isnan(rmse):
            return None
        return rmse

    def _apply_rmse_retrain_policy(self, target_col: str, current_rmse: float) -> Dict[str, Any]:
        previous_rmse = self._latest_logged_rmse(target_col)
        current_rmse = float(current_rmse)
        should_trigger = previous_rmse is not None and current_rmse < previous_rmse

        status: Dict[str, Any] = {
            "policy": "current_rmse_lower_than_previous_rmse",
            "current_rmse": current_rmse,
            "previous_rmse": previous_rmse,
            "triggered": should_trigger,
            "reason": None,
            "retrained_model": None,
        }

        if previous_rmse is None:
            status["reason"] = "no_previous_rmse"
        elif should_trigger:
            status["reason"] = "current_rmse_is_lower_than_previous_rmse"
            status["retrained_model"] = self.train_target(target_col)
        else:
            status["reason"] = "current_rmse_is_not_lower_than_previous_rmse"

        self._save_rmse_log(target_col, current_rmse)
        return status

    def should_retrain(self, target_col: str) -> bool:
        meta = self._target_dir(target_col) / "metadata.json"
        if not meta.exists():
            return True
        data = json.loads(meta.read_text(encoding="utf-8"))
        last = datetime.fromisoformat(data["trained_at_utc"].replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - last).days >= 90

    def ensure_model(self, target_col: str) -> None:
        # 1) 기존 저장 모델 탐색 (타깃별 또는 notebook 사전학습)
        # 2) 없으면 base 분석.csv만 사용해 해당 타깃 모델 생성
        try:
            self._find_model_bundle(target_col)
        except Exception:
            self.train_target(target_col)
            return

        # 타깃별 모델이 있고 3개월 경과 시 base 데이터로 재학습
        tdir = self._target_dir(target_col)
        if (tdir / "metadata.json").exists() and self.should_retrain(target_col):
            self.train_target(target_col)

    def _build_inference_df(self, upload_df: pd.DataFrame) -> pd.DataFrame:
        """Inference dataframe = base 분석.csv + uploaded csv (uploaded row overrides same key/date)."""
        base = pd.read_csv(self.base_analysis_csv, encoding="utf-8-sig")
        merged = pd.concat([base, upload_df], ignore_index=True)
        key = ["시군", "농가명", "작기", "개체번호", self.cfg.date_col]
        for c in key:
            if c not in merged.columns:
                merged[c] = np.nan
        merged[self.cfg.date_col] = pd.to_datetime(merged[self.cfg.date_col], errors="coerce")
        merged = merged.sort_values(self.cfg.date_col).drop_duplicates(subset=key, keep="last")
        return merged

    def _fill_features_with_fallback(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        1) 시계열 내부 ffill/bfill
        2) 여전히 비는 값은 base 분석.csv의 컬럼 중앙값으로 대체
        """
        out = df.copy()
        if not self._base_numeric_medians:
            base_all = pd.read_csv(self.base_analysis_csv, encoding="utf-8-sig")
            for col in base_all.columns:
                s = pd.to_numeric(base_all[col], errors="coerce")
                med = s.median()
                if pd.notna(med):
                    self._base_numeric_medians[col] = float(med)
        for c in feature_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out[c] = out[c].ffill().bfill()
            if out[c].isna().any():
                med = self._base_numeric_medians.get(c)
                if med is not None:
                    out[c] = out[c].fillna(med)
        return out

    def build_final_upload_dataset(self, upload_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], Path]:
        """
        요청한 흐름:
        1) 업로드 CSV에서 농가/개체 키 추출
        2) 분석.csv에서 해당 키 히스토리 추출
        3) 맨 마지막에 업로드 행을 추가(같은 날짜면 업로드 행으로 덮어쓰기)
        """
        key_map = self._pick_uploaded_key(upload_df)

        base = pd.read_csv(self.base_analysis_csv, encoding="utf-8-sig")
        base[self.cfg.date_col] = pd.to_datetime(base[self.cfg.date_col], errors="coerce")

        # 기본은 전체 키(시군/농가명/작기/개체번호), 누락 시 농가명+개체번호 fallback
        strict_cols = [c for c in self.cfg.group_cols if c in base.columns and c in upload_df.columns]
        fallback_cols = [c for c in ["농가명", "개체번호"] if c in base.columns and c in upload_df.columns]
        match_cols = strict_cols if strict_cols else fallback_cols
        if not match_cols:
            raise ValueError("업로드 CSV에서 농가/개체 식별 컬럼을 찾지 못했습니다.")

        m_base = np.ones(len(base), dtype=bool)
        m_upload = np.ones(len(upload_df), dtype=bool)
        for c in match_cols:
            val = key_map.get(c, "")
            m_base &= base[c].astype(str).eq(val)
            m_upload &= upload_df[c].astype(str).eq(val)

        hist = base.loc[m_base].copy()
        new_rows = upload_df.loc[m_upload].copy()
        new_rows[self.cfg.date_col] = pd.to_datetime(new_rows[self.cfg.date_col], errors="coerce")

        final_df = pd.concat([hist, new_rows], ignore_index=True)
        # 같은 날짜 중복이면 업로드 행(뒤쪽)이 우선
        final_df = final_df.sort_values(self.cfg.date_col).drop_duplicates(subset=[self.cfg.date_col], keep="last")
        final_df = final_df.reset_index(drop=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.uploads_dir / f"final_dataset_{ts}.csv"
        final_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        return final_df, key_map, out_path

    def _pick_uploaded_key(self, upload_df: pd.DataFrame) -> Dict[str, str]:
        """Pick the target series key from uploaded csv (latest row 기준)."""
        key_cols = list(self.cfg.group_cols)
        missing = [c for c in key_cols if c not in upload_df.columns]
        if missing:
            raise ValueError(f"업로드 CSV에 키 컬럼이 없습니다: {missing}")

        tmp = upload_df.copy()
        tmp[self.cfg.date_col] = pd.to_datetime(tmp[self.cfg.date_col], errors="coerce")
        tmp = tmp.sort_values(self.cfg.date_col)
        last = tmp.iloc[-1]
        return {c: str(last[c]) for c in key_cols}

    def train_target(
        self,
        target_col: str,
        epochs: int | None = None,
        batch_size: int | None = None,
    ) -> Dict[str, object]:
        epochs = self.train_epochs if epochs is None else epochs
        batch_size = self.train_batch_size if batch_size is None else batch_size

        raw = self.combined_training_df()
        tmp = self.base_dir / "_combined_train_tmp.csv"
        raw.to_csv(tmp, index=False, encoding="utf-8-sig")

        df = self._load_df(tmp, target_col)
        feat = self._resolve_features(df, target_col)
        df = self._prepare(df, target_col, feat)

        scaler = MinMaxScaler()
        df = df.copy()
        df.loc[:, feat] = scaler.fit_transform(df[feat])

        x, y = self._make_sequences(df, target_col, feat)
        if len(x) == 0:
            raise ValueError("학습 시퀀스가 생성되지 않았습니다.")

        sp = int(len(x) * 0.8)
        x_tr, y_tr = x[:sp], y[:sp]
        x_va, y_va = x[sp:], y[sp:]

        model = self._build_model(self.cfg.sequence_length, len(feat))
        model.fit(x_tr, y_tr, validation_data=(x_va, y_va) if len(x_va) else None, epochs=epochs, batch_size=batch_size, verbose=0)

        y_hat = model.predict(x_va, verbose=0).reshape(-1) if len(x_va) else np.array([])
        rmse = self._rmse(y_va, y_hat) if len(y_va) else float("nan")

        tdir = self._target_dir(target_col)
        model.save(tdir / "model.keras")
        pd.to_pickle(scaler, tdir / "feature_scaler.pkl")
        meta = {
            "trained_at_utc": self._now_utc_iso(),
            "target_col": target_col,
            "feature_cols": feat,
            "group_cols": list(self.cfg.group_cols),
            "sequence_length": self.cfg.sequence_length,
            "valid_rmse": rmse,
        }
        (tdir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta

    def _predict_last_series_plot(self, infer_df: pd.DataFrame, target_col: str, key_map: Dict[str, str]) -> Dict[str, object]:
        tdir, meta = self._find_model_bundle(target_col)
        feat = list(meta["feature_cols"])

        df = infer_df.copy()
        df[self.cfg.date_col] = pd.to_datetime(df[self.cfg.date_col], errors="coerce")
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        for c in feat:
            if c not in df.columns:
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce")

        key_cols = [*self.cfg.group_cols]
        srt = [self.cfg.date_col, *key_cols]
        df = df.sort_values(srt).reset_index(drop=True)

        # 업로드된 농가/개체 키를 명시적으로 선택해서 그 시계열만 사용
        mask = np.ones(len(df), dtype=bool)
        for c in key_cols:
            mask &= df[c].astype(str).eq(key_map[c])
        g = df.loc[mask].sort_values(self.cfg.date_col).reset_index(drop=True).copy()
        if g.empty:
            raise ValueError("업로드한 농가/개체 키에 해당하는 히스토리를 찾지 못했습니다.")

        g = self._fill_features_with_fallback(g, feat)
        g = g.dropna(subset=[target_col, *feat])
        if len(g) <= self.cfg.sequence_length:
            raise ValueError(
                f"해당 농가/개체 시계열 길이가 부족합니다. "
                f"(현재 {len(g)}주, 최소 {self.cfg.sequence_length + 1}주 필요)"
            )

        key = tuple(key_map[c] for c in key_cols)
        model = load_model(tdir / "model.keras")
        scaler: MinMaxScaler = pd.read_pickle(tdir / "feature_scaler.pkl")

        s = g.copy()
        s.loc[:, feat] = scaler.transform(s[feat])

        dates, actuals, preds = [], [], []
        for i in range(self.cfg.sequence_length, len(s)):
            x = s.iloc[i - self.cfg.sequence_length:i][feat].to_numpy(dtype=np.float32)
            y_hat = float(model.predict(np.expand_dims(x, axis=0), verbose=0).reshape(-1)[0])
            dates.append(g.loc[i, self.cfg.date_col])
            actuals.append(float(g.loc[i, target_col]))
            preds.append(y_hat)

        rmse = self._rmse(np.array(actuals), np.array(preds))

        plt.figure(figsize=(9, 5))
        plt.plot(dates, actuals, marker="o", label=f"실측값({target_col})")
        plt.plot(dates, preds, marker="o", label=f"예측값({target_col})")
        plt.title(f"{target_col} 예측 vs 실측")
        plt.xlabel("조사일자")
        plt.ylabel(target_col)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=140)
        plt.close()
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "plot_base64": f"data:image/png;base64,{b64}",
            "rmse": rmse,
            "key": {
                "시군": str(key[0]),
                "농가명": str(key[1]),
                "작기": str(key[2]),
                "개체번호": str(key[3]),
            },
            "actual_series": actuals,
            "pred_series": preds,
        }

    def _predict_last_point(self, infer_df: pd.DataFrame, target_col: str, key_map: Dict[str, str]) -> Dict[str, object]:
        """Predict target at the last available point and compare with actual."""
        try:
            tdir, meta = self._find_model_bundle(target_col)
        except ValueError:
            # 사전학습 모델이 없는 지표는 UI 카드에서 표시 불가
            return {"actual": None, "predicted": None, "delta": None, "direction": "flat"}
        feat = list(meta["feature_cols"])

        df = infer_df.copy()
        df[self.cfg.date_col] = pd.to_datetime(df[self.cfg.date_col], errors="coerce")
        if target_col not in df.columns:
            return {"actual": None, "predicted": None, "delta": None, "direction": "flat"}
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        for c in feat:
            if c not in df.columns:
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce")

        key_cols = list(self.cfg.group_cols)
        mask = np.ones(len(df), dtype=bool)
        for c in key_cols:
            mask &= df[c].astype(str).eq(key_map[c])
        chosen = df.loc[mask].sort_values(self.cfg.date_col).reset_index(drop=True).copy()
        chosen = self._fill_features_with_fallback(chosen, feat)
        chosen = chosen.dropna(subset=[target_col, *feat])
        if len(chosen) <= self.cfg.sequence_length:
            return {"actual": None, "predicted": None, "delta": None, "direction": "flat"}

        model = load_model(tdir / "model.keras")
        scaler: MinMaxScaler = pd.read_pickle(tdir / "feature_scaler.pkl")

        g = chosen.copy()
        g.loc[:, feat] = scaler.transform(g[feat])
        i = len(g) - 1
        x = g.iloc[i - self.cfg.sequence_length:i][feat].to_numpy(dtype=np.float32)
        if x.shape[0] != self.cfg.sequence_length:
            return {"actual": None, "predicted": None, "delta": None, "direction": "flat"}
        pred = float(model.predict(np.expand_dims(x, axis=0), verbose=0).reshape(-1)[0])
        actual = float(chosen.iloc[-1][target_col])
        diff = actual - pred
        direction = "down" if diff < 0 else ("up" if diff > 0 else "flat")
        return {"actual": actual, "predicted": pred, "delta": abs(diff), "direction": direction}

    def _build_quad_metrics(self, infer_df: pd.DataFrame, key_map: Dict[str, str]) -> Dict[str, Dict[str, object]]:
        fruit_col = "화방별착과수" if "화방별착과수" in infer_df.columns else ("착과수" if "착과수" in infer_df.columns else None)
        temp_col = "주간평균_온도_내부" if "주간평균_온도_내부" in infer_df.columns else ("온도_내부" if "온도_내부" in infer_df.columns else None)
        humid_col = "주간평균_상대습도_내부" if "주간평균_상대습도_내부" in infer_df.columns else ("상대습도_내부" if "상대습도_내부" in infer_df.columns else None)

        mapping = {
            "생장길이": "생장길이",
            "착과수": fruit_col,
            "온도": temp_col,
            "습도": humid_col,
        }

        out: Dict[str, Dict[str, object]] = {}
        for label, target in mapping.items():
            if not target:
                out[label] = {"actual": None, "predicted": None, "delta": None, "direction": "flat"}
                continue
            self.ensure_model(target)
            out[label] = self._predict_last_point(infer_df, target, key_map)
        return out

    def run_inference(self, uploaded_csv_path: Path, target_col: str) -> Dict[str, object]:
        # 추론 전용: 업로드 CSV는 학습 데이터에 누적하지 않음
        self.ensure_model(target_col)

        upload_df = pd.read_csv(uploaded_csv_path, encoding="utf-8-sig")
        final_df, key_map, final_dataset_path = self.build_final_upload_dataset(upload_df)
        plot_data = self._predict_last_series_plot(final_df, target_col, key_map)
        retrain_status = self._apply_rmse_retrain_policy(target_col, plot_data["rmse"])
        quad = self._build_quad_metrics(final_df, key_map)
        rmse_history = self.load_rmse_history(target_col)

        return {
            "target_col": target_col,
            "plot": plot_data,
            "retrain_status": retrain_status,
            "quad_metrics": quad,
            "rmse_history": rmse_history,
            "final_dataset_path": str(final_dataset_path),
        }


def available_targets(csv_path: Path) -> List[str]:
    df = pd.read_csv(csv_path, nrows=200, encoding="utf-8-sig")
    blocked = {"도", "시군", "품목", "작기", "농가명", "조사일자", "개체번호", "줄기번호", "비고", "주차시작일"}
    out = []
    for c in df.columns:
        if c in blocked:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            out.append(c)
    return out
