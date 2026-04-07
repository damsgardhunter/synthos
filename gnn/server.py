"""
server.py
=========
FastAPI service wrapping the PyTorch SuperconductorGNN.
Runs on the GCP instance (localhost:8765).
Launched by gcp-worker/gnn-loop.ts as a managed subprocess.

Endpoints
---------
GET  /health          → service status, model info
POST /train           → full ensemble training from TrainingSample[] JSON
POST /predict         → single-formula prediction with MC Dropout + OOD
GET  /metrics         → last training job metrics

Install
-------
pip install fastapi uvicorn[standard] torch torch-geometric asyncpg
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Path setup (server.py lives in gnn/ next to superconductor_gnn.py) ────────
sys.path.insert(0, str(Path(__file__).parent))

from superconductor_gnn import (
    SuperconductorGNN, GNNTrainer, GNNOutput,
    MahalanobisOOD, train_ensemble, ensemble_predict,
    SuperconGraph, ENSEMBLE_SIZE, TC_MAX_K,
)
from graph_builder import build_crystal_graph

# ── Config ────────────────────────────────────────────────────────────────────
PORT          = int(os.environ.get("GNN_SERVICE_PORT", "8765"))
WEIGHTS_DIR   = Path(os.environ.get("GNN_WEIGHTS_DIR", "/opt/qae/gnn_weights"))
MC_PASSES     = int(os.environ.get("GNN_MC_PASSES", "10"))
LOG_LEVEL     = os.environ.get("GNN_LOG_LEVEL", "INFO")

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="[GNN-Service] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gnn-service")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}")

# ── Global state ──────────────────────────────────────────────────────────────
_ensemble:    List[SuperconductorGNN] = []
_ood:         MahalanobisOOD          = MahalanobisOOD()
_last_metrics: Dict[str, Any]         = {}
_train_lock   = asyncio.Lock()

# ── XGBoost ───────────────────────────────────────────────────────────────────
import pickle

XGB_PATH   = WEIGHTS_DIR / "xgb_model.pkl"
_xgb_model = None

def load_xgb():
    global _xgb_model
    if XGB_PATH.exists():
        with open(XGB_PATH, "rb") as f:
            _xgb_model = pickle.load(f)
        log.info(f"XGBoost model loaded from {XGB_PATH}")
    else:
        log.warning(f"No XGBoost model found at {XGB_PATH}")

app = FastAPI(title="Quantum Alchemy GNN Service", version="2.0.0")

# ══════════════════════════════════════════════════════════════════════════════
# Pydantic schemas (matching TypeScript TrainingSample interface)
# ══════════════════════════════════════════════════════════════════════════════

class StructureInfo(BaseModel):
    spaceGroup:    Optional[str]   = None
    crystalSystem: Optional[str]   = None
    dimensionality: Optional[str]  = None

class TrainingSample(BaseModel):
    formula:         str
    tc:              float
    formationEnergy: Optional[float] = None
    bandgap:         Optional[float] = None
    lam:             Optional[float] = Field(None, alias="lambda")
    omegaLog:        Optional[float] = None     # K
    muStar:          Optional[float] = None
    dataConfidence:  Optional[str]   = None
    structure:       Optional[StructureInfo] = None
    prototype:       Optional[str]   = None
    sourceTag:       Optional[str]   = None

    class Config:
        populate_by_name = True

class TrainRequest(BaseModel):
    job_id:              int
    training_data:       List[TrainingSample]
    max_pretrain_epochs: int = 15
    # Metrics for quality gate check (from TS startup weights)
    startup_val_r2:      Optional[float] = None

class PredictRequest(BaseModel):
    formula:      str
    structure:    Optional[StructureInfo] = None
    pressure_gpa: float = 0.0
    lambda_hint:  Optional[float] = None
    omega_hint:   Optional[float] = None
    dos_hint:     Optional[float] = None
    fe_hint:      Optional[float] = None
    mc_passes:    int = MC_PASSES

class PredictResponse(BaseModel):
    tc:               float
    omega_log:        float
    formation_energy: float
    lam:              float
    bandgap:          float
    dos_proxy:        float
    stability_prob:   float
    confidence:       float
    phonon_stable:    float
    tc_var:           float
    lambda_var:       float
    latent_distance:  float
    tc_ci95_lo:       float
    tc_ci95_hi:       float
    epistemic_std:    float
    aleatoric_std:    float
    total_std:        float

class MetricsResponse(BaseModel):
    job_id:          Optional[int]   = None
    r2:              float           = 0.0
    mae:             float           = 0.0
    rmse:            float           = 0.0
    train_r2:        float           = 0.0
    train_mae:       float           = 0.0
    val_n:           int             = 0
    ci95_coverage:   float           = 0.0
    ci95_width:      float           = 0.0
    wall_seconds:    float           = 0.0
    n_samples:       int             = 0
    model_path:      Optional[str]   = None
    n_models:        int             = 0

class TrainResponse(BaseModel):
    job_id:        int
    status:        str    # "done" | "discarded" | "failed"
    reason:        Optional[str] = None
    metrics:       MetricsResponse

# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def sample_to_physics_hints(s: TrainingSample) -> Dict:
    hints: Dict = {}
    if s.lam        is not None: hints["lambda"]           = s.lam
    if s.omegaLog   is not None: hints["omega_log_k"]      = s.omegaLog
    if s.formationEnergy is not None: hints["formation_energy"] = s.formationEnergy
    if s.sourceTag  is not None: hints["source_tag"]       = s.sourceTag
    return hints

def sample_to_graph(s: TrainingSample) -> Optional[SuperconGraph]:
    try:
        structure = s.structure.model_dump() if s.structure else None
        hints     = sample_to_physics_hints(s)
        g = build_crystal_graph(
            formula       = s.formula,
            structure     = structure,
            physics_hints = hints,
            pressure_gpa  = 0.0,
        )
        g.target_tc     = s.tc       if s.tc > 0 else None
        g.target_fe     = s.formationEnergy
        g.target_lambda = s.lam
        g.target_omega  = s.omegaLog
        g.target_bg     = s.bandgap
        g.target_psc    = 1.0 if s.tc > 0 else 0.0
        g.formula       = s.formula
        return g
    except Exception as e:
        log.debug(f"Failed to build graph for {s.formula}: {e}")
        return None


def _split_train_val(
    graphs: List[SuperconGraph],
    val_frac: float = 0.20,
    seed: int = 42,
) -> tuple[List[SuperconGraph], List[SuperconGraph]]:
    """
    Stratified 80/20 split: SC samples stratified, Tc=0 all into train.
    Matches splitTrainValidation() in graph-neural-net.ts.
    """
    import random
    rng = random.Random(seed)

    sc  = [g for g in graphs if g.target_tc and g.target_tc > 0]
    nsc = [g for g in graphs if not g.target_tc or g.target_tc <= 0]

    rng.shuffle(sc)
    split = max(1, int(len(sc) * val_frac))
    val   = sc[:split]
    train = sc[split:] + nsc
    return train, val


# ══════════════════════════════════════════════════════════════════════════════
# Metrics helpers (matching TS computeMetrics / computeCalibration)
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    models: List[SuperconductorGNN],
    graphs: List[SuperconGraph],
    max_n:  int = 500,
) -> Dict[str, float]:
    """R², MAE, RMSE on held-out graphs."""
    if not models or not graphs:
        return {"r2": 0.0, "mae": 0.0, "rmse": 0.0, "n": 0}

    actuals, preds = [], []
    for g in graphs[:max_n]:
        if g.target_tc is None or g.target_tc <= 0:
            continue
        try:
            out = models[0].predict(g, device, mc_passes=1)
            tc  = float(out.tc.item())
            if not math.isfinite(tc) or tc > 500:
                continue
            actuals.append(g.target_tc)
            preds.append(tc)
        except Exception:
            pass

    if len(actuals) < 2:
        return {"r2": 0.0, "mae": 0.0, "rmse": 0.0, "n": len(actuals)}

    mean_a  = sum(actuals) / len(actuals)
    ss_tot  = sum((a - mean_a) ** 2 for a in actuals)
    ss_res  = sum((a - p) ** 2 for a, p in zip(actuals, preds))
    abs_err = [abs(a - p) for a, p in zip(actuals, preds)]

    r2   = 1 - ss_res / max(ss_tot, 1e-10)
    mae  = sum(abs_err) / len(abs_err)
    rmse = math.sqrt(sum(e**2 for e in abs_err) / len(abs_err))
    return {"r2": r2, "mae": mae, "rmse": rmse, "n": len(actuals)}


def compute_calibration(
    models:  List[SuperconductorGNN],
    graphs:  List[SuperconGraph],
) -> Dict[str, float]:
    """CI95 empirical coverage — should be ≈ 0.95 for calibrated ensemble."""
    if not models or len(models) < 2 or not graphs:
        return {"coverage": 0.0, "mean_width": 0.0, "n": 0}

    inside, widths, counted = 0, [], 0
    for g in graphs[:200]:
        if g.target_tc is None or g.target_tc <= 0:
            continue
        try:
            preds = [float(m.predict(g, device, mc_passes=1).tc.item()) for m in models]
            mean_tc = sum(preds) / len(preds)
            std_tc  = math.sqrt(sum((p - mean_tc)**2 for p in preds) / len(preds))
            lo = mean_tc - 1.96 * std_tc
            hi = mean_tc + 1.96 * std_tc
            if not (math.isfinite(lo) and math.isfinite(hi)):
                continue
            if lo <= g.target_tc <= hi:
                inside += 1
            widths.append(hi - lo)
            counted += 1
        except Exception:
            pass

    if counted == 0:
        return {"coverage": 0.0, "mean_width": 0.0, "n": 0}
    return {
        "coverage":   inside / counted,
        "mean_width": sum(widths) / len(widths),
        "n":          counted,
    }


def fit_ood(models: List[SuperconductorGNN], graphs: List[SuperconGraph]):
    """Fit Mahalanobis OOD detector on training latent embeddings."""
    global _ood
    embeddings = []
    for g in graphs[:200]:
        try:
            out = models[0].predict(g, device, mc_passes=1)
            embeddings.append(out.latent[0])
        except Exception:
            pass
    if len(embeddings) >= 10:
        emb_stack = torch.stack(embeddings, dim=0)
        _ood.fit(emb_stack)
        log.info(f"OOD detector fitted on {len(embeddings)} embeddings")


# ══════════════════════════════════════════════════════════════════════════════
# Ensemble persistence
# ══════════════════════════════════════════════════════════════════════════════

def save_ensemble(models: List[SuperconductorGNN], job_id: int) -> str:
    path = WEIGHTS_DIR / f"ensemble_job{job_id}.pt"
    state_dicts = [m.state_dict() for m in models]
    torch.save(state_dicts, path)
    log.info(f"Saved {len(models)}-model ensemble to {path}")
    return str(path)


def load_ensemble(path: str) -> List[SuperconductorGNN]:
    """
    Load an ensemble from a .pt file.
    Handles two formats:
      • list of state_dicts  — saved by save_ensemble() or the repackage script
      • single state_dict    — saved directly by Colab (torch.save(model.state_dict()))
    """
    raw = torch.load(path, map_location=device, weights_only=True)

    # Normalise: always work with a list of state_dicts
    if isinstance(raw, dict):
        if "model_state" in raw:
            # Training checkpoint format: {model_state, optimizer_state, epoch, ...}
            state_dicts = [raw["model_state"]]
        else:
            # Plain state dict (torch.save(model.state_dict()))
            state_dicts = [raw]
    elif isinstance(raw, list) and all(isinstance(s, dict) for s in raw):
        # Each element might also be a training checkpoint
        state_dicts = [s["model_state"] if "model_state" in s else s for s in raw]
    else:
        raise ValueError(f"Unrecognised checkpoint format in {path}: {type(raw)}")

    models = []
    for sd in state_dicts:
        m = SuperconductorGNN().to(device)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    log.info(f"Loaded {len(models)}-model ensemble from {path}")
    return models


def load_latest_ensemble() -> Optional[List[SuperconductorGNN]]:
    """
    Find the most-recently-modified .pt file in WEIGHTS_DIR.
    Accepts any filename — ensemble_job*.pt from GNN service, or
    best_model.pt / checkpoint_*.pt from Colab uploads.
    """
    pts = sorted(WEIGHTS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pts:
        return None
    for pt in pts:
        try:
            models = load_ensemble(str(pt))
            log.info(f"Active weights: {pt.name}  ({len(models)} model(s))")
            return models
        except Exception as e:
            log.warning(f"Skipping {pt.name}: {e}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# HTTP Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    global _ensemble
    log.info("GNN service starting…")
    cached = await asyncio.get_event_loop().run_in_executor(None, load_latest_ensemble)
    if cached:
        _ensemble = cached
        log.info(f"Loaded {len(_ensemble)} cached models from disk")
    else:
        log.info("No cached ensemble found — will train on first /train request")
    await asyncio.get_event_loop().run_in_executor(None, load_xgb)


@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "n_models":     len(_ensemble),
        "device":       str(device),
        "cuda":         torch.cuda.is_available(),
        "weights_dir":  str(WEIGHTS_DIR),
        "xgb_loaded":   _xgb_model is not None,
    }


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    return MetricsResponse(**_last_metrics) if _last_metrics else MetricsResponse()


class XGBPredictRequest(BaseModel):
    formula:      str
    pressure_gpa: float = 0.0

class XGBPredictResponse(BaseModel):
    tc:         float
    r2:         float
    n_features: int
    source:     str = "colab-xgb"

@app.post("/predict-xgb", response_model=XGBPredictResponse)
async def predict_xgb(req: XGBPredictRequest):
    """
    Predict Tc using the Colab-trained XGBoost model (xgb_model.pkl).
    Features are derived from the formula via graph_builder's global composition
    vector with pressure_gpa appended as the final column (23+1 = 24 features,
    matching the Cell 11 extractor used during Colab training).
    """
    if _xgb_model is None:
        raise HTTPException(status_code=503, detail="XGBoost model not loaded")
    try:
        graph = build_crystal_graph(req.formula, pressure_gpa=req.pressure_gpa)
        import numpy as np
        base     = np.array(graph.global_features, dtype=np.float32)
        pressure = np.array([getattr(graph, "pressure_gpa", req.pressure_gpa)], dtype=np.float32)
        x        = np.concatenate([base, pressure]).reshape(1, -1)
        tc_pred  = float(_xgb_model.predict(x)[0])
        tc_pred = max(0.0, tc_pred)
        return XGBPredictResponse(
            tc=round(tc_pred, 2),
            r2=0.9117,   # from xgb_metadata.json
            n_features=int(x.shape[1]),
            source="colab-xgb",
        )
    except Exception as e:
        log.warning(f"XGBoost predict failed for {req.formula}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", response_model=TrainResponse)
async def train(req: TrainRequest):
    """
    Full ensemble training from TrainingSample[] payload.

    Flow:
      1. Convert formula strings → SuperconGraph tensors (parallel)
      2. Stratified 80/20 train/val split (SC samples stratified, Tc=0 all to train)
      3. Cap Tc=0 contrast to 1:1 SC ratio
      4. Train ENSEMBLE_SIZE models in parallel (one per CPU thread)
      5. Evaluate on held-out val set
      6. Quality gate: discard if R²<-5 or MAE>200K or worse than startup
      7. Save .pt to disk, update _ensemble
      8. Fit OOD detector
      9. Return metrics for TS to write to gnn_training_jobs
    """
    global _ensemble, _last_metrics

    async with _train_lock:   # Only one training job at a time
        job_id = req.job_id
        t0     = time.perf_counter()

        log.info(f"[Job#{job_id}] Building graphs for {len(req.training_data)} samples…")

        # Build graphs in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        all_graphs = await loop.run_in_executor(
            None,
            lambda: [g for s in req.training_data if (g := sample_to_graph(s)) is not None],
        )

        log.info(f"[Job#{job_id}] Built {len(all_graphs)} valid graphs "
                 f"({len(req.training_data) - len(all_graphs)} skipped)")

        if len(all_graphs) < 10:
            return TrainResponse(
                job_id=job_id, status="failed",
                reason=f"Only {len(all_graphs)} valid graphs (need ≥10)",
                metrics=MetricsResponse(),
            )

        # Split
        train_graphs, val_graphs = _split_train_val(all_graphs, val_frac=0.20, seed=42)

        # Cap Tc=0 contrast
        sc_train  = [g for g in train_graphs if g.target_tc and g.target_tc > 0]
        nsc_train = [g for g in train_graphs if not g.target_tc or g.target_tc <= 0]
        capped    = nsc_train[:len(sc_train)]
        train_set = sc_train + capped

        log.info(
            f"[Job#{job_id}] train={len(train_set)} "
            f"(sc={len(sc_train)}, contrast={len(capped)}) | val={len(val_graphs)}"
        )

        # Train ensemble (CPU threads — one per model)
        try:
            models = await loop.run_in_executor(
                None,
                lambda: train_ensemble(
                    all_graphs,     # pass full set; trainer does internal split
                    size=ENSEMBLE_SIZE,
                    device=device,
                    n_epochs=None,  # auto-computed from dataset size
                    batch_size=min(64, len(train_set)),
                    pretrain_fe=True,
                ),
            )
        except Exception as e:
            log.error(f"[Job#{job_id}] Training failed: {e}", exc_info=True)
            return TrainResponse(
                job_id=job_id, status="failed",
                reason=str(e)[:500],
                metrics=MetricsResponse(),
            )

        wall_sec = time.perf_counter() - t0

        # ── Evaluate on held-out val set ──────────────────────────────────────
        val_m   = await loop.run_in_executor(None, lambda: compute_metrics(models, val_graphs))
        train_m = await loop.run_in_executor(
            None, lambda: compute_metrics(models, sc_train[:200])
        )
        cal     = await loop.run_in_executor(
            None, lambda: compute_calibration(models, val_graphs)
        )

        r2   = val_m["r2"];  mae  = val_m["mae"];  rmse = val_m["rmse"]
        val_n = int(val_m["n"])

        log.info(
            f"[Job#{job_id}] {wall_sec:.0f}s | "
            f"VAL(n={val_n}) R²={r2:.3f} MAE={mae:.1f}K RMSE={rmse:.1f}K | "
            f"TRAIN R²={train_m['r2']:.3f} MAE={train_m['mae']:.1f}K | "
            f"CI95-cov={cal['coverage']*100:.1f}% width={cal['mean_width']:.1f}K"
        )

        if cal['coverage'] < 0.80:
            log.warning(f"[Job#{job_id}] ⚠ CI95 under-coverage ({cal['coverage']*100:.1f}%)")
        elif cal['coverage'] > 0.99:
            log.warning(f"[Job#{job_id}] ⚠ CI95 over-coverage ({cal['coverage']*100:.1f}%)")

        # ── Quality gate ──────────────────────────────────────────────────────
        if r2 < -5 or mae > 200 or rmse > 100:
            reason = f"Quality below threshold (R²={r2:.3f}, MAE={mae:.1f}K, RMSE={rmse:.1f}K)"
            log.warning(f"[Job#{job_id}] Discarding — {reason}")
            return TrainResponse(
                job_id=job_id, status="discarded", reason=reason,
                metrics=MetricsResponse(r2=r2, mae=mae, rmse=rmse, val_n=val_n),
            )

        if req.startup_val_r2 is not None and r2 < req.startup_val_r2 - 0.05:
            reason = f"Val R²={r2:.3f} worse than startup R²={req.startup_val_r2:.3f}"
            log.warning(f"[Job#{job_id}] Discarding — {reason}")
            return TrainResponse(
                job_id=job_id, status="discarded", reason=reason,
                metrics=MetricsResponse(r2=r2, mae=mae, rmse=rmse, val_n=val_n),
            )

        # ── Save ensemble ─────────────────────────────────────────────────────
        model_path = await loop.run_in_executor(
            None, lambda: save_ensemble(models, job_id)
        )

        # ── Activate new ensemble + fit OOD ──────────────────────────────────
        _ensemble = models
        await loop.run_in_executor(None, lambda: fit_ood(models, train_set[:200]))

        metrics = MetricsResponse(
            job_id       = job_id,
            r2           = r2,
            mae          = mae,
            rmse         = rmse,
            train_r2     = train_m["r2"],
            train_mae    = train_m["mae"],
            val_n        = val_n,
            ci95_coverage= cal["coverage"],
            ci95_width   = cal["mean_width"],
            wall_seconds = wall_sec,
            n_samples    = len(req.training_data),
            model_path   = model_path,
            n_models     = len(models),
        )
        _last_metrics = metrics.model_dump()

        return TrainResponse(job_id=job_id, status="done", metrics=metrics)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Single-formula prediction.
    Uses the current ensemble with MC Dropout for uncertainty quantification.
    """
    if not _ensemble:
        raise HTTPException(503, "No trained models available — POST /train first")

    loop = asyncio.get_event_loop()

    def _run() -> PredictResponse:
        hints: Dict = {}
        if req.lambda_hint is not None: hints["lambda"]           = req.lambda_hint
        if req.omega_hint  is not None: hints["omega_log_k"]      = req.omega_hint
        if req.dos_hint    is not None: hints["dos_at_ef"]         = req.dos_hint
        if req.fe_hint     is not None: hints["formation_energy"]  = req.fe_hint

        structure = req.structure.model_dump() if req.structure else None

        try:
            graph = build_crystal_graph(
                formula       = req.formula,
                structure     = structure,
                physics_hints = hints,
                pressure_gpa  = req.pressure_gpa,
            )
        except Exception as e:
            raise HTTPException(400, f"Graph build failed for {req.formula!r}: {e}")

        # Ensemble + MC Dropout
        if len(_ensemble) > 1:
            out = ensemble_predict(_ensemble, graph, device)
        else:
            out = _ensemble[0].predict(graph, device, mc_passes=req.mc_passes)

        tc      = float(out.tc.item())
        tc_var  = float(out.tc_var.item())
        tc_std  = math.sqrt(max(tc_var, 0))
        aleatoric_var = float(out.tc_var.item())  # from variance head
        if len(_ensemble) > 1:
            # Epistemic from ensemble spread; aleatoric from variance head
            preds = [float(m.predict(graph, device, mc_passes=1).tc.item()) for m in _ensemble]
            mean_tc  = sum(preds) / len(preds)
            epi_std  = math.sqrt(sum((p - mean_tc)**2 for p in preds) / len(preds))
        else:
            epi_std = 0.0
        ale_std = math.sqrt(max(aleatoric_var / (TC_MAX_K ** 2), 0))  # normalised
        tot_std = math.sqrt(epi_std**2 + ale_std**2 * TC_MAX_K**2)

        # OOD distance
        lat_dist = _ood.distance(out.latent[0].detach())

        return PredictResponse(
            tc               = tc,
            omega_log        = float(out.omega_log.item()),
            formation_energy = float(out.formation_energy.item()),
            lam              = float(out.lam.item()),
            bandgap          = float(out.bandgap.item()),
            dos_proxy        = float(out.dos_proxy.item()),
            stability_prob   = float(out.stability_prob.item()),
            confidence       = float(out.confidence.item()),
            phonon_stable    = float(out.phonon_stable.item()),
            tc_var           = tc_var,
            lambda_var       = float(out.lambda_var.item()),
            latent_distance  = lat_dist,
            tc_ci95_lo       = max(0.0, tc - 1.96 * tc_std),
            tc_ci95_hi       = min(TC_MAX_K, tc + 1.96 * tc_std),
            epistemic_std    = epi_std,
            aleatoric_std    = ale_std * TC_MAX_K,
            total_std        = tot_std,
        )

    return await loop.run_in_executor(None, _run)


@app.post("/reload")
async def reload_weights(path: Optional[str] = None):
    """Hot-reload ensemble weights from disk."""
    global _ensemble
    loop = asyncio.get_event_loop()
    if path:
        models = await loop.run_in_executor(None, lambda: load_ensemble(path))
    else:
        models = await loop.run_in_executor(None, load_latest_ensemble)
    if not models:
        raise HTTPException(404, "No weights found to load")
    _ensemble = models
    return {"loaded": len(_ensemble), "path": path}


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info(f"Starting GNN service on port {PORT}")
    uvicorn.run(
        "server:app",
        host    = "0.0.0.0",
        port    = PORT,
        workers = 1,          # Single worker — GPU state is not fork-safe
        log_level = LOG_LEVEL.lower(),
        access_log = False,   # Reduce noise; GNN-Service logs its own
    )
