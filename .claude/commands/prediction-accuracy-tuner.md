# Prediction Accuracy Tuner — One Iteration

You are fixing and improving the per-material Tc prediction pipeline so that each material gets a UNIQUE, accurate prediction from the GNN, XGBoost, and ensemble models.

## Current Problem
1. The GCP PyTorch service (34.130.121.199:8765) is often unreachable from the local server
2. When unreachable, GNN predictions fall back to the local TS ensemble which has no trained weights → returns garbage/zeros/identical values for all materials
3. XGBoost predictions fall back to local JS gradient boosting which may also lack trained weights
4. The result: many materials show identical Tc predictions instead of unique per-material values

## Architecture
- **GCP PyTorch GNN service** (`GNN_PYTORCH_SERVICE_URL`): Runs on GCP VM, has trained ensemble + XGBoost. When reachable, provides best predictions.
- **Local TS GNN ensemble** (`graph-neural-net.ts`): TypeScript implementation. Only useful if GCP-trained weights have been loaded via the weight poller.
- **Local JS XGBoost** (`gradient-boost.ts`): JavaScript gradient boosting. Has its own trained models from the GCP XGB poller.
- **Colab XGB cache** (`colab-xgb-cache.json`): 15k+ cached predictions from previous GCP XGB runs. This is a goldmine for per-material predictions.
- **Physics engine** (`physics-engine.ts`): `computePhysicsTcUQ` — fully local, no GCP needed.

## Key Files
- `server/learning/graph-neural-net.ts` — GNN prediction, PyTorch fetch, local ensemble
  - `gnnPredictWithUncertainty()` (~line 4417) — sync path
  - `gnnPredictWithUncertaintyAsync()` (~line 4680) — async path with GCP fallback
  - `gnnPredictBestPressure()` (~line 4735) — pressure sweep
- `server/learning/gradient-boost.ts` — XGBoost prediction
  - `gbPredictWithUncertaintyAsync()` (~line 1640) — async with GCP fallback
  - `fetchColabXGBPrediction()` (~line 1592) — GCP XGB fetch
  - `gbPredictWithUncertainty()` (~line 1663) — local prediction
- `server/learning/ml-predictor.ts` — Unified consensus (inverse-variance weighted)
  - `computeUnifiedCI()` — combines GNN + XGB + Physics
- `server/routes.ts` — API routes
  - `/api/gnn/predict/:formula` (~line 3010) — GNN prediction route
  - `/api/unified-ci/:formula` (~line 5466) — Consensus route
- `colab-xgb-cache.json` — Cached XGB predictions (15k+ entries)

## Each Iteration
1. **Diagnose**: Check what predictions are currently being returned for specific materials. Verify GCP reachability. Check if local models have trained weights loaded.
2. **Fix**: Make the prediction pipeline gracefully handle GCP being down by:
   - Using the colab-xgb-cache.json as a high-quality fallback for XGB predictions
   - Having GNN return honest "unavailable" (confidence=0) instead of garbage when no good weights exist
   - Ensuring the consensus properly weights available models and excludes unavailable ones
3. **Test**: Check that different materials get different predictions
4. **Benchmark**: Compare predicted vs known Tc for verified compounds to track MAE/R²

## Goal
- Each material should get a UNIQUE prediction based on its specific composition and properties
- MAE target: under 8%, ideally approaching 5%
- When GCP is down, the local pipeline should still produce reasonable per-material predictions using cached XGB + physics engine
