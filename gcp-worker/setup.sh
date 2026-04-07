#!/usr/bin/env bash
# ==============================================================================
# Quantum Alchemy Engine — GCP Worker Setup Script
# Run as root (or with sudo) on the GCP GPU instance
# ==============================================================================
set -euo pipefail

APP_DIR="/opt/quantum-alchemy"
SERVICE_NAME="quantum-alchemy-gcp"
NODE_VERSION="20"

echo "=== Quantum Alchemy Engine: GCP Worker Setup ==="

# ── 1. System packages ────────────────────────────────────────────────────────
apt-get update -y
apt-get install -y curl git build-essential python3-pip python3-venv

# ── GPU: NVIDIA drivers (skip if already installed) ──────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
  apt-get install -y linux-headers-$(uname -r) nvidia-driver-550
  echo ">>> NVIDIA driver installed — REBOOT REQUIRED, then re-run this script"
  exit 0
fi
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

# ── Optional: Quantum ESPRESSO (only needed if ENABLE_DFT_WORKER=true) ───────
# apt-get install -y quantum-espresso

# ── 2. Node.js (via NodeSource) ───────────────────────────────────────────────
if ! command -v node &>/dev/null || [[ "$(node -v)" != v${NODE_VERSION}* ]]; then
  curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash -
  apt-get install -y nodejs
fi

echo "Node: $(node -v)   npm: $(npm -v)"

# ── 3. Clone / update the repo ────────────────────────────────────────────────
if [ -d "$APP_DIR/.git" ]; then
  echo "Updating existing repo..."
  cd "$APP_DIR"
  git pull --ff-only
else
  echo "Cloning repo..."
  git clone https://github.com/YOUR_ORG/quantum-alchemy-engine.git "$APP_DIR"
  cd "$APP_DIR"
fi

# ── 4. Install Node + Python dependencies ────────────────────────────────────
npm install --ignore-scripts

# GPU PyTorch + GNN service deps
if [ ! -d /opt/qae-venv ]; then
  python3 -m venv /opt/qae-venv
fi
/opt/qae-venv/bin/pip install --upgrade pip
/opt/qae-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
/opt/qae-venv/bin/pip install -r gnn/requirements.txt
/opt/qae-venv/bin/pip install torch-geometric 2>/dev/null || true

mkdir -p /opt/qae/gnn_weights

# ── 5. Environment file ───────────────────────────────────────────────────────
ENV_FILE="/etc/quantum-alchemy.env"
if [ ! -f "$ENV_FILE" ]; then
  cat > "$ENV_FILE" <<'EOF'
# Fill in your Neon connection string
DATABASE_URL=postgresql://USER:PASS@HOST/DATABASE?sslmode=require

# Python venv with GPU PyTorch
PYTHON_BIN=/opt/qae-venv/bin/python3

# GNN service
GNN_SERVICE_PORT=8765
GNN_WEIGHTS_DIR=/opt/qae/gnn_weights

# Worker toggles
ENABLE_DFT_WORKER=false
ENABLE_GNN_WORKER=true
ENABLE_XGB_WORKER=true
ENABLE_ML_WORKER=false

OMP_NUM_THREADS=2
NODE_ENV=production
EOF
  echo ""
  echo ">>> ACTION REQUIRED: edit $ENV_FILE and set DATABASE_URL"
  echo ""
fi

# ── 6. Systemd service ────────────────────────────────────────────────────────
cp "$APP_DIR/gcp-worker/quantum-alchemy-gcp.service" \
   "/etc/systemd/system/${SERVICE_NAME}.service"

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

echo ""
echo "=== Setup complete! ==="
echo "Service status:"
systemctl status "$SERVICE_NAME" --no-pager || true
echo ""
echo "Live logs:  journalctl -fu $SERVICE_NAME"
