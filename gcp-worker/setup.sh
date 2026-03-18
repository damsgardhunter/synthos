#!/usr/bin/env bash
# ==============================================================================
# Quantum Alchemy Engine — GCP Worker Setup Script
# Run as root (or with sudo) on the GCP instance (34.42.172.122)
# ==============================================================================
set -euo pipefail

APP_DIR="/opt/quantum-alchemy"
SERVICE_NAME="quantum-alchemy-gcp"
NODE_VERSION="20"

echo "=== Quantum Alchemy Engine: GCP Worker Setup ==="

# ── 1. System packages ────────────────────────────────────────────────────────
apt-get update -y
apt-get install -y curl git build-essential quantum-espresso

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

# ── 4. Install dependencies ───────────────────────────────────────────────────
npm install --ignore-scripts

# ── 5. Environment file ───────────────────────────────────────────────────────
ENV_FILE="/etc/quantum-alchemy.env"
if [ ! -f "$ENV_FILE" ]; then
  cat > "$ENV_FILE" <<'EOF'
# Fill in your Neon connection string
DATABASE_URL=postgresql://USER:PASS@HOST/DATABASE?sslmode=require

# Quantum ESPRESSO binary directory (installed by quantum-espresso package)
QE_BIN_DIR=/usr/bin

# Pseudopotential directory — copy your .UPF files here
PSEUDO_DIR=/opt/quantum-alchemy/server/dft/pseudo

# Main instance (DFT + GNN + XGB on same box):
#   7 concurrent QE runs × 4 threads = 28 vCPUs for DFT, 4 reserved for GNN/XGB
# Dedicated DFT-only instance (ENABLE_GNN_WORKER=false):
#   5 concurrent QE runs × 3 threads = 15 vCPUs (use n2-standard-8 or c2-standard-8)
OMP_NUM_THREADS=4
DFT_MAX_CONCURRENT=7

# Worker toggles (set to "false" to disable either loop)
ENABLE_DFT_WORKER=true
ENABLE_GNN_WORKER=true

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
