#!/usr/bin/env bash
# ==============================================================================
# DMFT Setup Script for gnn-training VM (c2-standard-30)
#
# Installs Docker (if needed), builds the TRIQS+solid_dmft container,
# and starts the DMFT HTTP service on port 8780.
#
# Prerequisites:
#   - Python 3 + CUDA already installed (existing GNN setup)
#   - Network access to Docker Hub (for triqs/triqs:3.3 base image)
#
# Usage:
#   sudo bash dmft/setup-dmft.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_PORT=8780

echo "=== QAE DMFT Infrastructure Setup ==="
echo "Project dir: $PROJECT_DIR"
echo "DMFT service port: $SERVICE_PORT"

# ── 1. Install Docker if not present ────────────────────────────────────────
if ! command -v docker &>/dev/null; then
  echo ">>> Installing Docker..."
  apt-get update -y
  apt-get install -y ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null
  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
  systemctl enable docker
  systemctl start docker
  echo ">>> Docker installed: $(docker --version)"
else
  echo "Docker already installed: $(docker --version)"
fi

# ── 2. Build the DMFT container ─────────────────────────────────────────────
echo ""
echo ">>> Building qae-dmft Docker image..."
echo "    This pulls triqs/triqs:3.3 (~4GB) and installs solid_dmft."
echo "    First build takes 10-20 minutes."

cd "$PROJECT_DIR"
docker build -t qae-dmft:latest -f dmft/Dockerfile .

echo ">>> Build complete. Image size:"
docker images qae-dmft:latest --format "{{.Size}}"

# ── 3. Verify TRIQS installation inside container ───────────────────────────
echo ""
echo ">>> Verifying TRIQS installation..."
docker run --rm qae-dmft:latest python3 /app/triqs-healthcheck.py
echo ">>> TRIQS verification passed."

# ── 4. Create data directories on host ──────────────────────────────────────
mkdir -p /data/dmft_jobs /data/dmft_results /data/dmft_bundles
echo ">>> Data directories created under /data/"

# ── 5. Load DATABASE_URL from existing env ──────────────────────────────────
ENV_FILE="/etc/quantum-alchemy.env"
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  echo ">>> Loaded environment from $ENV_FILE"
else
  echo "WARNING: $ENV_FILE not found. Set DATABASE_URL before starting."
fi

# Ensure DMFT_SERVICE_URL is set for the DFT workers to find us
if ! grep -q "DMFT_SERVICE_URL" "$ENV_FILE" 2>/dev/null; then
  echo "" >> "$ENV_FILE"
  echo "# DMFT service (Docker container on this VM)" >> "$ENV_FILE"
  echo "DMFT_SERVICE_URL=http://localhost:${SERVICE_PORT}" >> "$ENV_FILE"
  echo ">>> Added DMFT_SERVICE_URL to $ENV_FILE"
fi

# ── 6. Start the service ────────────────────────────────────────────────────
echo ""
echo ">>> Starting DMFT service via docker compose..."
cd "$PROJECT_DIR"

# Export DATABASE_URL for docker-compose interpolation
export DATABASE_URL="${DATABASE_URL:-}"
docker compose -f dmft/docker-compose.yml up -d

echo ""
echo ">>> Waiting for health check..."
sleep 10

HEALTH=$(curl -sf http://localhost:${SERVICE_PORT}/health 2>/dev/null || echo "FAILED")
if echo "$HEALTH" | grep -q '"healthy"'; then
  echo ">>> DMFT service is healthy!"
  echo "$HEALTH" | python3 -m json.tool 2>/dev/null || echo "$HEALTH"
else
  echo "WARNING: Health check did not return healthy. Check logs:"
  echo "  docker compose -f dmft/docker-compose.yml logs dmft"
  echo "Response: $HEALTH"
fi

# ── 7. Firewall rule for port 8780 (DMFT service) ──────────────────────────
echo ""
echo ">>> Checking GCP firewall for port 8780..."
if command -v gcloud &>/dev/null; then
  # Check if rule already exists
  if ! gcloud compute firewall-rules describe allow-dmft-8780 &>/dev/null 2>&1; then
    echo ">>> Creating firewall rule allow-dmft-8780..."
    gcloud compute firewall-rules create allow-dmft-8780 \
      --allow=tcp:8780 \
      --source-ranges=10.128.0.0/20,10.188.0.0/20 \
      --description="Allow DMFT service access from internal VMs" \
      --direction=INGRESS \
      --priority=1000 \
      2>/dev/null || echo "WARNING: Could not create firewall rule. Create manually:"
    echo "  gcloud compute firewall-rules create allow-dmft-8780 --allow=tcp:8780 --source-ranges=10.128.0.0/20"
  else
    echo ">>> Firewall rule allow-dmft-8780 already exists"
  fi
else
  echo ">>> gcloud CLI not found. Ensure port 8780 is open for DFT worker access:"
  echo "  gcloud compute firewall-rules create allow-dmft-8780 --allow=tcp:8780 --source-ranges=10.128.0.0/20"
fi

echo ""
echo "=== DMFT Setup Complete ==="
echo ""
echo "Service URL:  http://localhost:${SERVICE_PORT}"
echo "Health:       curl http://localhost:${SERVICE_PORT}/health"
echo "Capabilities: curl http://localhost:${SERVICE_PORT}/capabilities"
echo "Submit job:   curl -X POST -F bundle=@file.h5 http://localhost:${SERVICE_PORT}/submit"
echo ""
echo "Logs:         docker compose -f dmft/docker-compose.yml logs -f dmft"
echo "Stop:         docker compose -f dmft/docker-compose.yml down"
echo "Restart:      docker compose -f dmft/docker-compose.yml restart"
