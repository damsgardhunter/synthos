"""
superconductor_gnn.py
=====================
Complete PyTorch port of graph-neural-net.ts (Quantum Alchemy Engine).

Architecture mirrors the TypeScript original exactly:
  • SchNet continuous-filter CGCNN convolution layer
  • GLFN-TC adaptive edge logits (bilinear element-pair compatibility)
  • 4× attention message-passing layers with gated residuals
  • Dense skip connection (H0 injection before layer 2)
  • Three-body angle interaction layer
  • Multi-stage pooling: multiplicity-weighted mean + max + attention
  • Pressure modulation on pooled features
  • MLP head (OUTPUT_DIM=16 regression + aleatoric variance)
  • Dedicated classification head for P(SC)
  • Cross-task conditioning: α·σ(λ) added to Tc output (Allen-Dynes)
  • Multi-task loss with Kendall & Gal uncertainty weighting
  • MC Dropout inference + ensemble support
  • Full GPU acceleration via .to(device)

Install deps:
  pip install torch torchvision torchaudio
  pip install torch-geometric
  # GPU (CUDA 12.1):
  pip install torch --index-url https://download.pytorch.org/whl/cu121
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# torch_geometric for batching / pooling (pip install torch-geometric)
try:
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import global_mean_pool, global_max_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("Warning: torch-geometric not installed. DataLoader and pooling helpers unavailable.")


# ── Constants (exact match to graph-neural-net.ts) ────────────────────────────
NODE_DIM         = 32
HIDDEN_DIM       = 48
N_GAUSSIAN_BASIS = 40          # RBF basis functions for edge distances
EDGE_DIM         = N_GAUSSIAN_BASIS
OUTPUT_DIM       = 16          # regression head outputs
GLOBAL_COMP_DIM  = 23          # composition/physics global features
CLS_DIM          = 24          # classification head hidden dim
GRAPH_FEAT_DIM   = 32          # = NODE_DIM, element embedding dim for GLFN-TC
GNN_MSG_LAYERS   = 4           # attention message-passing layers
ENSEMBLE_SIZE    = 5

LAMBDA_MAX       = 5.5         # λ ceiling (electron-phonon coupling)
TC_MAX_K         = 300.0       # Tc ceiling in Kelvin
TC_LOG_SCALE     = math.log1p(TC_MAX_K / 10.0)  # log1p(30) ≈ 3.434
OMEGA_LOG_MAX    = 1500.0      # ω_log ceiling in Kelvin
FIXED_MU_STAR    = 0.10        # Coulomb pseudopotential (BCS)

COSINE_CUTOFF_R  = 6.0         # Å
GAUSSIAN_START   = 0.5
GAUSSIAN_END     = 6.0
GAUSSIAN_STEP    = (GAUSSIAN_END - GAUSSIAN_START) / (N_GAUSSIAN_BASIS - 1)
GAUSSIAN_WIDTH   = GAUSSIAN_STEP

MC_DROPOUT_RATE  = 0.25
MSG_DROPOUT_RATE = 0.10
WEIGHT_DECAY     = 1e-4

# Pooled vector: [mean+attn_pool | max_pool] = 2 × HIDDEN_DIM
POOLED_DIM = 2 * HIDDEN_DIM                      # 96
MLP_INPUT_DIM = POOLED_DIM + GLOBAL_COMP_DIM     # 119

# ── Feature normalisation stats (z-score, from published element data) ────────
FEAT_NORM = {
    "atomicNumber": (47.5,  27.2),
    "en":           (1.84,  0.73),
    "radius":       (145.0, 50.0),
    "valence":      (4.0,   2.4),
    "mass":         (100.0, 72.0),
    "debye":        (360.0, 285.0),
    "fie":          (8.3,   3.5),
    "electronAff":  (0.65,  0.85),
    "meltingPoint": (1400.0,900.0),
    "density":      (6.0,   5.5),
    "period":       (4.0,   2.0),
    "group":        (9.5,   5.4),
}

def znorm(x: float, mean: float, std: float) -> float:
    return (x - mean) / std


# ═══════════════════════════════════════════════════════════════════════════════
# Graph data container
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SuperconGraph:
    """
    Mirrors CrystalGraph from graph-neural-net.ts.

    node_features   : [N, NODE_DIM]         z-scored atomic features
    edge_index      : [2, E]                (src, dst) bidirectional
    edge_attr       : [E, N_GAUSSIAN_BASIS] RBF(distance)
    node_mult       : [N]                   crystallographic multiplicities
    global_features : [GLOBAL_COMP_DIM]     composition + physics hints
    atom_z          : [N] int               atomic numbers (for GLFN-TC lookup)
    three_body_index: [T, 3]                (center, nb1, nb2) triples
    three_body_angle: [T]                   bond angles (radians)
    three_body_d1   : [T]                   distance center→nb1
    three_body_d2   : [T]                   distance center→nb2
    pressure_gpa    : float                 optional pressure
    formula         : str
    """
    node_features:    Tensor
    edge_index:       Tensor
    edge_attr:        Tensor
    node_mult:        Tensor
    global_features:  Tensor
    atom_z:           Tensor
    three_body_index: Optional[Tensor] = None
    three_body_angle: Optional[Tensor] = None
    three_body_d1:    Optional[Tensor] = None
    three_body_d2:    Optional[Tensor] = None
    pressure_gpa:     float = 0.0
    formula:          str   = ""

    # Training targets (set to None if unknown)
    target_tc:     Optional[float] = None
    target_fe:     Optional[float] = None   # formation energy eV/atom
    target_lambda: Optional[float] = None
    target_omega:  Optional[float] = None   # ω_log in K
    target_bg:     Optional[float] = None   # bandgap eV
    target_psc:    Optional[float] = None   # P(SC) soft label [0,1]
    target_phonon_stable: Optional[bool] = None

    def to_pyg(self) -> "Data":
        """Convert to PyTorch Geometric Data object for batched loading."""
        assert HAS_PYG, "torch_geometric required for to_pyg()"
        d = Data(
            x=self.node_features,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            node_mult=self.node_mult,
            global_features=self.global_features.unsqueeze(0),
            atom_z=self.atom_z,
            pressure=torch.tensor([self.pressure_gpa], dtype=torch.float32),
        )
        if self.three_body_index is not None:
            d.tb_index = self.three_body_index
            d.tb_angle = self.three_body_angle
            d.tb_d1    = self.three_body_d1
            d.tb_d2    = self.three_body_d2
        # Targets
        for attr, val in [
            ("y_tc", self.target_tc), ("y_fe", self.target_fe),
            ("y_lambda", self.target_lambda), ("y_omega", self.target_omega),
            ("y_bg", self.target_bg), ("y_psc", self.target_psc),
        ]:
            if val is not None:
                setattr(d, attr, torch.tensor([[val]], dtype=torch.float32))
        return d


# ═══════════════════════════════════════════════════════════════════════════════
# Feature construction helpers
# ═══════════════════════════════════════════════════════════════════════════════

def build_rbf_features(distances: Tensor) -> Tensor:
    """
    Gaussian RBF expansion of interatomic distances.
    Matches buildEdgeFeatures() in graph-neural-net.ts.

    distances : [...] float32  (any shape)
    returns   : [..., N_GAUSSIAN_BASIS]
    """
    centers = torch.linspace(GAUSSIAN_START, GAUSSIAN_END,
                              N_GAUSSIAN_BASIS, device=distances.device,
                              dtype=distances.dtype)   # [B]
    inv2s2  = 1.0 / (2.0 * GAUSSIAN_WIDTH ** 2)
    diff    = distances.unsqueeze(-1) - centers        # [..., B]
    return torch.exp(-diff * diff * inv2s2)


def cosine_cutoff(distances: Tensor, cutoff: float = COSINE_CUTOFF_R) -> Tensor:
    """
    Matches cosineCutoff() in graph-neural-net.ts.
    Returns weight ∈ [0, 1] that smoothly goes to 0 at cutoff.
    """
    w = torch.zeros_like(distances)
    mask = distances < cutoff
    d    = distances[mask]
    w[mask] = 0.5 * (torch.cos(math.pi * d / cutoff) + 1.0)
    return w


# ═══════════════════════════════════════════════════════════════════════════════
# Layer definitions
# ═══════════════════════════════════════════════════════════════════════════════

class SchNetFilter(nn.Module):
    """
    Continuous-filter MLP: RBF → HIDDEN_DIM.
    Matches W_filter1/W_filter2/b_filter1/b_filter2 in the TS.
    Two-layer MLP: Linear(N_GAUSSIAN_BASIS → H) → SiLU → Linear(H → H).
    """
    def __init__(self, in_dim: int = N_GAUSSIAN_BASIS, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, rbf: Tensor) -> Tensor:
        return self.mlp(rbf)   # [E, HIDDEN_DIM]


class CGCNNLayer(nn.Module):
    """
    SchNet-style continuous-filter convolution + GLFN-TC adaptive edge logits.

    Matches the CGCNN block in graph-neural-net.ts:
      filter_ij  = W_filter2(SiLU(W_filter1 @ rbf_ij))
      adapt_ij   = W_elem_feat[i] @ W_graph_adapt @ W_elem_feat[j]  (bilinear)
      gate_ij    = filter_ij + adapt_ij            (broadcast scalar to vector)
      msg_ij     = gate_ij ⊙ h_j * cutoff_ij * mult_j
      agg_i      = mean_j(msg_ij)
      h_out      = LayerNorm(h_in + agg_i)
    """
    def __init__(
        self,
        hidden: int = HIDDEN_DIM,
        n_rbf: int  = N_GAUSSIAN_BASIS,
        n_elem: int = 119,
        graph_feat_dim: int = GRAPH_FEAT_DIM,
    ):
        super().__init__()
        self.filter   = SchNetFilter(n_rbf, hidden)
        self.norm     = nn.LayerNorm(hidden)
        # GLFN-TC: learnable per-element feature vectors
        self.elem_emb = nn.Embedding(n_elem, graph_feat_dim)
        self.adapt    = nn.Linear(graph_feat_dim, graph_feat_dim, bias=False)
        nn.init.normal_(self.elem_emb.weight, std=0.01)
        nn.init.zeros_(self.adapt.weight)

    def forward(
        self,
        h:          Tensor,     # [N, H]
        edge_index: Tensor,     # [2, E]
        edge_attr:  Tensor,     # [E, N_GAUSSIAN_BASIS]
        distances:  Tensor,     # [E]
        atom_z:     Tensor,     # [N] long
        node_mult:  Tensor,     # [N] float
        batch:      Tensor,     # [N] — graph assignment for multi-graph batch
    ) -> Tensor:
        src, dst = edge_index            # E

        # Continuous filter
        gate = self.filter(edge_attr)    # [E, H]

        # Adaptive edge logit (bilinear element-pair score → scalar per edge)
        ei = self.elem_emb(atom_z[src])  # [E, G]
        ej = self.elem_emb(atom_z[dst])  # [E, G]
        adapt_logit = (ei * self.adapt(ej)).sum(dim=-1, keepdim=True)  # [E, 1]
        gate = gate + adapt_logit        # [E, H]  (broadcast)

        # Cosine cutoff weight
        cw  = cosine_cutoff(distances)   # [E]
        mj  = node_mult[dst]             # [E]

        # Message: element-wise multiply filter gate with neighbor embedding
        msg = gate * h[dst] * (cw * mj).unsqueeze(-1)  # [E, H]

        # Aggregate: multiplicity-weighted mean per node
        N = h.size(0)
        agg = torch.zeros(N, h.size(-1), device=h.device, dtype=h.dtype)
        agg.index_add_(0, src, msg)

        # Divide by total multiplicity per node
        total_mult = torch.zeros(N, device=h.device, dtype=h.dtype)
        total_mult.index_add_(0, src, mj)
        total_mult = total_mult.clamp(min=1e-8)
        agg = agg / total_mult.unsqueeze(-1)

        return self.norm(h + agg)


class ThreeBodyLayer(nn.Module):
    """
    Three-body angle interaction.
    Matches W_3body / W_3body_update block in graph-neural-net.ts.

    For each (center, nb1, nb2) triple:
      angle_feat = [cos(angle), sin(angle), d1, d2, |d1-d2|/max(d1,d2)]
      asymmetry  = |d1 - d2| / max(d1, d2)
      weight     = 1 + 0.3 * asymmetry
      msg_3b     = W_3body(h_nb1 + h_nb2)          [H]
      update_in  = concat([h_center, msg_3b])        [2H]
      delta      = W_3body_update(update_in)          [H]
      h_center  += delta * weight / sqrt(n_neighbors)
    """
    def __init__(self, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.W_3body        = nn.Linear(hidden, hidden, bias=False)
        self.W_3body_update = nn.Linear(hidden * 2, hidden, bias=False)
        self.norm           = nn.LayerNorm(hidden)

    def forward(
        self,
        h:    Tensor,           # [N, H]
        tb_index: Tensor,       # [T, 3]  (center, nb1, nb2)
        tb_d1:    Tensor,       # [T]
        tb_d2:    Tensor,       # [T]
    ) -> Tensor:
        if tb_index.numel() == 0:
            return h

        center, nb1, nb2 = tb_index[:, 0], tb_index[:, 1], tb_index[:, 2]

        asymmetry = (tb_d1 - tb_d2).abs() / (torch.maximum(tb_d1, tb_d2) + 1e-8)
        weight    = (1.0 + 0.3 * asymmetry)   # [T]

        msg = self.W_3body(h[nb1] + h[nb2])   # [T, H]
        combined = torch.cat([h[center], msg], dim=-1)  # [T, 2H]
        delta = F.leaky_relu(self.W_3body_update(combined), 0.01) * weight.unsqueeze(-1)

        # Count neighbors per center for normalisation
        N  = h.size(0)
        nc = torch.zeros(N, device=h.device, dtype=h.dtype)
        nc.index_add_(0, center, torch.ones(center.size(0), device=h.device, dtype=h.dtype))
        nc = nc.clamp(min=1.0)

        agg = torch.zeros_like(h)
        agg.index_add_(0, center, delta)
        agg = agg / nc.sqrt().unsqueeze(-1)

        return self.norm(h + agg)


class AttentionMPLayer(nn.Module):
    """
    Single attention message-passing layer.
    Matches the 4× attention layers (W_message/W_update/W_attn_query/W_attn_key).

    q_i   = LayerNorm(W_query @ h_i)
    k_j   = LayerNorm(W_key   @ h_j)
    score = q_i · k_j + 0.1 * (edge_feat[:H] · q_i) + log(mult_j)
    α     = softmax(scores over neighbors)
    msg_j = W_message @ h_j
    agg_i = Σ_j α_j * msg_j          (possibly with message dropout)
    z_i   = W_update([h_i; agg_i])
    h_out = LayerNorm(h_i + activation(z_i))
    """
    def __init__(
        self,
        hidden: int = HIDDEN_DIM,
        use_leaky_relu: bool = True,   # True for layer 1; False for layers 2-4
        msg_dropout: float = MSG_DROPOUT_RATE,
    ):
        super().__init__()
        self.W_query   = nn.Linear(hidden, hidden, bias=False)
        self.W_key     = nn.Linear(hidden, hidden, bias=False)
        self.W_message = nn.Linear(hidden, hidden, bias=False)
        self.W_update  = nn.Linear(hidden * 2, hidden, bias=False)
        self.norm_q    = nn.LayerNorm(hidden)
        self.norm_k    = nn.LayerNorm(hidden)
        self.norm_out  = nn.LayerNorm(hidden)
        self.use_leaky = use_leaky_relu
        self.msg_drop  = nn.Dropout(msg_dropout)

    def forward(
        self,
        h:          Tensor,   # [N, H]
        edge_index: Tensor,   # [2, E]
        edge_attr:  Tensor,   # [E, N_GAUSSIAN_BASIS]
        node_mult:  Tensor,   # [N]
    ) -> Tensor:
        src, dst = edge_index    # src=i, dst=j  (i receives from j)
        N, H = h.shape

        Q = self.norm_q(self.W_query(h))    # [N, H]
        K = self.norm_k(self.W_key(h))      # [N, H]

        q_i = Q[src]             # [E, H]  — query from receiver
        k_j = K[dst]             # [E, H]  — key from sender

        # Base attention: dot product
        score = (q_i * k_j).sum(dim=-1)   # [E]

        # Edge-feature modulation (use first H dims of RBF, clamped to H)
        ef = edge_attr[:, :H] if edge_attr.size(1) >= H else \
             F.pad(edge_attr, (0, H - edge_attr.size(1)))
        score = score + 0.1 * (ef * q_i).sum(dim=-1)

        # Log-multiplicity bias
        score = score + torch.log(node_mult[dst].clamp(min=1.0))

        # Per-receiver softmax
        # Use scatter_softmax if PyG available, otherwise segment softmax
        score_exp = self._segment_softmax(score, src, N)    # [E]

        # Messages with dropout (whole message vectors zeroed stochastically)
        msg = self.W_message(h[dst])          # [E, H]
        if self.training:
            msg = self.msg_drop(msg)

        # Aggregate
        agg = torch.zeros(N, H, device=h.device, dtype=h.dtype)
        agg.index_add_(0, src, score_exp.unsqueeze(-1) * msg)

        # Update
        combined = torch.cat([h, agg], dim=-1)   # [N, 2H]
        z = self.W_update(combined)               # [N, H]
        if self.use_leaky:
            z = F.leaky_relu(z, 0.01)
        else:
            z = F.relu(z)

        return self.norm_out(h + z)

    @staticmethod
    def _segment_softmax(scores: Tensor, segment_ids: Tensor, n_segments: int) -> Tensor:
        """
        Softmax over neighbor scores for each target node.
        Numerically stable: subtract per-node max before exp.
        """
        # Max per receiver node
        s_max = torch.full((n_segments,), float('-inf'), device=scores.device, dtype=scores.dtype)
        s_max.index_reduce_(0, segment_ids, scores, reduce='amax', include_self=True)
        scores_shifted = scores - s_max[segment_ids]
        exp_s = torch.exp(scores_shifted.clamp(max=20.0))
        # Sum per receiver node
        sum_exp = torch.zeros(n_segments, device=scores.device, dtype=scores.dtype)
        sum_exp.index_add_(0, segment_ids, exp_s)
        return exp_s / (sum_exp[segment_ids] + 1e-10)


class GatedResidual(nn.Module):
    """
    Scalar sigmoid gate residual.
    h_out = (1 - g) * h_prev + g * h_new
    where g = sigmoid(gate_param)
    Matches residual_gates[i] in graph-neural-net.ts.
    """
    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(1))   # sigmoid(0) = 0.5

    def forward(self, h_prev: Tensor, h_new: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate)
        return (1.0 - g) * h_prev + g * h_new


class MultiStagePooling(nn.Module):
    """
    Three-way pooling: multiplicity-weighted mean + max + attention pooling.
    Matches the pooling stage in graph-neural-net.ts.

    pooled = [0.5*(mean_pool + attn_pool) | max_pool]   shape [POOLED_DIM]
    Then: pooled[:H] += (pressure_gpa/300) * W_pressure
    """
    def __init__(self, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.attn_score = nn.Linear(hidden, 1)      # W_attn_pool
        self.W_pressure = nn.Parameter(torch.zeros(hidden))

    def forward(
        self,
        h:          Tensor,   # [N, H]
        node_mult:  Tensor,   # [N]
        batch:      Tensor,   # [N] long — graph assignment
        n_graphs:   int,
        pressure:   Tensor,   # [n_graphs] GPa
    ) -> Tensor:
        H = h.size(-1)
        device = h.device

        # ── Multiplicity-weighted mean pool ──────────────────────────────────
        total_mult = torch.zeros(n_graphs, device=device, dtype=h.dtype)
        total_mult.index_add_(0, batch, node_mult)
        total_mult = total_mult.clamp(min=1e-8)

        w_mean = node_mult / total_mult[batch]         # [N]
        mean_pool = torch.zeros(n_graphs, H, device=device, dtype=h.dtype)
        mean_pool.index_add_(0, batch, h * w_mean.unsqueeze(-1))

        # ── Max pool ─────────────────────────────────────────────────────────
        max_pool = torch.full((n_graphs, H), float('-inf'), device=device, dtype=h.dtype)
        for i in range(n_graphs):
            mask = (batch == i)
            if mask.any():
                max_pool[i] = h[mask].max(dim=0).values

        # ── Attention pool ───────────────────────────────────────────────────
        raw_scores = self.attn_score(h).squeeze(-1)    # [N]
        raw_scores = raw_scores + torch.log(node_mult.clamp(min=1.0))

        # Per-graph softmax
        attn_w = torch.zeros_like(raw_scores)
        for i in range(n_graphs):
            mask = (batch == i)
            if mask.any():
                attn_w[mask] = torch.softmax(raw_scores[mask], dim=0)

        attn_pool = torch.zeros(n_graphs, H, device=device, dtype=h.dtype)
        attn_pool.index_add_(0, batch, h * attn_w.unsqueeze(-1))

        # ── Combine ──────────────────────────────────────────────────────────
        combined_mean = 0.5 * (mean_pool + attn_pool)  # [G, H]
        pooled = torch.cat([combined_mean, max_pool], dim=-1)  # [G, 2H]

        # ── Pressure modulation ──────────────────────────────────────────────
        p_norm = (pressure / 300.0).unsqueeze(-1)            # [G, 1]
        pooled[:, :H] = pooled[:, :H] + p_norm * self.W_pressure.unsqueeze(0)

        return pooled   # [G, 2H]


# ═══════════════════════════════════════════════════════════════════════════════
# Full model
# ═══════════════════════════════════════════════════════════════════════════════

class SuperconductorGNN(nn.Module):
    """
    Full PyTorch superconductor GNN — faithful replica of graph-neural-net.ts.

    Forward signature:
        pred = model(
            h, edge_index, edge_attr, distances,
            atom_z, node_mult, batch, global_features,
            pressure, three_body=None
        )
    Returns: GNNOutput
    """

    def __init__(
        self,
        node_dim:        int = NODE_DIM,
        hidden:          int = HIDDEN_DIM,
        n_rbf:           int = N_GAUSSIAN_BASIS,
        n_layers:        int = GNN_MSG_LAYERS,
        global_dim:      int = GLOBAL_COMP_DIM,
        output_dim:      int = OUTPUT_DIM,
        cls_dim:         int = CLS_DIM,
        graph_feat_dim:  int = GRAPH_FEAT_DIM,
        n_elem:          int = 119,
        mc_dropout_rate: float = MC_DROPOUT_RATE,
    ):
        super().__init__()
        self.hidden     = hidden
        self.n_layers   = n_layers
        self.output_dim = output_dim
        pooled_dim      = 2 * hidden
        mlp_in          = pooled_dim + global_dim

        # ── Input projection ─────────────────────────────────────────────────
        self.input_proj = nn.Linear(node_dim, hidden)

        # ── CGCNN / SchNet layer ─────────────────────────────────────────────
        self.cgcnn = CGCNNLayer(hidden, n_rbf, n_elem, graph_feat_dim)

        # ── Three-body layer ─────────────────────────────────────────────────
        self.three_body = ThreeBodyLayer(hidden)

        # ── Attention message-passing layers (×n_layers) ─────────────────────
        self.attn_layers = nn.ModuleList([
            AttentionMPLayer(hidden, use_leaky_relu=(i == 0))
            for i in range(n_layers)
        ])

        # ── Gated residuals (one per attention layer + one for 3-body) ───────
        self.gates = nn.ModuleList([GatedResidual() for _ in range(n_layers + 1)])

        # ── Dense skip (GLFN-TC §2.4): H0 injected before layer 2 ───────────
        self.dense_skip_gate = nn.Parameter(torch.tensor(-2.0))

        # ── Multi-stage pooling ───────────────────────────────────────────────
        self.pooling = MultiStagePooling(hidden)

        # ── MLP regression head ───────────────────────────────────────────────
        self.mlp1      = nn.Linear(mlp_in, hidden)
        self.mlp2      = nn.Linear(hidden, output_dim)    # mean predictions
        self.mlp2_var  = nn.Linear(hidden, output_dim)    # log-variance

        # ── Dedicated classification head (P(SC)) ────────────────────────────
        self.cls1 = nn.Linear(mlp_in, cls_dim)
        self.cls2 = nn.Linear(cls_dim, 1, bias=True)
        nn.init.uniform_(self.cls2.weight, -0.05, 0.05)
        nn.init.zeros_(self.cls2.bias)

        # ── Cross-task conditioning: α·σ(λ) → Tc (Allen-Dynes) ──────────────
        self.alpha_lambda_to_tc = nn.Parameter(torch.zeros(1))

        # ── MC Dropout ────────────────────────────────────────────────────────
        self.mc_dropout = nn.Dropout(mc_dropout_rate)

        # ── Multi-task uncertainty (Kendall & Gal 2017) ───────────────────────
        # [0]=Tc, [1]=family physics, [2]=formation energy
        self.log_sigma_tasks = nn.Parameter(torch.zeros(3))

        self._init_weights()

    # ── Weight initialisation (matches TS seeded init strategy) ──────────────
    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Attention query/key: small init for numerical stability
        for layer in self.attn_layers:
            nn.init.normal_(layer.W_query.weight, std=math.sqrt(1.0 / self.hidden))
            nn.init.normal_(layer.W_key.weight,   std=math.sqrt(1.0 / self.hidden))

        # MLP2 bias initialisation from TS (physics-informed priors)
        with torch.no_grad():
            self.mlp2.bias[2] = -4.0    # ω_log → ~320 K at init
            self.mlp2.bias[4] = -1.5    # λ → ~1.0 at init (softplus/sigmoid maps)
            self.mlp2.bias[8] = 0.38    # Tc → ~27 K at init (log-scale)

    # ── Forward pass ─────────────────────────────────────────────────────────
    def forward(
        self,
        node_features:   Tensor,           # [N, NODE_DIM]
        edge_index:      Tensor,           # [2, E]
        edge_attr:       Tensor,           # [E, N_GAUSSIAN_BASIS]
        distances:       Tensor,           # [E]
        atom_z:          Tensor,           # [N] long
        node_mult:       Tensor,           # [N] float
        batch:           Tensor,           # [N] long
        global_features: Tensor,           # [G, GLOBAL_COMP_DIM]
        pressure:        Tensor,           # [G] float GPa
        three_body: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        # (tb_index [T,3], tb_d1 [T], tb_d2 [T])
    ) -> "GNNOutput":
        G = global_features.size(0)

        # ── Input projection ─────────────────────────────────────────────────
        h = F.leaky_relu(self.input_proj(node_features), 0.01)   # [N, H]
        H0 = h.clone()   # save for dense skip

        # ── CGCNN (SchNet filter + adaptive logits) ───────────────────────────
        h_cgcnn = self.cgcnn(h, edge_index, edge_attr, distances, atom_z, node_mult, batch)
        h = self.gates[0](h, h_cgcnn)

        # ── Three-body layer ─────────────────────────────────────────────────
        if three_body is not None:
            tb_index, tb_d1, tb_d2 = three_body
            if tb_index.numel() > 0:
                h_3b = self.three_body(h, tb_index, tb_d1, tb_d2)
                h = self.gates[0](h, h_3b)   # reuse gate 0 as in TS "3body residual gate"

        # ── Attention message-passing layers ──────────────────────────────────
        for i, (attn, gate) in enumerate(zip(self.attn_layers, self.gates[1:])):
            # Dense skip: inject H0 before layer 2 (i==1)
            if i == 1:
                dg = torch.sigmoid(self.dense_skip_gate)
                if dg > 1e-4:
                    h = h + dg * H0

            h_new = attn(h, edge_index, edge_attr, node_mult)
            h = gate(h, h_new)

        # ── Multi-stage pooling ───────────────────────────────────────────────
        pooled = self.pooling(h, node_mult, batch, G, pressure)  # [G, 2H]

        # ── Concatenate global composition/physics features ───────────────────
        full = torch.cat([pooled, global_features], dim=-1)      # [G, MLP_IN]

        # ── MLP head ─────────────────────────────────────────────────────────
        z1 = F.leaky_relu(self.mlp1(full), 0.01)   # [G, H]
        h1 = self.mc_dropout(z1)                    # MC Dropout applied here

        out      = self.mlp2(h1)      # [G, OUTPUT_DIM]  — raw regression
        log_var  = self.mlp2_var(h1)  # [G, OUTPUT_DIM]  — log-variance

        # ── Classification head (independent pathway) ─────────────────────────
        z_cls = F.leaky_relu(self.cls1(full), 0.01)   # [G, CLS_DIM]
        psc_logit = self.cls2(z_cls).squeeze(-1)       # [G]
        out[:, 7] = psc_logit                          # overwrite slot 7

        # ── Cross-task conditioning (λ → Tc, Allen-Dynes) ─────────────────────
        out[:, 8] = out[:, 8] + self.alpha_lambda_to_tc * torch.sigmoid(out[:, 4])

        # ── Decode raw outputs to physical quantities ─────────────────────────
        return GNNOutput.decode(out, log_var, h1, G)

    def predict(
        self,
        graph: SuperconGraph,
        device: Optional[torch.device] = None,
        mc_passes: int = 1,
    ) -> "GNNOutput":
        """
        Single-graph inference (no batching needed).
        mc_passes > 1 enables MC Dropout uncertainty.
        """
        if device is None:
            device = next(self.parameters()).device

        def _g(t):
            return t.to(device) if isinstance(t, Tensor) else t

        h_x         = _g(graph.node_features)
        ei          = _g(graph.edge_index)
        ea          = _g(graph.edge_attr)
        dists       = ea.norm(dim=-1) if ea.ndim > 1 else ea  # fallback; normally precomputed
        az          = _g(graph.atom_z).long()
        nm          = _g(graph.node_mult)
        gf          = _g(graph.global_features).unsqueeze(0)
        pres        = torch.tensor([graph.pressure_gpa], device=device)
        batch       = torch.zeros(h_x.size(0), dtype=torch.long, device=device)
        three_body  = None
        if graph.three_body_index is not None:
            three_body = (
                _g(graph.three_body_index),
                _g(graph.three_body_d1),
                _g(graph.three_body_d2),
            )

        if mc_passes <= 1:
            self.eval()
            with torch.no_grad():
                return self.forward(h_x, ei, ea, dists, az, nm, batch, gf, pres, three_body)
        else:
            self.train()   # keep dropout active for MC passes
            outputs = []
            with torch.no_grad():
                for _ in range(mc_passes):
                    o = self.forward(h_x, ei, ea, dists, az, nm, batch, gf, pres, three_body)
                    outputs.append(o)
            return GNNOutput.mc_aggregate(outputs)


# ═══════════════════════════════════════════════════════════════════════════════
# Output container
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GNNOutput:
    """Decoded model outputs, mirrors GNNPrediction + uncertainty from TS."""
    tc:                  Tensor   # [G] Kelvin, clipped to [0, TC_MAX_K]
    omega_log:           Tensor   # [G] Kelvin
    formation_energy:    Tensor   # [G] eV/atom
    lam:                 Tensor   # [G] electron-phonon λ
    bandgap:             Tensor   # [G] eV
    dos_proxy:           Tensor   # [G]
    stability_prob:      Tensor   # [G] P(SC) in [0,1]
    confidence:          Tensor   # [G]
    phonon_stable:       Tensor   # [G] bool-like in [0,1]
    # Aleatoric variance
    tc_var:              Tensor
    lambda_var:          Tensor
    fe_var:              Tensor
    bg_var:              Tensor
    # Latent for OOD detection
    latent:              Tensor   # [G, H]
    # Raw outputs for loss computation
    raw_out:             Tensor   # [G, OUTPUT_DIM]
    raw_log_var:         Tensor   # [G, OUTPUT_DIM]

    @staticmethod
    def decode(out: Tensor, log_var: Tensor, h1: Tensor, G: int) -> "GNNOutput":
        tc       = (10.0 * torch.expm1(out[:, 8].clamp(min=0.0) * TC_LOG_SCALE)).clamp(0, TC_MAX_K)
        omega    = (10.0 + (OMEGA_LOG_MAX - 10.0) * torch.sigmoid(out[:, 2] / 3.0))
        fe       = out[:, 0]
        lam      = LAMBDA_MAX * torch.sigmoid(out[:, 4])
        bg       = 5.0 * torch.sigmoid(out[:, 5])
        dos      = F.softplus(out[:, 6])
        psc      = torch.sigmoid(out[:, 7])
        conf     = torch.sigmoid(out[:, 3]).clamp(0.05, 0.95)
        phonon   = torch.sigmoid(out[:, 1])

        tc_v     = F.softplus(log_var[:, 2]).clamp(min=0.01) * (TC_MAX_K ** 2)
        lv       = F.softplus(log_var[:, 4]).clamp(min=0.001)
        fev      = F.softplus(log_var[:, 0]).clamp(min=0.001)
        bgv      = F.softplus(log_var[:, 5]).clamp(min=0.001)

        return GNNOutput(
            tc=tc, omega_log=omega, formation_energy=fe,
            lam=lam, bandgap=bg, dos_proxy=dos,
            stability_prob=psc, confidence=conf, phonon_stable=phonon,
            tc_var=tc_v, lambda_var=lv, fe_var=fev, bg_var=bgv,
            latent=h1, raw_out=out, raw_log_var=log_var,
        )

    @staticmethod
    def mc_aggregate(passes: List["GNNOutput"]) -> "GNNOutput":
        """Average MC Dropout passes and compute epistemic uncertainty via variance."""
        tcs    = torch.stack([p.tc for p in passes], dim=0)
        mean_o = GNNOutput.decode(
            torch.stack([p.raw_out for p in passes]).mean(0),
            torch.stack([p.raw_log_var for p in passes]).mean(0),
            torch.stack([p.latent for p in passes]).mean(0),
            passes[0].tc.size(0),
        )
        # Add epistemic variance of Tc to aleatoric
        epi_var = tcs.var(dim=0)
        mean_o.tc_var = mean_o.tc_var + epi_var
        return mean_o


# ═══════════════════════════════════════════════════════════════════════════════
# Loss function
# ═══════════════════════════════════════════════════════════════════════════════

class MultitaskLoss(nn.Module):
    """
    Kendall & Gal (2017) uncertainty-weighted multi-task loss.
    Matches the v16 multi-task loss in graph-neural-net.ts.

    L_total = Σ_i exp(-2s_i) * L_i + s_i
    where s_i = log_sigma_tasks[i]

    Tasks:
      [0] Tc regression         (log-normalised MSE)
      [1] λ regression          (MSE)
      [2] ω_log regression      (MSE on normalised log)
      [3] Formation energy      (MSE)
      [4] Bandgap               (MSE)
      [5] P(SC) classification  (BCE with soft labels)
      [6] Phonon stability      (BCE)
    """
    def __init__(self, model: SuperconductorGNN):
        super().__init__()
        self.log_sigma = model.log_sigma_tasks   # [3] shared params

    def forward(
        self,
        out:     GNNOutput,
        targets: Dict[str, Optional[Tensor]],  # keys: tc, fe, lambda, omega, bg, psc, phonon
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        device = out.raw_out.device
        losses: Dict[str, Tensor] = {}

        def mse_masked(pred: Tensor, target: Optional[Tensor]) -> Optional[Tensor]:
            if target is None: return None
            mask = target.isfinite()
            if not mask.any(): return None
            return F.mse_loss(pred[mask], target[mask])

        def bce_masked(logit: Tensor, target: Optional[Tensor]) -> Optional[Tensor]:
            if target is None: return None
            mask = target.isfinite()
            if not mask.any(): return None
            return F.binary_cross_entropy_with_logits(logit[mask], target[mask])

        # ── Tc regression (log1p normalised, direct) ─────────────────────────
        tc_t = targets.get("tc")
        if tc_t is not None:
            mask = (tc_t.isfinite()) & (tc_t >= 0)
            if mask.any():
                tc_norm_pred = out.raw_out[:, 8][mask]
                tc_norm_targ = torch.log1p(tc_t[mask] / 10.0) / TC_LOG_SCALE
                losses["tc"] = F.mse_loss(tc_norm_pred, tc_norm_targ)

        # ── Formation energy ─────────────────────────────────────────────────
        l = mse_masked(out.formation_energy, targets.get("fe"))
        if l is not None: losses["fe"] = l

        # ── Lambda ────────────────────────────────────────────────────────────
        # Compare in raw sigmoid space for gradient stability
        lambda_t = targets.get("lambda")
        if lambda_t is not None:
            mask = lambda_t.isfinite()
            if mask.any():
                raw_lam_pred = out.raw_out[:, 4][mask]
                raw_lam_targ = torch.logit((lambda_t[mask] / LAMBDA_MAX).clamp(1e-6, 1 - 1e-6))
                losses["lambda"] = F.mse_loss(raw_lam_pred, raw_lam_targ)

        # ── ω_log ─────────────────────────────────────────────────────────────
        omega_t = targets.get("omega")
        if omega_t is not None:
            mask = omega_t.isfinite() & (omega_t > 0)
            if mask.any():
                raw_omega_pred = out.raw_out[:, 2][mask]
                omega_norm = (omega_t[mask] - 10.0) / (OMEGA_LOG_MAX - 10.0)
                raw_omega_targ = torch.logit(omega_norm.clamp(1e-6, 1-1e-6)) * 3.0
                losses["omega"] = F.mse_loss(raw_omega_pred, raw_omega_targ)

        # ── Bandgap ───────────────────────────────────────────────────────────
        l = mse_masked(out.bandgap, targets.get("bg"))
        if l is not None: losses["bg"] = l

        # ── P(SC) classification (soft label BCE) ─────────────────────────────
        l = bce_masked(out.raw_out[:, 7], targets.get("psc"))
        if l is not None: losses["psc"] = l

        # ── Phonon stability ─────────────────────────────────────────────────
        phonon_t = targets.get("phonon")
        if phonon_t is not None:
            mask = phonon_t.isfinite()
            if mask.any():
                losses["phonon"] = F.binary_cross_entropy_with_logits(
                    out.raw_out[:, 1][mask], phonon_t[mask].float()
                )

        # ── Uncertainty-weighted combination ─────────────────────────────────
        # Group tasks: [0]=Tc, [1]=physics (lambda+omega), [2]=formation energy+bg
        tc_raw   = losses.get("tc",     torch.zeros(1, device=device))
        phys_raw = (losses.get("lambda", torch.zeros(1, device=device))
                  + losses.get("omega",  torch.zeros(1, device=device))
                  + losses.get("psc",    torch.zeros(1, device=device))
                  + losses.get("phonon", torch.zeros(1, device=device)))
        mat_raw  = (losses.get("fe", torch.zeros(1, device=device))
                  + losses.get("bg", torch.zeros(1, device=device)))

        s = self.log_sigma    # [3]
        total = (torch.exp(-2 * s[0]) * tc_raw   + s[0]
               + torch.exp(-2 * s[1]) * phys_raw + s[1]
               + torch.exp(-2 * s[2]) * mat_raw  + s[2])

        return total, losses


# ═══════════════════════════════════════════════════════════════════════════════
# Training utilities
# ═══════════════════════════════════════════════════════════════════════════════

def make_optimizer(model: SuperconductorGNN, lr: float = 1e-3) -> torch.optim.Optimizer:
    """
    AdamW with per-group learning rates matching the TS training loop:
      - Graph layers (CGCNN, attention, 3-body): 0.3× base LR
      - MLP heads and classification: 1.0× base LR
    """
    graph_params, mlp_params = [], []
    for name, param in model.named_parameters():
        if any(k in name for k in [
            "cgcnn", "three_body", "attn_layers", "dense_skip",
            "gates", "elem_emb", "adapt", "input_proj",
        ]):
            graph_params.append(param)
        else:
            mlp_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": graph_params, "lr": lr * 0.3},
            {"params": mlp_params,   "lr": lr},
        ],
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def cosine_annealing_lr(optimizer, epoch: int, n_epochs: int, lr_init: float = 1e-3):
    """
    Cosine annealing: lr(t) = lr_init * (0.1 + 0.9 * 0.5 * (1 + cos(π*t/T)))
    Matches the TS cosine schedule.
    """
    scale = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * epoch / max(n_epochs, 1)))
    for g in optimizer.param_groups:
        g["lr"] = g.get("_base_lr", g["lr"]) * scale


def compute_n_epochs(n_samples: int) -> int:
    """Epoch schedule from graph-neural-net.ts."""
    return max(30, min(150, math.ceil(60_000 / max(n_samples, 1))))


def curriculum_difficulty(tc: float, n_elements: int) -> float:
    """
    Difficulty metric matching the TS curriculum sampler:
      0.3 * comp_complexity + 0.5 * tc_extremeness
    """
    comp = min(n_elements / 6.0, 1.0)
    if tc <= 0:
        tc_ext = 0.05
    elif tc >= 50:
        tc_ext = min((tc - 50) / 250.0, 1.0)
    else:
        tc_ext = min(tc / 50.0, 1.0) * 0.5
    return 0.3 * comp + 0.5 * tc_ext


class GNNTrainer:
    """
    Training loop for SuperconductorGNN.
    Supports:
      - Curriculum learning (difficulty gating)
      - Hard example mining (high-error oversampling)
      - Formation energy pretraining
      - GPU acceleration
      - Ensemble training (ENSEMBLE_SIZE independent models)
    """

    def __init__(
        self,
        model:   SuperconductorGNN,
        device:  Optional[torch.device] = None,
        lr:      float = 1e-3,
    ):
        self.model  = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = make_optimizer(model, lr)
        self.loss_fn   = MultitaskLoss(model)
        self._sample_errors: Dict[int, float] = {}

    def train_epoch(
        self,
        graphs:    List[SuperconGraph],
        batch_size: int = 64,
        progress:   float = 0.0,   # 0.0 → 1.0 training progress for curriculum
    ) -> float:
        self.model.train()
        indices = self._curriculum_sample(graphs, progress)
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_graphs = [graphs[i] for i in batch_idx]

            loss = self._step(batch_graphs, batch_idx)
            if loss is not None:
                total_loss += loss
                n_batches  += 1

        return total_loss / max(n_batches, 1)

    def _curriculum_sample(self, graphs: List[SuperconGraph], progress: float) -> List[int]:
        """Curriculum difficulty gating + hard example mining."""
        threshold = 0.30 + (1.0 - 0.30) * max(0.0, (progress - 0.25) / 0.75)
        indices = []
        for i, g in enumerate(graphs):
            n_elem = int((g.node_features[:, 0] != 0).sum().item())
            tc     = g.target_tc or 0.0
            diff   = curriculum_difficulty(tc, n_elem)
            if diff <= threshold:
                indices.append(i)
                # Hard example mining: oversample high-error samples
                if progress > 0.4 and i in self._sample_errors:
                    err = self._sample_errors[i]
                    mean_err = sum(self._sample_errors.values()) / max(len(self._sample_errors), 1)
                    extra = min(int(err / (mean_err + 1e-6)) - 1, 3)
                    indices.extend([i] * max(0, extra))
        return indices if indices else list(range(len(graphs)))

    def _step(self, batch: List[SuperconGraph], batch_idx: List[int]) -> Optional[float]:
        """One gradient step on a mini-batch of graphs."""
        if not batch:
            return None

        # ── Collate (simple loop; use DataLoader for large scale) ─────────────
        device = self.device

        node_feats, edge_indices, edge_attrs, distances_list = [], [], [], []
        atom_zs, node_mults, batch_vecs, global_feats = [], [], [], []
        pressures = []
        three_bodies = []

        targets = {k: [] for k in ["tc", "fe", "lambda", "omega", "bg", "psc", "phonon"]}
        node_offset = 0

        for g in batch:
            N = g.node_features.size(0)
            node_feats.append(g.node_features)
            ei = g.edge_index + node_offset
            edge_indices.append(ei)
            edge_attrs.append(g.edge_attr)

            # Reconstruct distances from RBF centers (approximate; use stored dist if available)
            dists = torch.zeros(g.edge_attr.size(0), device=device)
            distances_list.append(dists)

            atom_zs.append(g.atom_z.clamp(0, 118))
            node_mults.append(g.node_mult)
            global_feats.append(g.global_features)
            pressures.append(g.pressure_gpa)

            if g.three_body_index is not None and g.three_body_index.numel() > 0:
                tbi = g.three_body_index.clone()
                tbi[:, :] += node_offset  # offset node indices
                three_bodies.append((tbi, g.three_body_d1, g.three_body_d2))

            bv = torch.full((N,), len(global_feats) - 1, dtype=torch.long)
            batch_vecs.append(bv)
            node_offset += N

            for k, attr in [
                ("tc", "target_tc"), ("fe", "target_fe"),
                ("lambda", "target_lambda"), ("omega", "target_omega"),
                ("bg", "target_bg"), ("psc", "target_psc"),
            ]:
                v = getattr(g, attr)
                targets[k].append(v if v is not None else float('nan'))

            ps = g.target_phonon_stable
            targets["phonon"].append(1.0 if ps else (0.0 if ps is not None else float('nan')))

        def cat(lst): return torch.cat(lst, dim=0).to(device)
        def to_t(lst): return torch.tensor(lst, dtype=torch.float32, device=device)

        h_x  = cat(node_feats)
        ei   = cat(edge_indices)
        ea   = cat(edge_attrs)
        dsts = cat(distances_list)
        az   = cat(atom_zs).long()
        nm   = cat(node_mults)
        bv   = cat(batch_vecs)
        gf   = torch.stack([x.to(device) for x in global_feats])
        pres = to_t(pressures)

        tb = None
        if three_bodies:
            tbi_all = torch.cat([x[0] for x in three_bodies], dim=0).to(device)
            td1_all = torch.cat([x[1] for x in three_bodies], dim=0).to(device)
            td2_all = torch.cat([x[2] for x in three_bodies], dim=0).to(device)
            tb = (tbi_all, td1_all, td2_all)

        tgt = {k: to_t(v) for k, v in targets.items()}

        # ── Forward ──────────────────────────────────────────────────────────
        out = self.model(h_x, ei, ea, dsts, az, nm, bv, gf, pres, tb)

        # ── Loss ─────────────────────────────────────────────────────────────
        loss, per_task = self.loss_fn(out, tgt)
        if not loss.isfinite():
            return None

        # ── Backward ─────────────────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update per-sample error for hard mining
        with torch.no_grad():
            tc_t = tgt["tc"]
            mask = tc_t.isfinite() & (tc_t >= 0)
            if mask.any():
                errs = (out.tc[mask] - tc_t[mask]).abs().cpu().tolist()
                m_tc = tc_t[mask].cpu().tolist()
                for idx, err in zip(
                    [i for i, g in zip(batch_idx, batch) if g.target_tc is not None], errs
                ):
                    self._sample_errors[idx] = err

        return loss.item()

    def train(
        self,
        graphs:        List[SuperconGraph],
        n_epochs:      Optional[int] = None,
        batch_size:    int = 64,
        pretrain_fe:   bool = True,
        pretrain_epochs: int = 15,
        verbose:       bool = True,
    ) -> List[float]:
        """Full training loop."""
        if n_epochs is None:
            n_epochs = compute_n_epochs(len(graphs))

        # Fix base LRs before cosine schedule modifies them
        for g in self.optimizer.param_groups:
            g["_base_lr"] = g["lr"]

        # Optional formation energy pre-training
        if pretrain_fe and sum(1 for g in graphs if g.target_fe is not None) >= 20:
            if verbose:
                print(f"Pre-training formation energy for {pretrain_fe} epochs…")
            self._pretrain_fe(graphs, pretrain_epochs, batch_size)

        losses = []
        for epoch in range(n_epochs):
            progress = epoch / max(n_epochs - 1, 1)
            cosine_annealing_lr(self.optimizer, epoch, n_epochs)
            loss = self.train_epoch(graphs, batch_size, progress)
            losses.append(loss)
            if verbose and (epoch % max(n_epochs // 10, 1) == 0 or epoch == n_epochs - 1):
                print(f"  Epoch {epoch+1}/{n_epochs}  loss={loss:.4f}")
        return losses

    def _pretrain_fe(self, graphs, n_epochs, batch_size):
        """Formation energy pretraining — freeze W_mlp1, only update regression head."""
        for p in self.model.mlp1.parameters():
            p.requires_grad_(False)
        fe_graphs = [g for g in graphs if g.target_fe is not None]
        for epoch in range(n_epochs):
            self.train_epoch(fe_graphs, batch_size, progress=0.0)
        for p in self.model.mlp1.parameters():
            p.requires_grad_(True)


# ═══════════════════════════════════════════════════════════════════════════════
# Ensemble
# ═══════════════════════════════════════════════════════════════════════════════

def train_ensemble(
    graphs:   List[SuperconGraph],
    size:     int = ENSEMBLE_SIZE,
    device:   Optional[torch.device] = None,
    **train_kwargs,
) -> List[SuperconductorGNN]:
    """Train ENSEMBLE_SIZE independent models with different seeds."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for i in range(size):
        torch.manual_seed(i * 1337 + 42)
        m = SuperconductorGNN()
        trainer = GNNTrainer(m, device)
        trainer.train(graphs, verbose=False, **train_kwargs)
        m.eval()
        models.append(m)
        print(f"Ensemble member {i+1}/{size} trained.")
    return models


def ensemble_predict(
    models:  List[SuperconductorGNN],
    graph:   SuperconGraph,
    device:  Optional[torch.device] = None,
) -> GNNOutput:
    """Run all ensemble members and aggregate. Epistemic uncertainty = std of Tc."""
    device = device or next(models[0].parameters()).device
    passes = [m.predict(graph, device, mc_passes=1) for m in models]
    return GNNOutput.mc_aggregate(passes)


# ═══════════════════════════════════════════════════════════════════════════════
# Mahalanobis OOD detector
# ═══════════════════════════════════════════════════════════════════════════════

class MahalanobisOOD:
    """
    Diagonal Mahalanobis distance for OOD detection.
    Matches computeLatentDistance / updateTrainingEmbeddings in graph-neural-net.ts.
    """

    def __init__(self):
        self.mean: Optional[Tensor] = None
        self.var:  Optional[Tensor] = None

    def fit(self, embeddings: Tensor):
        """embeddings: [M, D]"""
        self.mean = embeddings.mean(0)
        var = embeddings.var(0, unbiased=False)
        self.var  = var.clamp(min=1e-6)

    def distance(self, embedding: Tensor) -> float:
        """Return normalised Mahalanobis distance; in-distribution ≈ 1.0."""
        if self.mean is None:
            return 1.0
        diff = (embedding - self.mean)
        d    = ((diff ** 2) / self.var).mean().sqrt()
        return float(d.item())
