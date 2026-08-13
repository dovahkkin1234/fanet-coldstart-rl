"""
model_gnn_attn.py — M4 Step 1. Graph encoder + query-key candidate scorer.

DESIGN (M4_EXECUTION_PLAN §1.2), with three deviations, each justified below.

    Stage 1  ENCODER, once per FRAME, cached
                h = Linear(node_feat -> d)
                repeat L times:  h = LayerNorm(h + Mixer(h, adj, edge_feat))
    Stage 2  QUERY   q = MLP([h_cur, dst_repr, query_feat]) -> d
    Stage 3  KEY     k_u = MLP([h_u, edge_feat(cur,u), dst_repr - h_u, cand_feat_u])
                     logit_u = <q, k_u> / sqrt(d);  invalid -> -inf

Feature widths are read from features_v2 at construction, never hardcoded:
after the v1-v6 passes they are NODE 9, EDGE 4, QUERY 6, CANDIDATE 4, and the
schema boundary exists precisely so a stale constant here cannot go unnoticed.

────────────────────────────────────────────────────────────────────────────
DEVIATION 1 — no separate `encoder_horizon`. L IS the horizon.
────────────────────────────────────────────────────────────────────────────
The plan proposed a k-hop attention mask on top of the depth sweep. That is
redundant and harmful:

  - redundant: an L-layer GNN propagates information exactly L hops, which the
    plan's own text says. The sweep L in {0,1,2,3} IS the encoder-horizon
    sweep.
  - harmful: a mask centred on the CURRENT node makes the encoding depend on
    the decision rather than the frame, which destroys per-frame caching. In
    very_dense (num_flows = N//4 = 11) that is up to an 11x cost increase, and
    the cache is correctness-adjacent, not an optimisation.

────────────────────────────────────────────────────────────────────────────
DEVIATION 2 — `dst_encoding` is a switch, because h_dst leaks global state
────────────────────────────────────────────────────────────────────────────
The plan's Stage 2 uses h_dst, the destination's ENCODED representation. After
L layers that aggregates the destination's L-hop neighbourhood -- queue
occupancies, link qualities and degrees of nodes clustered around the
destination, potentially many hops from the forwarding drone.

A real router knows the destination's POSITION (it rides in the packet header,
GPSR's standing assumption). It does not know the destination's neighbourhood.
So h_dst quietly reintroduces exactly the centralisation that
LOCAL_HORIZON = 2 was chosen to remove, and it would do so invisibly.

  dst_encoding='encoded'  h_dst          the plan as written. DEFAULT, so the
                                         headline number matches the approved
                                         design.
  dst_encoding='raw'      Linear(x_dst)  destination's own node features only,
                                         projected to d. Locally obtainable.

Run both. If they are close, the decentralisation claim is free and should be
made loudly; if they are not, the gap is the honest price of decentralisation
and belongs in the paper as a number rather than a caveat.

────────────────────────────────────────────────────────────────────────────
DEVIATION 3 — the MLP control has MATCHED CAPACITY, not zero capacity
────────────────────────────────────────────────────────────────────────────
The plan describes the MLP baseline as "no graph encoder". Taken literally
that is L=0, which removes message passing AND ~all per-node processing at the
same time -- so a win for the GNN would be attributable to either.

    mixer='attention'  DenseGAT over the adjacency          the GNN
    mixer='mlp'        node-wise MLP, same depth and width  matched control
    L=0                no mixer at all                      trivial control

`mixer='mlp'` differs from the GNN in exactly one thing: whether a node may
read its neighbours. That is the comparison check 2 is supposed to make. L=0
is kept as a floor and reported alongside, not instead.
"""

import math
import os

# MUST be set before CUDA context creation, which is why it is here at module
# import and not inside set_determinism(). cuBLAS reads this once, when the
# context is created; setting it from inside a function called later is a
# no-op, and that is exactly why the first determinism fix did nothing.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
import torch.nn as nn
import torch.nn.functional as F_

try:
    import features_v2 as FEAT
except ImportError:                                    # allow standalone import
    FEAT = None

def set_determinism(warn_only=False):
    """Make training and rollout bit-reproducible under a fixed seed.

    G4 check 6 FAILED without this, and only for the MLP: attention drifted by
    0.00e+00 across all three seeds while mlp drifted by up to 3.81e-02 network
    PDR on individual episodes. The asymmetry is the giveaway.

    MECHANISM. score() uses torch.gather and advanced indexing; both have
    scatter_add backward passes, which on CUDA use atomicAdd and are therefore
    nondeterministic at float precision. That noise is tiny -- but EARLY
    STOPPING amplifies it into a discrete outcome: when two epochs' validation
    scores sit within ~1e-5 of each other, a float-level wobble flips which
    epoch is selected as best, and a DIFFERENT CHECKPOINT is returned. The MLP
    trains ~8 epochs longer on average (36.5 vs 28.3) with a flatter tail, so
    it has far more near-ties to flip. Same noise, more amplification.

    IMPORTANT: this is a REPRODUCIBILITY failure, not a validity one.
    Nondeterminism adds variance, it does not add bias, so paired comparisons
    across seeds remain unbiased -- the checkpoint lottery is part of the
    seed-to-seed variance those CIs already measure. What breaks is the
    ability to rerun and get the same number, which is an artifact property a
    paper should have.

    warn_only=False BY DESIGN. The first version defaulted to True, which is
    cosmetic: ops lacking a deterministic CUDA kernel then print a warning and
    run the nondeterministic one anyway. A --repro run afterwards still failed,
    on BOTH architectures. If an op genuinely has no deterministic
    implementation this now raises, which is the outcome you want -- a loud
    failure naming the op, rather than a silent one three hours later.
    """
    if torch.cuda.is_available() and torch.cuda.is_initialized() \
            and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError(
            'CUDA is already initialised and CUBLAS_WORKSPACE_CONFIG is unset. '
            'Import model_gnn_attn BEFORE touching CUDA, or export '
            'CUBLAS_WORKSPACE_CONFIG=:4096:8 in the shell. Setting it now '
            'would be silently ignored.')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except TypeError:
        torch.use_deterministic_algorithms(True)


NEG_INF = -1e9          # not float('-inf'): softmax over an all-masked row
                        # would produce NaN, and a fully-masked row is possible
                        # for a padded decision slot.


# ─────────────────────────────────────────────────────────────────────────────
# assertions — these live in the model, not in the gate
# ─────────────────────────────────────────────────────────────────────────────
def assert_adjacency_symmetric(adj):
    """extract_frame stores each edge ONCE with i<j.

    If the dense mask is built from edge_index_flat without symmetrising, the
    encoder silently becomes a directed message passer: it still trains, still
    converges, still reports a plausible accuracy. That is this project's
    recurring failure signature -- a plausible number resting on a dead
    mechanism -- so the check belongs where the tensor is used, not in a gate
    that might not be run.
    """
    if not torch.equal(adj, adj.transpose(-1, -2)):
        bad = (adj != adj.transpose(-1, -2)).float().sum().item()
        raise AssertionError(
            f"adjacency is NOT symmetric ({bad:.0f} mismatched entries). "
            f"extract_frame stores edges once with i<j; the dense mask must be "
            f"symmetrised at load time or the encoder sees a triangular graph.")


def assert_no_self_loops(adj):
    """Self-attention is provided by the residual connection. A self-loop in
    the mask would double-count the node's own features and quietly change the
    effective depth."""
    d = torch.diagonal(adj, dim1=-2, dim2=-1)
    if bool(d.any()):
        raise AssertionError("adjacency mask contains self-loops; the residual "
                             "connection already carries the node's own state")


class EncoderCacheStats:
    """Encoder calls must scale with FRAMES, not DECISIONS.

    Without caching the encoder re-runs per decision -- up to ~11x in
    very_dense, where num_flows = N//4 = 11. Assert it rather than hope for it.
    """

    def __init__(self):
        # COUNT FRAMES AND DECISIONS, NOT CALLS.
        # The first version counted invocations, which is the wrong
        # granularity: a batched trainer calls encode() once for 48 frames and
        # score() once for ~1000 decisions, giving a call ratio of 1.00 and a
        # spurious failure. What matters is how much WORK each did.
        self.frames_encoded = 0
        self.decisions_scored = 0

    def assert_cached(self, max_ratio=0.5):
        """encode_calls / score_calls must be WELL BELOW 1.

        The expected value is frames/decisions = 48,000/533,237 ~= 0.09. A
        ratio of 1.0 means one encode per decision, i.e. no cache at all.
        0.5 is a loose bound that catches that without firing on an unusual
        batch composition.

        (The first draft of this compared against 1.05, which passed at
        ratio 1.0 -- it would have declared "cached" precisely when nothing was
        cached. Caught by the verification harness, not by reading it.)
        """
        if self.decisions_scored == 0:
            return
        ratio = self.frames_encoded / max(self.decisions_scored, 1)
        if ratio > max_ratio:
            raise AssertionError(
                f"encoded {self.frames_encoded} frames for "
                f"{self.decisions_scored} decisions (ratio {ratio:.2f}, "
                f"expected ~0.09). The per-frame cache is not being used; "
                f"training will be several times slower and the depth sweep "
                f"will be mis-costed.")


# ─────────────────────────────────────────────────────────────────────────────
# mixers
# ─────────────────────────────────────────────────────────────────────────────
class DenseGATLayer(nn.Module):
    """Multi-head attention over a dense adjacency, with edge features.

    Dense N x N rather than a sparse/scatter implementation because N is 20-45
    here. torch_geometric and DGL are deliberately excluded: at this size the
    gather/scatter overhead exceeds the dense matmul, and a dependency-free
    implementation is one fewer thing that can silently change under us.
    """

    def __init__(self, d, heads, edge_dim, dropout=0.1, attn_dropout=0.0):
        super().__init__()
        assert d % heads == 0, f"d={d} not divisible by heads={heads}"
        self.h, self.dk = heads, d // heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.e = nn.Linear(edge_dim, heads)       # edge bias, one scalar / head
        self.out = nn.Linear(d, d)
        # DROPOUT PARITY (fix 3).
        # The first version applied `dropout` to the ATTENTION WEIGHTS while the
        # MLP control applied the same p to a 256-dim hidden layer. Same nominal
        # rate, wildly different severity: with node degree 2.7-17.9, dropping
        # 10% of attention weights removes a real share of a node's
        # neighbourhood, whereas dropping 10% of 256 hidden units is mild
        # regularisation. That is an M-14 parity break, and a reviewer can
        # argue the GNN was handicapped.
        # Now: both mixers apply exactly ONE feature dropout, on the layer
        # OUTPUT before the residual add. Attention dropout is a SEPARATE,
        # GNN-only knob defaulting to 0.0, swept explicitly rather than
        # inherited silently.
        self.attn_drop = nn.Dropout(attn_dropout)
        self.feat_drop = nn.Dropout(dropout)

    def forward(self, h, adj, edge_dense):
        # h [B,N,d]   adj [B,N,N] bool   edge_dense [B,N,N,Fe]
        B, N, _ = h.shape
        q = self.q(h).view(B, N, self.h, self.dk).transpose(1, 2)   # [B,H,N,dk]
        k = self.k(h).view(B, N, self.h, self.dk).transpose(1, 2)
        v = self.v(h).view(B, N, self.h, self.dk).transpose(1, 2)

        logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)     # [B,H,N,N]
        logits = logits + self.e(edge_dense).permute(0, 3, 1, 2)    # edge bias

        mask = adj.unsqueeze(1)                                     # [B,1,N,N]
        logits = logits.masked_fill(~mask, NEG_INF)
        att = torch.softmax(logits, dim=-1)
        # An isolated node has an all-masked row: softmax over NEG_INF is
        # uniform, not NaN, but the message would be meaningless. Zero it and
        # let the residual carry the node's own state through unchanged.
        att = att * mask.any(dim=-1, keepdim=True).float()
        att = self.attn_drop(att)

        msg = (att @ v).transpose(1, 2).reshape(B, N, -1)
        return self.feat_drop(self.out(msg))


class NodeMLPLayer(nn.Module):
    """Matched-capacity control: identical shape, cannot read neighbours."""

    def __init__(self, d, heads, edge_dim, dropout=0.1, attn_dropout=0.0):
        super().__init__()
        del heads, edge_dim, attn_dropout         # signature parity with GAT
        # One feature dropout on the OUTPUT, matching DenseGATLayer exactly.
        # The internal Dropout of the first version sat between the two linears
        # instead, which is a different placement from the GAT's and is what
        # made the parity argument attackable.
        self.net = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.feat_drop = nn.Dropout(dropout)

    def forward(self, h, adj, edge_dense):
        del adj, edge_dense
        return self.feat_drop(self.net(h))


class DenseGATEdgeKeyLayer(nn.Module):
    """GAT variant where edge features enter the KEYS AND VALUES, not a bias.

    WHY (fix 4). DenseGATLayer uses edge features only as a per-head scalar
    added to the attention logit:
        logits += Linear(edge_dim, heads)(e_ij)
    That is a weak way to consume 4 edge features. If the GNN loses partly
    because MY attention layer under-uses edge information, the negative result
    is about this implementation rather than about message passing -- an
    objection a reviewer will raise and I could not answer.

    Here k_ij and v_ij are computed from [h_j ; e_ij], so link quality,
    distance, lifetime and relative velocity shape both what a neighbour is
    matched on and what it contributes.

    COST: keys and values become [B,N,N,d] instead of [B,N,d]. At N<=45, d=128,
    48 frames per batch that is ~50 MB each -- fine on 16 GB. Parameter count
    rises ~1.5% over DenseGATLayer, so this variant is a GNN-vs-GNN control
    ("was the GNN underpowered?") and NOT the matched-capacity comparison,
    which stays attention-vs-mlp.
    """

    def __init__(self, d, heads, edge_dim, dropout=0.1, attn_dropout=0.0):
        super().__init__()
        assert d % heads == 0, f"d={d} not divisible by heads={heads}"
        self.h, self.dk, self.d = heads, d // heads, d
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d + edge_dim, 2 * d)
        self.out = nn.Linear(d, d)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.feat_drop = nn.Dropout(dropout)

    def forward(self, h, adj, edge_dense):
        B, N, d = h.shape
        H, dk = self.h, self.dk
        hj = h.unsqueeze(1).expand(B, N, N, d)                 # h_j per (i,j)
        kv = self.kv(torch.cat([hj, edge_dense], dim=-1))      # [B,N,N,2d]
        k, v = kv.chunk(2, dim=-1)
        k = k.view(B, N, N, H, dk)
        v = v.view(B, N, N, H, dk)
        q = self.q(h).view(B, N, H, dk)

        logits = torch.einsum('bihd,bijhd->bhij', q, k) / math.sqrt(dk)
        mask = adj.unsqueeze(1)
        logits = logits.masked_fill(~mask, NEG_INF)
        att = torch.softmax(logits, dim=-1)
        att = att * mask.any(dim=-1, keepdim=True).float()
        att = self.attn_drop(att)

        msg = torch.einsum('bhij,bijhd->bihd', att, v).reshape(B, N, d)
        return self.feat_drop(self.out(msg))


MIXERS = {'attention': DenseGATLayer,
          'attention_edgekey': DenseGATEdgeKeyLayer,
          'mlp': NodeMLPLayer}


# ─────────────────────────────────────────────────────────────────────────────
# model
# ─────────────────────────────────────────────────────────────────────────────
class FANETRouter(nn.Module):
    def __init__(self, node_dim=None, edge_dim=None, query_dim=None,
                 cand_dim=None, d=128, layers=2, heads=4, dropout=0.1,
                 mixer='attention', dst_encoding='encoded', attn_dropout=0.0):
        super().__init__()
        if FEAT is not None:
            node_dim = node_dim or len(FEAT.NODE_FEATURES)
            edge_dim = edge_dim or len(FEAT.EDGE_FEATURES)
            query_dim = query_dim or len(FEAT.QUERY_FEATURES)
            cand_dim = cand_dim or len(FEAT.CANDIDATE_FEATURES)
            self.schema_version = getattr(FEAT, 'FEATURE_SCHEMA_VERSION', None)
        else:
            self.schema_version = None
        for nm, val in (('node_dim', node_dim), ('edge_dim', edge_dim),
                        ('query_dim', query_dim), ('cand_dim', cand_dim)):
            if val is None:
                raise ValueError(f"{nm} unknown: pass it, or make features_v2 "
                                 f"importable so widths come from the schema")
        if mixer not in MIXERS:
            raise ValueError(f"mixer must be one of {sorted(MIXERS)}")
        if dst_encoding not in ('encoded', 'raw'):
            raise ValueError("dst_encoding must be 'encoded' or 'raw'")

        self.d, self.L = d, layers
        self.mixer_name, self.dst_encoding = mixer, dst_encoding
        self.dims = dict(node=node_dim, edge=edge_dim, query=query_dim,
                         cand=cand_dim)
        self.stats = EncoderCacheStats()

        self.node_in = nn.Linear(node_dim, d)
        self.mix = nn.ModuleList(
            [MIXERS[mixer](d, heads, edge_dim, dropout, attn_dropout)
             for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        # 'raw' destination projection: position/velocity/queue of the
        # destination only, no neighbourhood aggregation. Always constructed so
        # a checkpoint loads under either setting.
        self.dst_in = nn.Linear(node_dim, d)

        self.query_mlp = nn.Sequential(
            nn.Linear(2 * d + query_dim, 2 * d), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.key_mlp = nn.Sequential(
            nn.Linear(2 * d + edge_dim + cand_dim, 2 * d), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * d, d))

    # ---- Stage 1 -----------------------------------------------------------
    def encode(self, node_feat, adj, edge_dense, node_mask=None, check=True):
        """Run ONCE PER FRAME and cache. [B,N,Fn] -> [B,N,d]."""
        if check:
            assert_adjacency_symmetric(adj)
            assert_no_self_loops(adj)
        if adj.dtype != torch.bool:
            adj = adj.bool()
        h = self.node_in(node_feat)
        for mix, norm in zip(self.mix, self.norm):
            h = norm(h + mix(h, adj, edge_dense))
        if node_mask is not None:
            h = h * node_mask.unsqueeze(-1).to(h.dtype)
        self.stats.frames_encoded += node_feat.shape[0]
        return h

    # ---- Stages 2 and 3 ----------------------------------------------------
    def score(self, h, node_feat, cur_idx, dst_idx, query_feat,
              cand_idx, cand_feat, cand_edge_feat, cand_mask):
        """
        h              [B,N,d]     cached encoder output for each decision's frame
        node_feat      [B,N,Fn]    raw node features (needed by dst_encoding='raw')
        cur_idx        [B]         current node
        dst_idx        [B]         destination node
        query_feat     [B,Fq]
        cand_idx       [B,K]       candidate node ids, padded
        cand_feat      [B,K,Fc]
        cand_edge_feat [B,K,Fe]    edge features of (current, candidate)
        cand_mask      [B,K] bool  True where the slot is a real candidate
        returns logits [B,K], masked slots at NEG_INF
        """
        B, N, d = h.shape
        bi = torch.arange(B, device=h.device)
        h_cur = h[bi, cur_idx]                                       # [B,d]
        if self.dst_encoding == 'encoded':
            dst_repr = h[bi, dst_idx]
        else:
            dst_repr = self.dst_in(node_feat[bi, dst_idx])
        q = self.query_mlp(torch.cat([h_cur, dst_repr, query_feat], dim=-1))

        idx = cand_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, d)
        h_cand = torch.gather(h, 1, idx)                             # [B,K,d]
        dst_rel = dst_repr.unsqueeze(1) - h_cand
        k = self.key_mlp(torch.cat(
            [h_cand, cand_edge_feat, dst_rel, cand_feat], dim=-1))   # [B,K,d]

        logits = (k @ q.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.d)
        logits = logits.masked_fill(~cand_mask.bool(), NEG_INF)
        self.stats.decisions_scored += logits.shape[0]
        return logits

    def forward(self, node_feat, adj, edge_dense, cur_idx, dst_idx, query_feat,
                cand_idx, cand_feat, cand_edge_feat, cand_mask, node_mask=None):
        """Uncached path. Convenient for tests; the trainer should call
        encode() once per frame and score() per decision."""
        h = self.encode(node_feat, adj, edge_dense, node_mask)
        return self.score(h, node_feat, cur_idx, dst_idx, query_feat,
                          cand_idx, cand_feat, cand_edge_feat, cand_mask)

    # ---- housekeeping ------------------------------------------------------
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self):
        return (f"FANETRouter(d={self.d}, L={self.L}, mixer={self.mixer_name}, "
                f"dst={self.dst_encoding}, schema=v{self.schema_version}, "
                f"params={self.n_params():,})")


def build_pair(**kw):
    """The GNN and its matched control, guaranteed identical except message
    passing. Built together so the two can never drift apart through a
    copy-pasted constructor -- M-14 is about comparability, and comparability
    that depends on remembering to edit two call sites is not comparability.
    """
    gnn = FANETRouter(mixer='attention', **kw)
    mlp = FANETRouter(mixer='mlp', **kw)
    return gnn, mlp


def densify(edge_index, edge_feat, n_nodes, symmetric=True, device=None):
    """[2,E] + [E,Fe] -> adjacency [N,N] bool and dense edge features [N,N,Fe].

    extract_frame stores each edge once with i<j, so symmetric=True is the
    correct default and turning it off is almost certainly a bug. The
    assertions in encode() catch it either way.
    """
    device = device or edge_feat.device
    fe = edge_feat.shape[-1]
    adj = torch.zeros(n_nodes, n_nodes, dtype=torch.bool, device=device)
    dense = torch.zeros(n_nodes, n_nodes, fe, dtype=edge_feat.dtype,
                        device=device)
    i, j = edge_index[0].long(), edge_index[1].long()
    adj[i, j] = True
    dense[i, j] = edge_feat
    if symmetric:
        adj[j, i] = True
        dense[j, i] = edge_feat
    return adj, dense
