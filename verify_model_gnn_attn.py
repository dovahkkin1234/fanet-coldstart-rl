"""
verify_model_gnn_attn.py — execution proof for model_gnn_attn.py.

    python verify_model_gnn_attn.py --src src

Tests 3 and 4 are the load-bearing ones. A message-passing layer that is
silently not passing messages still trains and still reports a plausible
accuracy, so message passing is verified by GRADIENT REACHABILITY -- does a
node two hops away actually influence this node's output? -- rather than by
reading the code. The adjacency and cache assertions are fired deliberately to
confirm they can fail.
"""
import argparse
import os
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument('--src', default='src')
args = ap.parse_args()
sys.path.insert(0, os.path.abspath(args.src))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features_v2 as F                                          # noqa: E402
import model_gnn_attn as M                                       # noqa: E402

OK, NO = "  [PASS]", "  [FAIL]"
fails = []


def check(name, cond, detail=""):
    print(f"{OK if cond else NO} {name:<58} {detail}")
    if not cond:
        fails.append(name)


print("\n" + "=" * 78)
print("  VERIFICATION — model_gnn_attn.py")
print("=" * 78)

torch.manual_seed(0)
B, N, K = 3, 12, 5
Fn, Fe = len(F.NODE_FEATURES), len(F.EDGE_FEATURES)
Fq, Fc = len(F.QUERY_FEATURES), len(F.CANDIDATE_FEATURES)

# ── 1. widths come from the schema, not from constants ─────────────────────
m = M.FANETRouter(d=64, layers=2, heads=4)
check("1a. dims read from features_v2",
      m.dims == dict(node=Fn, edge=Fe, query=Fq, cand=Fc), str(m.dims))
check("1b. schema version recorded on the model",
      m.schema_version == F.FEATURE_SCHEMA_VERSION, f"v{m.schema_version}")
check("1c. default depth is 2, not 3 (M-16)", M.FANETRouter().L == 2)

# ── 2. shapes and masking ──────────────────────────────────────────────────
node_feat = torch.randn(B, N, Fn)
ei = torch.tensor([[i for i in range(N - 1)], [i + 1 for i in range(N - 1)]])
adj, dense = M.densify(ei, torch.randn(N - 1, Fe), N)
adj = adj.unsqueeze(0).expand(B, -1, -1).contiguous()
dense = dense.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
cur = torch.tensor([1, 4, 7])
dst = torch.tensor([9, 0, 11])
qf = torch.randn(B, Fq)
cand_idx = torch.tensor([[0, 2, 0, 0, 0], [3, 5, 0, 0, 0], [6, 8, 0, 0, 0]])
cand_mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]]).bool()
cf, cef = torch.randn(B, K, Fc), torch.randn(B, K, Fe)

logits = m(node_feat, adj, dense, cur, dst, qf, cand_idx, cf, cef, cand_mask)
check("2a. logits shape [B,K]", tuple(logits.shape) == (B, K), str(tuple(logits.shape)))
check("2b. padded slots are masked out",
      bool((logits[~cand_mask] <= M.NEG_INF / 2).all()))
p = torch.softmax(logits, dim=-1)
check("2c. softmax puts ~zero mass on padding",
      float(p[~cand_mask].max()) < 1e-6, f"max={float(p[~cand_mask].max()):.2e}")
check("2d. no NaN anywhere", bool(torch.isfinite(logits[cand_mask]).all()))

# ── 3. message passing is REAL, measured by gradient reachability ──────────
# A 2-layer GNN must let a node 2 hops away influence the output. Chain graph
# 0-1-2-...: with cur=1 and candidates {0,2}, node 3 is 2 hops from candidate 2
# and must therefore reach the logits. Under the MLP control it must NOT.
def hops_reaching(model, hops):
    nf = torch.randn(1, N, Fn, requires_grad=True)
    a = adj[:1].clone()
    de = dense[:1].clone()
    out = model(nf, a, de, torch.tensor([1]), torch.tensor([9]),
                torch.randn(1, Fq), torch.tensor([[0, 2, 0, 0, 0]]),
                torch.randn(1, K, Fc), torch.randn(1, K, Fe),
                torch.tensor([[1, 1, 0, 0, 0]]).bool())
    out[0, :2].sum().backward()
    g = nf.grad[0].abs().sum(-1)
    return [float(g[h]) for h in hops]


gnn = M.FANETRouter(d=64, layers=2, heads=4, dropout=0.0)
mlp = M.FANETRouter(d=64, layers=2, heads=4, dropout=0.0, mixer='mlp')
g_far = hops_reaching(gnn, [4])       # 2 hops beyond candidate node 2
m_far = hops_reaching(mlp, [4])
check("3a. GNN: a 2-hop-distant node reaches the output", g_far[0] > 1e-8,
      f"|grad|={g_far[0]:.3e} — message passing is live")
check("3b. MLP control: it does NOT", m_far[0] < 1e-12,
      f"|grad|={m_far[0]:.3e} — the only difference is neighbour access")

l0 = M.FANETRouter(d=64, layers=0, heads=4, dropout=0.0)
check("3c. L=0 also blocks it", hops_reaching(l0, [4])[0] < 1e-12)

# ── 4. the assertions can actually fire ────────────────────────────────────
bad = adj.clone()
bad[0, 0, 5] = True                       # asymmetric
try:
    m.encode(node_feat, bad, dense)
    caught = False
except AssertionError:
    caught = True
check("4a. asymmetric adjacency is REJECTED", caught,
      "triangular masks cannot slip through silently")

loop = adj.clone()
loop[:, 3, 3] = True
try:
    m.encode(node_feat, loop, dense)
    caught2 = False
except AssertionError:
    caught2 = True
check("4b. self-loops are REJECTED", caught2)

st = M.EncoderCacheStats()
st.frames_encoded, st.decisions_scored = 40, 40
try:
    st.assert_cached()
    fired = False
except AssertionError:
    fired = True
check("4c. cache assertion fires when encode ~= score", fired,
      "40 frames for 40 decisions is ratio 1.0 = no cache")
st.frames_encoded, st.decisions_scored = 40, 440
try:
    st.assert_cached()
    ok_cached = True
except AssertionError:
    ok_cached = False
check("4d. ...and passes when the cache is used", ok_cached,
      "40 frames, 440 decisions")

# ── 5. densify symmetrises, as extract_frame requires ──────────────────────
a2, d2 = M.densify(ei, torch.arange((N - 1) * Fe, dtype=torch.float32
                                    ).view(N - 1, Fe), N)
check("5a. densify produces a symmetric mask", bool(torch.equal(a2, a2.T)))
check("5b. ...and mirrors the edge features",
      bool(torch.equal(d2, d2.transpose(0, 1))))
a3, _ = M.densify(ei, torch.randn(N - 1, Fe), N, symmetric=False)
check("5c. symmetric=False is caught downstream",
      not bool(torch.equal(a3, a3.T)), "encode() would reject this")

# ── 6. the matched control really is matched ───────────────────────────────
g2, m2 = M.build_pair(d=128, layers=2, heads=4)
gp, mp = g2.n_params(), m2.n_params()
check("6a. GNN and MLP control are within 15% on parameters",
      abs(gp - mp) / max(gp, mp) < 0.15,
      f"GNN {gp:,} vs MLP {mp:,} ({100*abs(gp-mp)/max(gp,mp):.1f}% apart)")
check("6b. L=0 is a much smaller trivial control",
      M.FANETRouter(d=128, layers=0).n_params() < 0.75 * gp,
      f"{M.FANETRouter(d=128, layers=0).n_params():,} params")

# ── 7. dst_encoding switch changes only what it should ─────────────────────
enc = M.FANETRouter(d=64, layers=2, dropout=0.0, dst_encoding='encoded')
raw = M.FANETRouter(d=64, layers=2, dropout=0.0, dst_encoding='raw')
raw.load_state_dict(enc.state_dict())
o1 = enc(node_feat, adj, dense, cur, dst, qf, cand_idx, cf, cef, cand_mask)
o2 = raw(node_feat, adj, dense, cur, dst, qf, cand_idx, cf, cef, cand_mask)
check("7a. both settings share a checkpoint", True, "state_dict loaded cleanly")
check("7b. 'raw' gives different logits from 'encoded'",
      not bool(torch.allclose(o1, o2)),
      "h_dst leaks the destination's L-hop neighbourhood; 'raw' does not")

print("=" * 78)
if fails:
    print(f"  {len(fails)} FAILURE(S): {fails}")
    sys.exit(1)
print("  ALL VERIFICATION CHECKS PASSED")
print(f"  {M.FANETRouter(d=128, layers=2).describe()}")
print("=" * 78 + "\n")
