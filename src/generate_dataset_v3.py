"""
generate_dataset_v3.py  --  Dataset V3 generation (specification: docs/DATASET_V3_SPEC.md).

    python src\\generate_dataset_v3.py --smoke --max_workers 8          # plumbing, minutes
    python src\\generate_dataset_v3.py --out data\\v3 --max_workers 8   # the real run

WHAT CHANGED FROM generate_dataset_v2 AND WHY (each item is a measured finding)
  * ONE FILE PER EPISODE, written by the worker. A 1000 s episode at the band
    rates makes 0.7-1.2 M routing decisions (the whole v2 dataset had 533 k); v2
    held every decision of every episode as Python dicts (~2.1 KB each) in the
    parent and wrote once at the end. The parent here holds nothing.
  * TWO RECORD STREAMS from the same simulation:
      contexts  every DISTINCT decision context of the episode, once, with its
                multiplicity. A context = (frame, node, destination, hop count,
                candidate set): two decisions with the same context have an
                identical label and identical features EXCEPT own_queue_live,
                which is read live (schema v6) and changes between the copies
                (77-86% of all decisions are copies). So every context also
                keeps the exact histogram of its own-queue values over its
                occurrences (o_ctx / o_ownq / o_count). The value stored in
                c_query is only the FIRST occurrence's, and it is biased high:
                the first decision of a frame meets the fullest queue (smoke:
                56-77% of decisions differ from it, by 4-13 packets on average).
                Contexts + histograms reproduce the raw decision stream exactly
                -- lossless for imitation; export_phaseb_v3 draws the own queue
                from the histogram, never the first-occurrence value.
      steps     per-decision transition rows for a deterministic PACKET sample
                (hash(seed, pid) < rl_packet_frac), every hop of a sampled
                packet: context id, action, epsilon flag, live own-queue length,
                slot time, hop outcome, ARQ attempts. Contiguous chains for M5.
  * LABEL = da_gpsr (teacher-choice test verdict, 2026-10-02), via the restricted
    picker in teacher_pickers_v3 (pinned against the real teacher, pins proven
    able to fail). The label scores of every candidate are stored too.
  * BEHAVIOUR MIX: 70% of episodes driven by da_gpsr, 30% by gpsr / spbp /
    dijkstra_nodrop (10% each), assigned by SEED and stratified per split, so
    every split carries the same mix. epsilon = 0.10 uniform detours in every
    episode (they create 91-100% of the contexts that occur only once -- the
    recovery states).
  * ACTION -1 (drop) is RESERVED in the schema and validated by the checks; no
    policy generates it (decision 2026-10-02).
  * FEATURE SCHEMA v6: own_queue_live is the deciding node's queue read LIVE at
    decision time (excluding the packet being forwarded); every neighbour and
    2-hop queue signal stays the frame-start snapshot, which is what the teacher
    reads. Labels are unaffected: da_gpsr never reads the node's own queue.
  * OPERATING POINT from config_v2.OPERATING_POINT (1000 s, 100-300 m, battery
    8000) -- not BASE, which stays the 40 s parity reference until v8b.
  * GRID from config_v2.DATASET_GRID (usable band + one low-load anchor per
    scenario). Cells must be PASS in results/grid_verification.json
    (verify_dataset_grid_v3.py) unless --allow_unverified_cells.
  * strict queue-slot conservation, the node-id invariant checked every frame,
    per-episode metadata kept (v2 collected it and threw it away), SHA-256 of
    every array recorded for the reproducibility check.

DETERMINISM. The only random stream this file consumes is ds_rng (seed+900000),
exactly as v2: one draw per decision, plus one more when epsilon fires. Packet
sampling and behaviour assignment are pure functions of (seed, pid) / seed, so
recording more or fewer packets never changes the trajectory, and a smaller
packet sample is a strict subset of a larger one.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features_v2 as F                                       # noqa: E402
import config_v2 as C                                         # noqa: E402
import teacher_pickers_v3 as T                                # noqa: E402  (pins at import)
from simulator_v2 import FANETSimulatorV2, TTL, FRAME_DT, MAX_QUEUE  # noqa: E402
from generate_dataset_v2 import canonical_candidates          # noqa: E402
from mobility import MOBILITY_VERSION                          # noqa: E402

RECORD_SCHEMA_VERSION = 1
EPSILON = 0.10
OTHER_SHARE = 0.30
OTHER_TEACHERS = ('gpsr', 'spbp', 'dijkstra_nodrop')
RL_PACKET_FRAC = 0.10
ACTION_DROP = -1

HOP_OUTCOMES = ('moved', 'delivered', 'link_error', 'queue_overflow',
                'energy_depleted', 'policy_drop')
FATES = ('delivered', 'queue_overflow', 'link_error', 'ttl_expired', 'no_route',
         'energy_depleted', 'episode_end')
DROP_REASONS = FATES[1:]
SALT_PACKET, SALT_BEHAVIOUR, SALT_CONTEXT, SALT_OWNQ = 1, 2, 3, 4
GRID_VERIFICATION = os.path.join('results', 'grid_verification.json')
OWNQ_BASE = 64                     # (context, own queue) pair key = ctx * 64 + length
assert MAX_QUEUE < OWNQ_BASE <= 127

# v28: CODE SIGNATURE. A shard is valid only for the code that wrote it: every
# module that shapes an episode (simulator, mobility, link model, teachers,
# features, config, this generator). Recorded in every shard and the manifest;
# resume, the manifest, G3.5 v3, the export and the gate compare it with the
# current code. Line endings are normalised (git on Windows rewrites LF/CRLF).
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNATURE_MODULES = ('config_v2', 'features_v2', 'generate_dataset_v2', 'generate_dataset_v3',
                     'link_model_v2', 'mobility', 'models', 'routing_teachers',
                     'routing_teachers_v2', 'routing_teachers_v3_local', 'simulator_v2',
                     'teacher_panel', 'teacher_pickers_v3')


def code_signature(src_dir=SRC_DIR):
    h = hashlib.sha256()
    for name in SIGNATURE_MODULES:
        h.update(name.encode() + b'\0')
        with open(os.path.join(src_dir, name + '.py'), 'rb') as f:
            h.update(f.read().replace(b'\r\n', b'\n'))
    return h.hexdigest()[:16]


CODE_SIGNATURE = code_signature()

_M64 = (1 << 64) - 1


def h64(a, b, salt=0):
    """splitmix64 of (a, b, salt) -> uniform float in [0, 1). Deterministic across
    processes and platforms; consumes no simulator randomness."""
    x = (((int(a) & 0xFFFFFFFF) << 32) | (int(b) & 0xFFFFFFFF)) & _M64
    x ^= (int(salt) * 0xD1B54A32D192ED03) & _M64
    x = (x + 0x9E3779B97F4A7C15) & _M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M64
    x ^= x >> 31
    return (x >> 11) / 9007199254740992.0      # top 53 bits: exact in a float


def h64_np(a, b, salt=0):
    """Vectorised h64 over an integer array b (same bits as h64, pinned in self_test)."""
    with np.errstate(over='ignore'):
        b = np.asarray(b, dtype=np.uint64) & np.uint64(0xFFFFFFFF)
        x = (np.uint64(int(a) & 0xFFFFFFFF) << np.uint64(32)) | b
        x ^= np.uint64((int(salt) * 0xD1B54A32D192ED03) & _M64)
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) / 9007199254740992.0


def packet_sampled(seed, pid, frac):
    return h64(seed, pid, SALT_PACKET) < frac


def episode_int(key):
    """32-bit integer id of an episode key ('scenario@rate#seed') for h64."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def ownq_sample(d, key, sel):
    """One own-queue LENGTH per selected context, drawn from the context's own
    occurrence histogram with probability count / multiplicity, deterministically
    (h64(episode, context, SALT_OWNQ)). Weighted by multiplicity, the drawn values
    have the raw decision stream's own-queue distribution in expectation -- the
    first-occurrence value (c_ownq_first) does not: it is biased high."""
    sel = np.asarray(sel, np.int64)
    mult = d['c_mult'].astype(np.int64)
    excl = np.concatenate([[0], np.cumsum(mult)[:-1]])        # decisions before ctx i
    cum = np.cumsum(d['o_count'].astype(np.int64))           # pairs sorted by (ctx, len)
    u = h64_np(episode_int(key), sel, SALT_OWNQ)
    target = excl[sel] + np.minimum(np.floor(u * mult[sel]).astype(np.int64), mult[sel] - 1)
    j = np.searchsorted(cum, target, side='right')
    if len(j) and not np.array_equal(d['o_ctx'][j].astype(np.int64), sel):
        raise AssertionError('own-queue histogram inconsistent with multiplicities')
    return d['o_ownq'][j]


def behaviour_table(seeds=None):
    """seed -> behaviour policy. Stratified per split: in every split the first
    round(0.3 n) seeds in hash order run an 'other' teacher, cycled through
    OTHER_TEACHERS with a per-split offset so all three appear across splits."""
    seeds = list(C.DATASET_SEEDS if seeds is None else seeds)
    out = {}
    for k, (split, (lo, hi)) in enumerate(C.DATASET_SPLITS.items()):
        ss = [s for s in seeds if lo <= s <= hi]
        order = sorted(ss, key=lambda s: (h64(s, 0, SALT_BEHAVIOUR), s))
        n_other = int(OTHER_SHARE * len(ss) + 0.5)
        for i, s in enumerate(order):
            out[s] = (OTHER_TEACHERS[(i + 2 * k) % len(OTHER_TEACHERS)]
                      if i < n_other else 'da_gpsr')
    for s in seeds:
        if s not in out:
            raise KeyError(f'seed {s} is in no dataset split {C.DATASET_SPLITS}')
    return out


def shard_name(scenario, rate, seed):
    return os.path.join('shards', scenario, f'{scenario}_r{rate:g}_s{seed}')


# ─────────────────────────────────────────────────────────────────────────────
class DatasetSimulatorV3(FANETSimulatorV2):
    """Overrides _select_next_hop (record + act), _try_forward (hop outcome),
    _finish_packet (packet fate) and _build_graph (frame record). Everything
    else -- ARQ, queues, energy, drop taxonomy -- is the validated simulator."""

    def __init__(self, config, behaviour, epsilon=EPSILON, rl_frac=RL_PACKET_FRAC):
        config = dict(config)
        config['strict_conservation'] = True           # a phantom slot RAISES
        super().__init__(config)
        if behaviour not in T.BEHAVIOUR_TEACHERS:
            raise ValueError(f'behaviour {behaviour!r} not in {T.BEHAVIOUR_TEACHERS}')
        self.behaviour = behaviour
        self.eps = float(epsilon)
        self.rl_frac = float(rl_frac)
        self.ds_rng = np.random.default_rng(self.seed + 900_000)
        self.nc = F.norm_constants(config)
        if self.nc['schema_version'] != F.FEATURE_SCHEMA_VERSION or F.FEATURE_SCHEMA_VERSION < 6:
            raise RuntimeError('Dataset V3 requires feature schema v6 (apply v26)')
        # frames
        self._fid = -1
        self.fr_nf, self.fr_ei, self.fr_ef, self.fr_ne = [], [], [], []
        self.fr_cum = []
        # contexts: per-frame staging lists, flushed into chunks at frame end
        self._ctx_key = {}
        self._canon_cache = {}
        self._hcache = {}
        self._stage = self._new_stage()
        self._chunks = []
        self.n_ctx = 0
        self.ctx_mult = []           # global, mutable
        self._ctx_label = []         # global: label node per context
        self._ctx_beh = []           # global: behaviour choice node per context
        self._oq_cid = array('i')    # per decision: context id ...
        self._oq_len = array('b')    # ... and the live own-queue length (5 B/decision)
        # steps / packets
        self.st = {k: [] for k in ('ctx', 'pid', 'hop', 'action', 'eps', 'ownq',
                                   't', 'outcome', 'attempts')}
        self.pk = {k: [] for k in ('pid', 'flow', 'src', 'dst', 'gen_t', 'fate',
                                   'deliv_t', 'hops', 'delay_ms')}
        self._pending = None
        self._t = 0.0
        # counters
        self.c = dict(n_recorded=0, n_empty=0, n_eps=0, n_dst_adjacent=0,
                      n_dst_adjacent_not_chosen=0, n_beh_disagree=0)

    @staticmethod
    def _new_stage():
        return {k: [] for k in ('frame', 'cur', 'dst', 'hops', 'k', 'cands', 'cfeat',
                                'query', 'label', 'scores', 'votes', 'beh', 'ownq')}

    # -- frames --------------------------------------------------------------
    def _flush_stage(self):
        s = self._stage
        n = len(s['frame'])
        if n == 0:
            return
        ch = {
            'frame': np.asarray(s['frame'], np.int32),
            'cur': np.asarray(s['cur'], np.int16),
            'dst': np.asarray(s['dst'], np.int16),
            'hops': np.asarray(s['hops'], np.int8),
            'k': np.asarray(s['k'], np.int16),
            'cands': np.concatenate(s['cands']).astype(np.int16),
            'cfeat': np.concatenate(s['cfeat']).astype(np.float32),
            'query': np.stack(s['query']).astype(np.float32),
            'label': np.asarray(s['label'], np.int16),
            'scores': np.concatenate(s['scores']).astype(np.float32),
            'votes': np.asarray(s['votes'], np.int16).reshape(n, len(T.VOTE_TEACHERS)),
            'beh': np.asarray(s['beh'], np.int16),
            'ownq': np.asarray(s['ownq'], np.int8),
        }
        self._chunks.append(ch)
        self._stage = self._new_stage()

    def _build_graph(self):
        G = super()._build_graph()
        self._assert_node_id_invariant(G)          # cand ids are frame-local rows
        self._flush_stage()
        self._fid += 1
        self._ctx_key = {}
        self._canon_cache = {}
        self._hcache = {}
        _ids, nf, ei, ef = F.extract_frame(G, self.nc)
        self.fr_nf.append(nf)
        self.fr_ei.append(ei.astype(np.int16))
        self.fr_ef.append(ef)
        self.fr_ne.append(ei.shape[1])
        self.fr_cum.append(self._cum())
        return G

    def _cum(self):
        return [self.n_generated, self.n_delivered, self.n_dropped] + \
               [int(self.drop_reasons.get(r, 0)) for r in DROP_REASONS]

    # -- decisions -----------------------------------------------------------
    def _canon(self, G, c, dst, visited):
        """canonical_candidates with a per-frame cache of the sorted neighbour
        list. Sorting does not depend on `visited`, so sort-then-filter equals
        canonical_candidates' filter-then-sort exactly (pinned in self_test)."""
        key = (c, dst)
        srt = self._canon_cache.get(key)
        if srt is None:
            srt = canonical_candidates(G, c, dst, ())
            self._canon_cache[key] = srt
        return [u for u in srt if u not in visited]

    def _h(self, G, dst):
        h = self._hcache.get(dst)
        if h is None:
            h = F.hop_distances_to(G, dst)
            self._hcache[dst] = h
        return h

    def _new_context(self, G, pkt, c, dst, cands):
        h = self._h(G, dst)
        label_node, scores = T.da_gpsr_pick_scored(G, c, dst, cands)
        idx = {u: j for j, u in enumerate(cands)}
        votes = []
        for v in T.VOTE_TEACHERS:
            p = T.PICKERS[v](G, c, dst, cands, h)
            votes.append(idx[p] if p is not None else -1)
        beh_node = (label_node if self.behaviour == 'da_gpsr'
                    else T.PICKERS[self.behaviour](G, c, dst, cands, h))
        qf, cf = F.extract_decision(G, pkt, cands, self.nc, h, 0.0, 0.0,
                                    ttl_const=TTL,
                                    own_queue_occ=self.queues[c].occupancy)
        s = self._stage
        s['frame'].append(self._fid); s['cur'].append(c); s['dst'].append(dst)
        s['hops'].append(pkt.hops); s['k'].append(len(cands))
        s['cands'].append(np.asarray(cands, np.int16)); s['cfeat'].append(cf)
        s['query'].append(qf); s['label'].append(idx[label_node])
        s['scores'].append(scores); s['votes'].append(votes)
        s['beh'].append(idx[beh_node]); s['ownq'].append(self.queues[c].length)
        cid = self.n_ctx
        self.n_ctx += 1
        self.ctx_mult.append(0)
        self._ctx_label.append(label_node)
        self._ctx_beh.append(beh_node)
        return cid

    def _select_next_hop(self, G, pkt, neighbors):
        c, dst = pkt.current, pkt.dst
        cands = self._canon(G, c, dst, set(pkt.path))
        if not cands:
            self.c['n_empty'] += 1
            return None                         # base class drops it: no_route
        self.c['n_recorded'] += 1
        key = (c, dst, pkt.hops, tuple(cands))
        cid = self._ctx_key.get(key)
        if cid is None:
            cid = self._new_context(G, pkt, c, dst, cands)
            self._ctx_key[key] = cid
        self.ctx_mult[cid] += 1
        self._oq_cid.append(cid)
        self._oq_len.append(self.queues[c].length)
        label_node, beh_node = self._ctx_label[cid], self._ctx_beh[cid]
        if dst in cands:
            self.c['n_dst_adjacent'] += 1
            if label_node != dst:
                self.c['n_dst_adjacent_not_chosen'] += 1
        if beh_node != label_node:
            self.c['n_beh_disagree'] += 1
        eps_fired = bool(self.ds_rng.random() < self.eps)
        action = (cands[int(self.ds_rng.integers(len(cands)))] if eps_fired
                  else beh_node)
        if eps_fired:
            self.c['n_eps'] += 1
        if packet_sampled(self.seed, pkt.pid, self.rl_frac):
            st = self.st
            self._pending = len(st['ctx'])
            st['ctx'].append(cid); st['pid'].append(pkt.pid); st['hop'].append(pkt.hops)
            st['action'].append(cands.index(action)); st['eps'].append(eps_fired)
            st['ownq'].append(self.queues[c].length); st['t'].append(self._t)
            st['outcome'].append(-1); st['attempts'].append(-1)
        return action

    def _try_forward(self, G, t, pkt):
        self._t = t
        self._pending = None
        n_att = len(self.tx_attempts)
        finished = super()._try_forward(G, t, pkt)
        i = self._pending
        if i is not None:
            if pkt.delivered:
                oc = 'delivered'
            elif pkt.dropped:
                oc = pkt.drop_reason
            else:
                oc = 'moved'
            self.st['outcome'][i] = HOP_OUTCOMES.index(oc)
            self.st['attempts'][i] = (int(self.tx_attempts[-1])
                                      if len(self.tx_attempts) > n_att else 0)
            self._pending = None
        return finished

    def _finish_packet(self, pkt):
        super()._finish_packet(pkt)
        if packet_sampled(self.seed, pkt.pid, self.rl_frac):
            p = self.pk
            p['pid'].append(pkt.pid); p['flow'].append(pkt.flow_id)
            p['src'].append(pkt.src); p['dst'].append(pkt.dst)
            p['gen_t'].append(pkt.gen_time)
            p['fate'].append(FATES.index('delivered' if pkt.delivered else pkt.drop_reason))
            p['deliv_t'].append(pkt.delivery_time if pkt.delivered else np.nan)
            p['hops'].append(pkt.hops); p['delay_ms'].append(pkt.cum_delay_ms)

    # -- output --------------------------------------------------------------
    def ownq_table(self):
        """(context, own-queue length) pairs with their decision counts, sorted by
        context then length."""
        cid = np.asarray(self._oq_cid, dtype=np.int64)
        ln = np.asarray(self._oq_len, dtype=np.int64)
        if len(ln) and (ln.min() < 0 or ln.max() > MAX_QUEUE):
            raise AssertionError('own-queue length outside [0, MAX_QUEUE]')
        u, cnt = np.unique(cid * OWNQ_BASE + ln, return_counts=True)
        return ((u // OWNQ_BASE).astype(np.int32), (u % OWNQ_BASE).astype(np.int8),
                cnt.astype(np.int32))

    def arrays(self):
        self._flush_stage()
        cat = lambda key, dt: (np.concatenate([c[key] for c in self._chunks]).astype(dt)
                               if self._chunks else np.zeros(0, dt))
        n_frames = len(self.fr_nf)
        o_ctx, o_ownq, o_count = self.ownq_table()
        out = {
            # frames
            'f_node_feat': np.concatenate(self.fr_nf).astype(np.float32),
            'f_n_nodes': np.full(n_frames, self.N, np.int16),
            'f_edge_index': np.concatenate(self.fr_ei, axis=1).astype(np.int16),
            'f_edge_feat': np.concatenate(self.fr_ef).astype(np.float32),
            'f_n_edges': np.asarray(self.fr_ne, np.int32),
            'f_t': (np.arange(n_frames) * FRAME_DT).astype(np.float32),
            'f_cum': np.asarray(self.fr_cum + [self._cum()], np.int64),
            'f_mean_occ': np.asarray(self.ts_mean_occ, np.float32),
            'f_max_occ': np.asarray(self.ts_max_occ, np.float32),
            'f_mean_activity': np.asarray(self.ts_mean_activity, np.float32),
            'f_inflight': np.asarray(self.ts_inflight, np.int32),
            'f_mean_lq': np.asarray(self.ts_mean_linkq, np.float32),
            # contexts
            'c_frame': cat('frame', np.int32), 'c_current': cat('cur', np.int16),
            'c_dst': cat('dst', np.int16), 'c_hops': cat('hops', np.int8),
            'c_k': cat('k', np.int16), 'c_cands': cat('cands', np.int16),
            'c_cand_feat': (np.concatenate([c['cfeat'] for c in self._chunks]).astype(np.float32)
                            if self._chunks else np.zeros((0, len(F.CANDIDATE_FEATURES)), np.float32)),
            'c_query': (np.concatenate([c['query'] for c in self._chunks]).astype(np.float32)
                        if self._chunks else np.zeros((0, len(F.QUERY_FEATURES)), np.float32)),
            'c_label': cat('label', np.int16), 'c_scores': cat('scores', np.float32),
            'c_votes': (np.concatenate([c['votes'] for c in self._chunks]).astype(np.int16)
                        if self._chunks else np.zeros((0, len(T.VOTE_TEACHERS)), np.int16)),
            'c_beh': cat('beh', np.int16), 'c_ownq_first': cat('ownq', np.int8),
            'c_mult': np.asarray(self.ctx_mult, np.int32),
            # own-queue histogram per context (the one feature that differs
            # between a context's occurrences)
            'o_ctx': o_ctx, 'o_ownq': o_ownq, 'o_count': o_count,
            # steps
            's_ctx': np.asarray(self.st['ctx'], np.int32),
            's_pid': np.asarray(self.st['pid'], np.int32),
            's_hop': np.asarray(self.st['hop'], np.int8),
            's_action': np.asarray(self.st['action'], np.int16),
            's_eps': np.asarray(self.st['eps'], bool),
            's_ownq': np.asarray(self.st['ownq'], np.int8),
            's_t': np.asarray(self.st['t'], np.float32),
            's_outcome': np.asarray(self.st['outcome'], np.int8),
            's_attempts': np.asarray(self.st['attempts'], np.int8),
            # packets (sampled only)
            'p_pid': np.asarray(self.pk['pid'], np.int32),
            'p_flow': np.asarray(self.pk['flow'], np.int16),
            'p_src': np.asarray(self.pk['src'], np.int16),
            'p_dst': np.asarray(self.pk['dst'], np.int16),
            'p_gen_t': np.asarray(self.pk['gen_t'], np.float32),
            'p_fate': np.asarray(self.pk['fate'], np.int8),
            'p_deliv_t': np.asarray(self.pk['deliv_t'], np.float32),
            'p_hops': np.asarray(self.pk['hops'], np.int8),
            'p_delay_ms': np.asarray(self.pk['delay_ms'], np.float32),
        }
        return out


def array_hashes(arrs):
    return {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()[:16]
            for k, v in sorted(arrs.items())}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def run_episode(scenario, rate, seed, behaviour, epsilon=EPSILON,
                rl_frac=RL_PACKET_FRAC, duration=None):
    """Simulate one episode; return (arrays, meta). Module-level for pickling."""
    over = {'scenario_id': scenario, 'actor': 'da_gpsr'}
    if duration is not None:
        over['duration'] = float(duration)
    cfg = C.dataset_episode_config(scenario, rate, seed, **over)
    t0 = time.time()
    sim = DatasetSimulatorV3(cfg, behaviour, epsilon=epsilon, rl_frac=rl_frac)
    m = sim.run()
    arrs = sim.arrays()
    n_dec = int(sim.c['n_recorded'])
    if int(arrs['c_mult'].sum()) != n_dec:
        raise AssertionError('context multiplicities do not sum to recorded decisions')
    if n_dec + sim.c['n_empty'] != sim.n_decisions:
        raise AssertionError('recorded + empty-candidate decisions != simulator decisions')
    if m.get('n_phantom_slots', 0):
        raise AssertionError('phantom queue slots -- v12 leak present')
    if int((arrs['s_outcome'] < 0).sum()):
        raise AssertionError('a recorded step never received its hop outcome')
    n_ctx = len(arrs['c_mult'])
    o_ctx, o_len, o_cnt = (arrs['o_ctx'].astype(np.int64), arrs['o_ownq'].astype(np.int64),
                           arrs['o_count'].astype(np.int64))
    if not np.array_equal(np.bincount(o_ctx, weights=o_cnt, minlength=n_ctx).astype(np.int64),
                          arrs['c_mult'].astype(np.int64)):
        raise AssertionError('own-queue histogram does not sum to the context multiplicities')
    pair = o_ctx * OWNQ_BASE + o_len
    first = np.arange(n_ctx) * OWNQ_BASE + arrs['c_ownq_first'].astype(np.int64)
    if not np.isin(first, pair).all():
        raise AssertionError("a context's first own-queue value is missing from its histogram")
    if not np.isin(arrs['s_ctx'].astype(np.int64) * OWNQ_BASE + arrs['s_ownq'], pair).all():
        raise AssertionError("a step's own-queue value is missing from its context histogram")
    f_len = arrs['c_ownq_first'].astype(np.int64)[o_ctx]
    ownq_diag = {
        'n_pairs': int(len(o_cnt)),
        'share_decisions_differing_from_first': float((o_cnt * (o_len != f_len)).sum() / max(n_dec, 1)),
        'mean_first_minus_decision': float((o_cnt * (f_len - o_len)).sum() / max(n_dec, 1)),
        'mean_decision_length': float((o_cnt * o_len).sum() / max(n_dec, 1)),
    }
    g = max(m['n_generated'], 1)
    meta = {
        'complete': True,
        'record_schema_version': RECORD_SCHEMA_VERSION,
        'code_signature': CODE_SIGNATURE, 'mobility': MOBILITY_VERSION,
        'feature_schema_version': F.FEATURE_SCHEMA_VERSION,
        'key': f'{scenario}@{rate:g}#{seed}',
        'scenario': scenario, 'suite': C.DATASET_GRID[scenario]['suite'],
        'rate': float(rate), 'rate_index': C.rate_index(scenario, rate),
        'rate_role': C.rate_role(scenario, rate),
        'load_bucket': C.load_bucket_rel(scenario, rate),
        'seed': int(seed), 'split': C.split_of_seed(seed),
        'generalisation': scenario == C.GENERALISATION_SCENARIO,
        'behaviour': behaviour, 'epsilon': float(epsilon), 'rl_packet_frac': float(rl_frac),
        'config': _jsonable(cfg), 'norm_constants': _jsonable(sim.nc),
        'metrics': {k: _jsonable(m.get(k)) for k in (
            'network_pdr', 'pdr_predrain', 'n_generated', 'n_delivered', 'n_dropped',
            'drop_reasons', 'mean_hops', 'mean_delay_ms', 'override_rate',
            'n_decisions', 'mean_queue_occ', 'max_queue_occ', 'mean_tx_attempts')},
        'drop_share': {k: v / g for k, v in m['drop_reasons'].items()},
        'counts': {
            'n_frames': int(len(arrs['f_n_edges'])), 'n_contexts': int(len(arrs['c_mult'])),
            'n_steps': int(len(arrs['s_ctx'])), 'n_packets_sampled': int(len(arrs['p_pid'])),
            'n_recorded_decisions': n_dec, 'n_empty_candidate_decisions': int(sim.c['n_empty']),
            'n_eps_fired': int(sim.c['n_eps']),
            'n_behaviour_disagrees_with_label': int(sim.c['n_beh_disagree']),
            'n_dst_adjacent': int(sim.c['n_dst_adjacent']),
            'n_dst_adjacent_not_chosen': int(sim.c['n_dst_adjacent_not_chosen']),
            'dead_nodes': int(sum(1 for e in sim.energy if e <= 0)),
            'n_ownq_pairs': ownq_diag['n_pairs'],
        },
        'own_queue': ownq_diag,
        'seconds': time.time() - t0,
        'sha256_16': array_hashes(arrs),
    }
    return arrs, meta


def _job(job):
    out, scenario, rate, seed, behaviour, eps, rl_frac, duration = job
    base = os.path.join(out, shard_name(scenario, rate, seed))
    os.makedirs(os.path.dirname(base), exist_ok=True)
    arrs, meta = run_episode(scenario, rate, seed, behaviour, eps, rl_frac, duration)
    tmp = base + '.tmp.npz'
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, base + '.npz')
    meta['shard'] = os.path.relpath(base + '.npz', out).replace('\\', '/')
    meta['bytes'] = os.path.getsize(base + '.npz')
    with open(base + '.json.tmp', 'w') as f:
        json.dump(meta, f, indent=1)
    os.replace(base + '.json.tmp', base + '.json')
    return meta


def shard_complete(out, scenario, rate, seed):
    base = os.path.join(out, shard_name(scenario, rate, seed))
    if not (os.path.isfile(base + '.npz') and os.path.isfile(base + '.json')):
        return False
    try:
        m = json.load(open(base + '.json'))
    except (OSError, ValueError):
        return False
    return bool(m.get('complete')) and m.get('record_schema_version') == RECORD_SCHEMA_VERSION \
        and 'n_ownq_pairs' in m.get('counts', {}) and m.get('code_signature') == CODE_SIGNATURE


# ─────────────────────────────────────────────────────────────────────────────
def load_grid_verification(path):
    if not os.path.isfile(path):
        return None
    return json.load(open(path))


def unverified_cells(gv, cells):
    """Cells that are not PASS in the grid verification. v28: a verification written
    for another mobility version verifies nothing -- its bands were measured on
    another simulator."""
    gv = gv or {}
    out = []
    for s, r in cells:
        st = gv.get('cells', {}).get(f'{s}@{r:g}', {}).get('status')
        if st == 'PASS' and gv.get('mobility') != MOBILITY_VERSION:
            st = f"PASS-but-mobility-{gv.get('mobility') or 'pre-v28'}"
        if st != 'PASS':
            out.append(f'{s}@{r:g}={st or "missing"}')
    return out


def self_test():
    """Cheap import-time checks of THIS file's own shortcuts."""
    # 1. the canonical-candidate cache equals canonical_candidates exactly
    import networkx as nx
    rng = np.random.default_rng(3)
    for _ in range(200):
        G = T._random_graph(rng)
        n = G.number_of_nodes()
        c, d = 0, n - 1
        nb = list(G.neighbors(c))
        vis = {c} | set(rng.choice(nb, size=int(rng.integers(0, len(nb) + 1)),
                                   replace=False).tolist()) if nb else {c}
        srt = canonical_candidates(G, c, d, ())
        if [u for u in srt if u not in vis] != canonical_candidates(G, c, d, vis):
            raise AssertionError('canonical cache is not equivalent to canonical_candidates')
    # 2. schema v6 refuses a snapshot own-queue value it was not given
    G = nx.Graph()
    G.add_node(0, x=0.0, y=0.0, z=100.0, queue_occupancy=0.0)
    G.add_node(1, x=100.0, y=0.0, z=100.0, queue_occupancy=0.0)
    G.add_edge(0, 1, link_quality=1.0)

    class _P:
        current, dst, hops, path = 0, 1, 0, [0]
    nc = F.norm_constants({'area_x': 1000, 'area_y': 1000, 'z_min': 100, 'z_max': 300,
                           'comm_range': 250, 'num_drones': 2})
    try:
        F.extract_decision(G, _P(), [1], nc, {0: 1, 1: 0}, 0.0, 0.0, ttl_const=TTL)
    except ValueError:
        pass
    else:
        raise AssertionError('schema v6 accepted a missing own_queue_occ (silent snapshot)')
    # 3. the vectorised hash is bit-identical to the scalar one
    for a, salt in ((101, SALT_PACKET), (149, SALT_CONTEXT), (7, SALT_BEHAVIOUR)):
        bs = np.arange(0, 20000, 7)
        if not np.array_equal(h64_np(a, bs, salt), np.array([h64(a, int(b), salt) for b in bs])):
            raise AssertionError('h64_np differs from h64')
    # 4. packet sampling is nested across fractions
    for s in (101, 136):
        a = {p for p in range(5000) if packet_sampled(s, p, 0.03)}
        b = {p for p in range(5000) if packet_sampled(s, p, 0.10)}
        if not a <= b:
            raise AssertionError('packet sample is not nested')
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--out', default=os.path.join('data', 'v3'))
    ap.add_argument('--max_workers', type=int, default=8,
                    help='default 8: on Windows 16 spawn workers stalled a run (project report)')
    ap.add_argument('--cells', nargs='+', default=None,
                    help="'scenario' or 'scenario@rate' filters; default: the whole grid")
    ap.add_argument('--seeds', type=int, nargs='+', default=None)
    ap.add_argument('--rl_packet_frac', type=float, default=RL_PACKET_FRAC)
    ap.add_argument('--epsilon', type=float, default=EPSILON)
    ap.add_argument('--grid_verification', default=GRID_VERIFICATION)
    ap.add_argument('--allow_unverified_cells', action='store_true')
    ap.add_argument('--smoke', action='store_true',
                    help='60 s episodes, 2 seeds, one band rate per scenario, '
                         'out=<out>_smoke. Numbers are meaningless; plumbing only.')
    ap.add_argument('--duration', type=float, default=None, help='--smoke only')
    ap.add_argument('--manifest_only', action='store_true',
                    help='(re)build manifest.json from the shards already on disk')
    args = ap.parse_args()

    if args.duration is not None and not args.smoke:
        raise SystemExit('--duration changes the operating point; allowed with --smoke only')
    out = args.out + ('_smoke' if args.smoke and not args.out.endswith('_smoke') else '')

    print('=' * 78)
    print('  DATASET V3 GENERATION' + ('   *** SMOKE: numbers meaningless ***' if args.smoke else ''))
    print('=' * 78)
    self_test()
    T.assert_pins(verbose=True)
    print('  self-test: canonical cache exact, v6 refuses a missing live own-queue, '
          'packet sample nested  OK')
    # (multiprocessing aliases __main__ as __mp_main__ in every process that imports it)
    stray = sorted(n for n, mod in list(sys.modules.items())
                   if n not in ('__main__', '__mp_main__') and getattr(mod, '__file__', None)
                   and os.path.dirname(os.path.abspath(mod.__file__)) == SRC_DIR
                   and n not in SIGNATURE_MODULES)
    if stray:
        raise SystemExit(f'  the code signature misses modules the generator loaded: {stray} '
                         f'-- add them to SIGNATURE_MODULES')
    print(f'  code signature {CODE_SIGNATURE} (mobility {MOBILITY_VERSION}): shards written '
          f'by other code are regenerated, never resumed')

    if args.manifest_only:
        man = build_manifest(out, smoke=args.smoke)
        print(f"  manifest rebuilt: {man['totals']}")
        return 0

    cells = C.dataset_cells()
    if args.cells:
        want = set(args.cells)
        cells = [(s, r) for s, r in cells if s in want or f'{s}@{r:g}' in want]
    seeds = list(args.seeds) if args.seeds else list(C.DATASET_SEEDS)
    duration = None
    if args.smoke:
        duration = args.duration or 60.0
        seeds = list(args.seeds) if args.seeds else [101, 136]
        if not args.cells:
            cells = [(s, max(C.DATASET_GRID[s]['band'])) for s in C.DATASET_GRID]

    # grid gate
    gv = load_grid_verification(args.grid_verification)
    unverified = unverified_cells(gv, cells)
    if unverified and not (args.allow_unverified_cells or args.smoke):
        raise SystemExit(
            '  GRID GATE: these cells are not PASS in '
            f'{args.grid_verification}:\n    ' + '\n    '.join(unverified) +
            '\n  Run src\\verify_dataset_grid_v3.py first (v28: every band is re-measured '
            'with the current mobility -- the verifier prints the commands), or pass '
            '--allow_unverified_cells to '
            'generate them anyway (recorded in the manifest).')

    table = behaviour_table()
    jobs = [(out, s, r, sd, table[sd], args.epsilon, args.rl_packet_frac, duration)
            for s, r in cells for sd in seeds if not shard_complete(out, s, r, sd)]
    op = dict(C.OPERATING_POINT)
    if duration:
        op['duration'] = duration
    print(f'  out={out}  operating point {op}')
    print(f'  cells={len(cells)}  seeds={len(seeds)}  episodes to run={len(jobs)} '
          f'(resume-aware)  epsilon={args.epsilon}  rl_packet_frac={args.rl_packet_frac}')
    if unverified:
        print(f'  UNVERIFIED cells generated anyway: {unverified}')
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            m = fu.result()
            el = time.time() - t0
            print(f"    {i}/{len(jobs)}  {m['key']:<24} {m['behaviour']:<15} "
                  f"pdr={m['metrics']['network_pdr']:.3f}  "
                  f"decisions={m['counts']['n_recorded_decisions']:>8}  "
                  f"contexts={m['counts']['n_contexts']:>7}  ownq-pairs={m['counts']['n_ownq_pairs']:>7}  "
                  f"steps={m['counts']['n_steps']:>6}  "
                  f"{m['bytes']/1e6:.0f}MB  {m['seconds']:.0f}s  "
                  f"(~{el / i * (len(jobs) - i) / 60:.0f} min left)")
    man = build_manifest(out, smoke=args.smoke, unverified=unverified)
    tt = man['totals']
    print(f"\n  manifest: {tt['episodes']} episodes  {tt['contexts']:,} contexts  "
          f"{tt['ownq_pairs']:,} (context, own queue) pairs  "
          f"{tt['steps']:,} steps  {tt['decisions']:,} decisions  {tt['gb']:.2f} GB")
    print(f'  total {time.time() - t0:.0f}s')
    return 0


def build_manifest(out, smoke=False, unverified=None):
    metas, skipped = [], []
    root = os.path.join(out, 'shards')
    for dp, _dn, fn in os.walk(root):
        for f in sorted(fn):
            if f.endswith('.json') and not f.endswith('.tmp'):
                m = json.load(open(os.path.join(dp, f)))
                if (m.get('complete') and m.get('record_schema_version') == RECORD_SCHEMA_VERSION
                        and 'n_ownq_pairs' in m.get('counts', {})
                        and m.get('code_signature') == CODE_SIGNATURE):
                    metas.append(m)
                else:
                    skipped.append(f)
                    print(f'  manifest: skipping incomplete or stale shard {f}')
    if skipped and not metas:
        # v28: rebuilding the manifest of a pre-v28 dataset would silently empty it
        raise SystemExit(f'  manifest NOT written: all {len(skipped)} shards in {out} are incomplete '
                         f'or written by other code (current signature {CODE_SIGNATURE}). '
                         f'Regenerate into a new --out; the existing manifest.json is untouched.')
    metas.sort(key=lambda m: (m['scenario'], m['rate'], m['seed']))
    norm = {}
    for m in metas:
        nc = m['norm_constants']
        if m['scenario'] in norm and norm[m['scenario']] != nc:
            raise AssertionError(f"norm constants differ between episodes of {m['scenario']}")
        norm[m['scenario']] = nc
    table = behaviour_table()
    man = {
        'dataset': 'FANET Dataset V3', 'spec': 'docs/DATASET_V3_SPEC.md',
        'record_schema_version': RECORD_SCHEMA_VERSION,
        'feature_schema_version': F.FEATURE_SCHEMA_VERSION,
        'node_features': F.NODE_FEATURES, 'edge_features': F.EDGE_FEATURES,
        'query_features': F.QUERY_FEATURES, 'candidate_features': F.CANDIDATE_FEATURES,
        'local_horizon': F.LOCAL_HORIZON,
        'label_teacher': T.LABEL_TEACHER, 'vote_teachers': list(T.VOTE_TEACHERS),
        'behaviour': {'other_share': OTHER_SHARE, 'other_teachers': list(OTHER_TEACHERS),
                      'table': {str(k): v for k, v in sorted(table.items())},
                      'epsilon': EPSILON},
        'rl_packet_frac': sorted({m['rl_packet_frac'] for m in metas}),
        'hash': {'function': 'splitmix64', 'salts': {'packet': SALT_PACKET,
                 'behaviour': SALT_BEHAVIOUR, 'context': SALT_CONTEXT, 'own_queue': SALT_OWNQ}},
        'codebooks': {'hop_outcome': list(HOP_OUTCOMES), 'fate': list(FATES),
                      'action_drop': ACTION_DROP, 'votes': list(T.VOTE_TEACHERS),
                      'f_cum': ['n_generated', 'n_delivered', 'n_dropped'] + list(DROP_REASONS)},
        'provenance': C.dataset_provenance(),
        'split_plan': {'train_seeds': list(C.DATASET_SPLITS['train']),
                       'val_seeds': list(C.DATASET_SPLITS['val']),
                       'test_seeds': list(C.DATASET_SPLITS['test']),
                       'generalisation_scenario': C.GENERALISATION_SCENARIO},
        'norm_constants_per_scenario': norm,
        'smoke': bool(smoke), 'unverified_cells_generated': list(unverified or []),
        'episodes': [{k: m[k] for k in ('key', 'shard', 'scenario', 'rate', 'rate_index',
                                        'rate_role', 'load_bucket', 'seed', 'split',
                                        'behaviour', 'bytes')}
                     | {'n_contexts': m['counts']['n_contexts'],
                        'n_ownq_pairs': m['counts']['n_ownq_pairs'],
                        'n_steps': m['counts']['n_steps'],
                        'n_decisions': m['counts']['n_recorded_decisions'],
                        'pdr': m['metrics']['network_pdr']} for m in metas],
        'totals': {'episodes': len(metas),
                   'contexts': sum(m['counts']['n_contexts'] for m in metas),
                   'ownq_pairs': sum(m['counts']['n_ownq_pairs'] for m in metas),
                   'steps': sum(m['counts']['n_steps'] for m in metas),
                   'decisions': sum(m['counts']['n_recorded_decisions'] for m in metas),
                   'frames': sum(m['counts']['n_frames'] for m in metas),
                   'gb': sum(m['bytes'] for m in metas) / 1e9},
        'code_signature': CODE_SIGNATURE, 'mobility': MOBILITY_VERSION,
        'generator_sha256_16': hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16],
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(out, 'manifest.json.tmp'), 'w') as f:
        json.dump(man, f, indent=1)
    os.replace(os.path.join(out, 'manifest.json.tmp'), os.path.join(out, 'manifest.json'))
    return man


if __name__ == '__main__':
    sys.exit(main())
