"""apply_delivery_leak_fix_v12.py -- fix the destination-queue leak.

Assertion-guarded str.replace. Anchors must match EXACTLY ONCE; edits staged in
memory; nothing written unless every anchor matches. Idempotent.

THE DEFECT

`_try_forward` enqueues every packet at `next_hop` BEFORE checking whether
`next_hop` is the destination. Delivery is detected afterwards, the packet is
marked delivered and dropped from `active`, and its queue slot is NEVER
released -- the only `buffer.remove` in the file is for packets being forwarded.
Destination queues therefore fill permanently and never drain.

MEASURED IMPACT

  Convergecast (all traffic to one sink): delivery pinned at exactly MAX_QUEUE=50
  in 400/400 episodes across a 43x range of offered traffic (228..9900 generated).
  All 8 teachers scored identically because the cap bound everyone equally.

  Suite A (random destinations -- leak spread over many nodes, so it degrades
  smoothly instead of hitting a wall):
      dur  200 s  ->  PDR 96.8%   max queue occupancy 18%
      dur  400 s  ->  PDR 89.5%   max queue occupancy 38%
      dur  700 s  ->  PDR 71.0%   max queue occupancy 60%
      dur 1000 s  ->  PDR 63.9%   max queue occupancy 78%
  ...with ZERO overflow drops. The decay is entirely phantom occupancy.

  Queue-slot audit at episode end: 102-356 slots held, 0 packets actually in
  flight. 100% phantom.

SEMANTICS -- DECIDED AND CONFIRMED (user, this session)

A delivered packet is CONSUMED ON ARRIVAL and occupies zero buffer. Rationale:
  * it is what real stacks and mainstream network simulators do -- an arriving
    packet is passed up to the application and its buffer is freed;
  * the alternative (hold one slot for one service interval, then drain) needs
    NEW machinery, because a delivered packet is removed from `active` and the
    slot loop only services `active` packets, so nothing would ever drain it;
  * per-hop processing delay is already modelled by HOP_DELAY_MS.
CONFIRMED BY THE USER as the project decision. Recorded here so it is a decision
on the record rather than an implementation default.

SIDE EFFECT, DELIBERATE. Checking the destination BEFORE admission also removes
spurious `queue_overflow` drops at the destination itself. A packet that has
physically arrived should not be tail-dropped by the queue of the node it
arrived at. This CHANGES measured numbers beyond simply freeing slots, and is
part of why every 1000 s measurement must be re-run.

VALIDATION -- THE FIX IS NOT TOO PERMISSIVE

Post-fix, dense_slow shows PDR 1.0000 at the old rates, which looks alarming until
you check capacity: per-node service is SLOTS_PER_FRAME/FRAME_DT = 50/0.5 = 100
pkt/s, while rate 4.0 offers 7 flows x 4.0 = 28 pkt/s ACROSS THE WHOLE NETWORK.
The old grid simply never approached capacity. Stress test confirms congestion
appears exactly where it should:

    rate    offered      pdr   q_ovf  link_err  phantom
     4.0     28 p/s   1.0000       0         0        0
    20.0    140 p/s   0.9898       0        43        0
    60.0    420 p/s   0.6441      96      2643        0
   150.0   1050 p/s   0.1409   12495      4171        0

Real queue overflow, real link errors, zero phantom slots. The fix is correct;
the old operating point was far below the congestion regime it claimed to test.

WHAT ELSE THIS PATCH ADDS

  * QUEUE-SLOT CONSERVATION INVARIANT (the check that was missing). Packet-count
    conservation -- generated == delivered + dropped -- PASSED 8/8 cells even
    with the leak present, which is exactly why 4 milestones of gates never
    caught it. The right invariant is on SLOTS: every packet occupying a queue
    slot must be neither delivered nor dropped. Enabled by default at episode
    end; `--strict-conservation` config key makes it raise instead of warn.
  * NODE-ID INVARIANT ASSERTION. `cand_flat` stores GLOBAL drone ids but they
    are used directly as FRAME-LOCAL row indices to gather node embeddings
    (`dense[fr, cu, cand_idx]`). These coincide only because every frame graph
    contains all N nodes with ids 0..N-1 in sorted order. Verified true today
    (80 frames x 4 scenarios, 0 violations) but nothing asserted it. If a future
    change ever drops isolated nodes -- a natural optimisation in sparse_fast --
    the GNN would silently gather the WRONG nodes with no error.

USAGE
    python apply_delivery_leak_fix_v12.py --src src --dry-run
    python apply_delivery_leak_fix_v12.py --src src
    python verify_delivery_leak_fix_v12.py --src src
"""
import argparse, io, os, sys

TARGET = 'simulator_v2.py'
GUARD = '_assert_queue_conservation'

# ── 1. the leak fix: check destination BEFORE queue admission ──────────────
OLD_A = """        # receiver queue admission (tail-drop)
        if not self.queues[next_hop].enqueue(pkt):
            pkt.dropped = True; pkt.drop_reason = 'queue_overflow'
            self._record_transition(pkt, next_hop, -10.0, True, dropped=True,
                                    lq=lq_eff)
            return True

        # commit the hop (TX energy already charged per attempt in the ARQ loop)"""

NEW_A = """        # v12 LEAK FIX. Determine delivery BEFORE queue admission.
        #
        # Previously every packet -- including one arriving at its own
        # destination -- was enqueued here, then marked delivered below and
        # removed from `active`. Nothing ever released the slot, because the
        # only buffer.remove() in this file is for packets being FORWARDED. So
        # destination queues filled permanently and never drained. In
        # convergecast this pinned delivery at exactly MAX_QUEUE forever; in
        # Suite A it decayed PDR smoothly with episode length, with zero
        # overflow drops, which is why it went unnoticed for four milestones.
        #
        # SEMANTICS (recorded decision): a delivered packet is consumed on
        # arrival and occupies zero buffer. See the module docstring.
        delivered_now = (next_hop == dst)

        # receiver queue admission (tail-drop) -- ONLY for packets that still
        # need forwarding. A packet that has physically arrived is not
        # tail-dropped by the queue of the node it arrived at.
        if not delivered_now:
            if not self.queues[next_hop].enqueue(pkt):
                pkt.dropped = True; pkt.drop_reason = 'queue_overflow'
                self._record_transition(pkt, next_hop, -10.0, True, dropped=True,
                                        lq=lq_eff)
                return True

        # commit the hop (TX energy already charged per attempt in the ARQ loop)"""

# ── 2. remove the now-duplicate delivery determination ─────────────────────
OLD_B = """        delivered_now = (next_hop == dst)
        if delivered_now:
            pkt.delivered = True
            pkt.delivery_time = t"""

NEW_B = """        # delivered_now was computed above, before queue admission (v12)
        if delivered_now:
            pkt.delivered = True
            pkt.delivery_time = t"""

# ── 3. the conservation invariant + node-id assertion ──────────────────────
OLD_C = """    def _make_obs(self, G, pkt, neighbors):"""

NEW_C = '''    def _assert_queue_conservation(self, where=''):
        """QUEUE-SLOT CONSERVATION -- the invariant whose absence hid the v12 leak.

        Packet-COUNT conservation (generated == delivered + dropped) passed 8/8
        cells even with the leak present, because delivered packets were counted
        correctly; it was their SLOTS that leaked. The right invariant is:

            every packet occupying a queue slot is neither delivered nor dropped

        Returns the number of phantom slots (0 when healthy). Raises instead of
        warning when the config sets strict_conservation=True.
        """
        # 'episode_end' is NOT a leak: those packets were legitimately in flight,
        # occupying real queue slots, when the episode stopped. The drain sweep
        # marks them dropped just before _metrics() runs, so counting them would
        # be a false positive of this check's own placement -- which is exactly
        # what the first draft did, flagging 9 slots in sparse_fast.
        phantom = [(nid, p.pid, p.drop_reason) for nid, q in enumerate(self.queues)
                   for p in q.buffer
                   if (p.delivered or p.dropped) and p.drop_reason != 'episode_end']
        if phantom:
            msg = (f"QUEUE-SLOT CONSERVATION VIOLATED{(' at ' + where) if where else ''}: "
                   f"{len(phantom)} slot(s) held by already-delivered/dropped packets "
                   f"(first: node={phantom[0][0]} pid={phantom[0][1]} "
                   f"reason={phantom[0][2]!r}). "
                   f"This is the v12 destination-queue leak or a regression of it.")
            # Read from self.cfg, matching how every other option in this class
            # is read. A first draft used getattr(self, 'strict_conservation')
            # -- an attribute nothing ever sets -- so strict mode silently
            # warned instead of raising. Caught by the negative control.
            if bool(self.cfg.get('strict_conservation', False)):
                raise AssertionError(msg)
            import warnings
            warnings.warn(msg, RuntimeWarning)
        return len(phantom)

    def _assert_node_id_invariant(self, G):
        """NODE-ID INVARIANT -- load-bearing for the GNN, previously unasserted.

        `cand_flat` stores GLOBAL drone ids, but they are used directly as
        FRAME-LOCAL row indices to gather node embeddings. That is only valid
        while every frame graph holds all N nodes with ids 0..N-1 in sorted
        order, making features_v2's idx_of the identity map. True today
        (verified: 80 frames x 4 scenarios, 0 violations) but nothing checked
        it. If a future change drops isolated nodes -- a natural optimisation in
        sparse_fast -- the GNN would silently gather the WRONG nodes' embeddings
        with no error raised anywhere.
        """
        n = G.number_of_nodes()
        if n != self.N or min(G.nodes()) != 0 or max(G.nodes()) != self.N - 1:
            raise AssertionError(
                f"NODE-ID INVARIANT VIOLATED: frame graph has {n} nodes with id "
                f"range [{min(G.nodes())}, {max(G.nodes())}], expected {self.N} "
                f"nodes with ids 0..{self.N - 1}. Global candidate ids are used "
                f"as frame-local embedding row indices and would now gather the "
                f"WRONG nodes silently. See simulator_v2._assert_node_id_invariant.")

    def _make_obs(self, G, pkt, neighbors):'''

# ── 4. wire the conservation check in at episode end ───────────────────────
OLD_D = """    def _metrics(self):
        pdr = self.n_delivered / max(self.n_generated, 1)
        return {"""

NEW_D = """    def _metrics(self):
        # v12: check the slot invariant once per episode. Cheap (one pass over
        # the queues) and it is the check that would have caught the leak.
        n_phantom = self._assert_queue_conservation('episode end')
        pdr = self.n_delivered / max(self.n_generated, 1)
        return {
            'n_phantom_slots': n_phantom,"""

# ── 5. SECOND LEAK: the TTL sweep drops a packet without freeing its slot ──
# Found by the v12 conservation invariant on its first Suite A run -- the check
# earned its keep immediately. Same defect class as the delivery leak: the
# packet is marked dropped and removed from `active`, but nothing removes it
# from the queue it is sitting in, so the slot is held forever.
OLD_E = """                survivors = []
                for pkt in active:
                    if pkt.hops >= TTL and not (pkt.delivered or pkt.dropped):
                        pkt.dropped = True; pkt.drop_reason = 'ttl_expired'
                        self._finish_packet(pkt)
                    else:
                        survivors.append(pkt)
                active = survivors"""

NEW_E = """                survivors = []
                for pkt in active:
                    if pkt.hops >= TTL and not (pkt.delivered or pkt.dropped):
                        pkt.dropped = True; pkt.drop_reason = 'ttl_expired'
                        # v12: free the slot. Without this the TTL-dropped packet
                        # keeps occupying its queue forever -- the same leak class
                        # as the delivery bug, found by the new invariant on its
                        # first run.
                        q = self.queues[pkt.current]
                        if pkt in q.buffer:
                            q.buffer.remove(pkt)
                        self._finish_packet(pkt)
                    else:
                        survivors.append(pkt)
                active = survivors"""

EDITS = [(OLD_A, NEW_A), (OLD_B, NEW_B), (OLD_C, NEW_C), (OLD_D, NEW_D),
         (OLD_E, NEW_E)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='src')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    path = os.path.join(a.src, TARGET)
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found"); return 1
    text = io.open(path, encoding='utf-8').read()
    if GUARD in text:
        print("  ALREADY APPLIED. Nothing to do."); return 0

    staged, ok = text, True
    for i, (old, new) in enumerate(EDITS, 1):
        n = staged.count(old)
        if n != 1:
            print(f"  anchor {i}: matched {n} times, expected 1  <-- ABORT")
            ok = False
        else:
            print(f"  anchor {i}: OK")
            staged = staged.replace(old, new, 1)

    if not ok:
        print("\n  NO FILE WRITTEN."); return 1
    if a.dry_run:
        print(f"\n  DRY RUN OK -- {len(EDITS)}/{len(EDITS)} anchors matched. "
              f"Nothing written."); return 0

    io.open(path, 'w', encoding='utf-8').write(staged)
    print(f"\n  WROTE {path}")
    print("  NEXT: python verify_delivery_leak_fix_v12.py --src " + a.src)
    print("\n  REMINDER: this CHANGES measured numbers. Every 1000 s measurement")
    print("  must be re-run -- convergecast panel, rate probe, headroom.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
