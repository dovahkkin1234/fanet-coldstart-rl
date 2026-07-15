"""
link_model_v2.py
Interference-aware wireless link model for multi-packet FANET simulation.

Extends link_model.py (log-distance path loss, distance-only) with the three
pieces of physics that turn "link quality" from a static function of distance
into a dynamic function of distance AND current network load:

  1. SINR (signal-to-interference-plus-noise ratio)
        SINR = P_signal / (N0 + Sum_k P_interferer_k)
     where each concurrent transmitter within interference range contributes
     its received power at the receiver. Replaces the v1 SNR everywhere.

  2. Bianchi (2000) CSMA/CA collision model
        MAC-layer collision probability as a function of the number of
        contending stations in carrier-sense range, solved from the standard
        Bianchi two-equation fixed point (tau, p). This is the 802.11-style
        contention loss that pure SINR does not capture.

  3. Log-normal shadowing
        A zero-mean Gaussian (dB) perturbation on received power. Breaks the
        pure distance->quality determinism of v1, so PER and link_quality
        carry information beyond distance alone (fixes the Approach-1 finding
        that candidate_packet_error_rate was informationally inert).

BACKWARD COMPATIBILITY (required for the interference on/off ablation):
  With interference_mw=0, n_contenders<=1, and shadowing_db=0,
  compute_link_features_v2() reduces EXACTLY to
  link_model.compute_link_features(). Verified by assertion in
  preflight_interference_check.py.

References:
  Bianchi, "Performance Analysis of the IEEE 802.11 DCF", IEEE JSAC 2000.
"""

import numpy as np

# ── Radio parameters (identical to link_model.py for drop-in compatibility) ──
TX_POWER_DBM        = 20.0     # transmit power
NOISE_FLOOR_DBM     = -95.0    # receiver thermal noise floor
FREQ_GHZ            = 2.4      # 2.4 GHz ISM band
PATH_LOSS_EXPONENT  = 2.0      # free-space (UAVs fly above obstacles)
REF_DISTANCE        = 1.0      # reference distance (m)
RSSI_SENSITIVITY    = -85.0    # link exists if rssi > this threshold
PACKET_BITS         = 1024 * 8 # 1024-byte packet

# ── Interference / channel parameters (new in v2) ────────────────────────────
INTERFERENCE_RANGE_MULT = 2.0  # interferers reach 2x comm range (weak but real)
SHADOWING_SIGMA_DB      = 5.0  # log-normal shadowing std-dev (dB); 0 disables
CARRIER_SENSE_MULT      = 1.0  # carrier-sense range = this x comm_range

# ── Bianchi CSMA/CA parameters (IEEE 802.11-style DCF) ───────────────────────
CW_MIN         = 16    # minimum contention window (W)
BACKOFF_STAGES = 6     # m, so CW_MAX = CW_MIN * 2^m = 1024

_WAVELENGTH = 3e8 / (FREQ_GHZ * 1e9)
_PL_D0 = 20.0 * np.log10(4.0 * np.pi * REF_DISTANCE / _WAVELENGTH)
_NOISE_MW = 10.0 ** (NOISE_FLOOR_DBM / 10.0)


# ── Unit helpers ─────────────────────────────────────────────────────────────

def dbm_to_mw(dbm):
    return 10.0 ** (np.asarray(dbm, dtype=float) / 10.0)


def mw_to_dbm(mw):
    return 10.0 * np.log10(np.maximum(mw, 1e-30))


def path_loss_db(distance_m):
    d = max(float(distance_m), 0.1)
    return _PL_D0 + 10.0 * PATH_LOSS_EXPONENT * np.log10(d / REF_DISTANCE)


def rx_power_dbm(distance_m, shadowing_db=0.0):
    """Received power (dBm) at a given distance, optional shadowing offset."""
    return TX_POWER_DBM - path_loss_db(distance_m) + shadowing_db


def rx_power_mw(distance_m, shadowing_db=0.0):
    """Received power in mW — used to sum interferer contributions linearly."""
    return dbm_to_mw(rx_power_dbm(distance_m, shadowing_db))


# ── Bianchi CSMA/CA collision model ──────────────────────────────────────────

def bianchi_tau_p(n_contenders, W=CW_MIN, m=BACKOFF_STAGES, iters=200, tol=1e-10):
    """Solve the Bianchi (2000) fixed point for (tau, p):
        tau = 2(1-2p) / [ (1-2p)(W+1) + pW(1-(2p)^m) ]
        p   = 1 - (1-tau)^(n-1)
    tau = stationary per-slot transmission probability of a station,
    p   = conditional collision probability seen by a transmitting station.
    Returns (tau, p). For n<=1 there is no contention -> (0, 0)."""
    n = int(n_contenders)
    if n <= 1:
        return 0.0, 0.0
    p = 0.0
    for _ in range(iters):
        two_p = 2.0 * p
        # guard (2p)^m against overflow when p -> 0.5+
        pow_term = (two_p ** m) if two_p < 1.0 else 1.0
        den = (1.0 - two_p) * (W + 1) + p * W * (1.0 - pow_term)
        if abs(den) < 1e-15:
            tau = 1.0
        else:
            tau = 2.0 * (1.0 - two_p) / den
        tau = min(max(tau, 0.0), 1.0)
        p_new = 1.0 - (1.0 - tau) ** (n - 1)
        if abs(p_new - p) < tol:
            p = p_new
            break
        p = p_new
    return float(tau), float(p)


def bianchi_collision_prob(n_contenders):
    """MAC-layer collision probability for a station contending against
    (n_contenders - 1) others in carrier-sense range."""
    _, p = bianchi_tau_p(n_contenders)
    return p


# ── Core link-feature computation ────────────────────────────────────────────

def compute_link_features_v2(distance_m, interference_mw=0.0,
                             n_contenders=1, shadowing_db=0.0):
    """
    Interference-aware link features. Drop-in extension of
    link_model.compute_link_features().

    Args:
        distance_m      : 3D distance between tx and rx (m)
        interference_mw : summed received power (mW) of all concurrent
                          interferers at the receiver (0 => interference-free)
        n_contenders    : number of stations contending in carrier-sense range
                          (>=1; 1 => no MAC contention)
        shadowing_db    : log-normal shadowing sample (dB) for this link
                          (0 => deterministic path loss, matches v1)

    Returns (rssi, sinr, link_quality, per):
        rssi         : received signal strength (dBm), incl. shadowing
        sinr         : signal-to-interference-plus-noise ratio (dB)
        link_quality : normalized [0, 1] from SINR
        per          : total packet error rate = 1 - (1-per_phy)(1-p_collision)
    """
    d = max(float(distance_m), 0.1)
    rssi = TX_POWER_DBM - path_loss_db(d) + shadowing_db
    signal_mw = dbm_to_mw(rssi)

    sinr_linear = signal_mw / (_NOISE_MW + max(interference_mw, 0.0))
    sinr_db = 10.0 * np.log10(max(sinr_linear, 1e-30))

    link_quality = float(np.clip(sinr_db / 30.0, 0.0, 1.0))

    # Physical-layer error from SINR (same BER/PER form as v1, SINR replaces SNR)
    ber = 0.5 * np.exp(-sinr_linear / 2.0)
    per_phy = 1.0 - (1.0 - ber) ** PACKET_BITS
    per_phy = float(np.clip(per_phy, 0.0, 1.0))

    # MAC-layer collision loss (Bianchi). n_contenders<=1 => 0.
    p_coll = bianchi_collision_prob(n_contenders)

    per_total = 1.0 - (1.0 - per_phy) * (1.0 - p_coll)
    per_total = float(np.clip(per_total, 0.0, 1.0))

    return float(rssi), float(sinr_db), link_quality, per_total


def compute_link_features_v2_verbose(distance_m, interference_mw=0.0,
                                     n_contenders=1, shadowing_db=0.0):
    """Same as compute_link_features_v2 but also returns the decomposition
    (per_phy, p_collision, sinr_linear) for diagnostics / the preflight."""
    d = max(float(distance_m), 0.1)
    rssi = TX_POWER_DBM - path_loss_db(d) + shadowing_db
    signal_mw = dbm_to_mw(rssi)
    sinr_linear = signal_mw / (_NOISE_MW + max(interference_mw, 0.0))
    sinr_db = 10.0 * np.log10(max(sinr_linear, 1e-30))
    link_quality = float(np.clip(sinr_db / 30.0, 0.0, 1.0))
    ber = 0.5 * np.exp(-sinr_linear / 2.0)
    per_phy = float(np.clip(1.0 - (1.0 - ber) ** PACKET_BITS, 0.0, 1.0))
    p_coll = bianchi_collision_prob(n_contenders)
    per_total = float(np.clip(1.0 - (1.0 - per_phy) * (1.0 - p_coll), 0.0, 1.0))
    return {
        'rssi': float(rssi), 'sinr_db': float(sinr_db),
        'link_quality': link_quality, 'per': per_total,
        'per_phy': per_phy, 'p_collision': float(p_coll),
        'sinr_linear': float(sinr_linear),
    }


# ── Interference aggregation (called by simulator_v2 and the preflight) ──────

def total_interference_mw(rx_pos, interferer_positions, shadowing_db=None):
    """Sum received power (mW) at rx_pos from a list of concurrent interferer
    positions. Caller is responsible for passing only interferers within
    interference range (see interferers_in_range)."""
    if len(interferer_positions) == 0:
        return 0.0
    total = 0.0
    for k, tx_pos in enumerate(interferer_positions):
        d = float(np.linalg.norm(np.asarray(rx_pos) - np.asarray(tx_pos)))
        sh = 0.0 if shadowing_db is None else float(shadowing_db[k])
        total += rx_power_mw(d, sh)
    return float(total)


def interferers_in_range(rx_pos, candidate_positions, comm_range):
    """Return indices of candidate transmitters within INTERFERENCE_RANGE of
    the receiver. Interference range = INTERFERENCE_RANGE_MULT x comm_range."""
    r = INTERFERENCE_RANGE_MULT * comm_range
    rx = np.asarray(rx_pos)
    idx = []
    for k, p in enumerate(candidate_positions):
        if float(np.linalg.norm(rx - np.asarray(p))) <= r:
            idx.append(k)
    return idx


# ── Unchanged geometry helpers (identical to v1) ─────────────────────────────

def link_exists(distance_m, comm_range):
    """A link exists if within comm_range AND deterministic rssi > sensitivity.
    (Uses shadowing_db=0; shadowing affects quality/PER, not existence, to keep
    topology construction stable — same convention as v1.)"""
    if distance_m > comm_range:
        return False
    rssi = rx_power_dbm(distance_m, shadowing_db=0.0)
    return rssi > RSSI_SENSITIVITY


def estimate_link_lifetime(pos_i, pos_j, vel_i, vel_j, comm_range,
                           max_lifetime=60.0):
    """Seconds until the link breaks, from relative position/velocity.
    Identical to link_model.estimate_link_lifetime (geometry only)."""
    rel_pos = pos_j - pos_i
    rel_vel = vel_j - vel_i
    dist = np.linalg.norm(rel_pos)
    if dist < 1e-6:
        return max_lifetime
    unit_pos = rel_pos / dist
    radial_speed = float(np.dot(rel_vel, unit_pos))
    if radial_speed <= 0:
        return max_lifetime
    remaining = comm_range - dist
    if remaining <= 0:
        return 0.0
    return float(min(remaining / radial_speed, max_lifetime))
