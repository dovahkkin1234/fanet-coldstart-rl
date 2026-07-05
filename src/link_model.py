"""
link_model.py
Wireless link feature computation using a log-distance path loss model.

Produces RSSI, SNR, link quality, packet error rate, and estimated link lifetime.
"""

import numpy as np

# Radio parameters (held constant across all scenarios)
TX_POWER_DBM = 20.0        # transmit power
NOISE_FLOOR_DBM = -95.0    # receiver noise floor
FREQ_GHZ = 2.4             # 2.4 GHz ISM band
PATH_LOSS_EXPONENT = 2.0   # free-space (UAVs fly above obstacles)
REF_DISTANCE = 1.0         # reference distance (m)
RSSI_SENSITIVITY = -85.0   # link exists if rssi > this threshold
PACKET_BITS = 1024 * 8     # 1024-byte packet

_WAVELENGTH = 3e8 / (FREQ_GHZ * 1e9)
_PL_D0 = 20.0 * np.log10(4.0 * np.pi * REF_DISTANCE / _WAVELENGTH)


def compute_link_features(distance_m):
    """
    Given a 3D distance in meters, return (rssi, snr, link_quality, per).

    rssi          : received signal strength in dBm
    snr           : signal-to-noise ratio in dB
    link_quality  : normalized [0, 1]
    per           : packet error rate [0, 1]
    """
    d = max(float(distance_m), 0.1)

    path_loss = _PL_D0 + 10.0 * PATH_LOSS_EXPONENT * np.log10(d / REF_DISTANCE)
    rssi = TX_POWER_DBM - path_loss
    snr = rssi - NOISE_FLOOR_DBM

    link_quality = float(np.clip((snr - 0.0) / (30.0 - 0.0), 0.0, 1.0))

    snr_linear = 10.0 ** (snr / 10.0)
    ber = 0.5 * np.exp(-snr_linear / 2.0)
    per = 1.0 - (1.0 - ber) ** PACKET_BITS
    per = float(np.clip(per, 0.0, 1.0))

    return float(rssi), float(snr), link_quality, per


def link_exists(distance_m, comm_range):
    """A link exists if within comm_range AND signal is above sensitivity."""
    if distance_m > comm_range:
        return False
    rssi, _, _, _ = compute_link_features(distance_m)
    return rssi > RSSI_SENSITIVITY


def estimate_link_lifetime(pos_i, pos_j, vel_i, vel_j, comm_range,
                           max_lifetime=60.0):
    """
    Estimate seconds until the link breaks, from relative position/velocity.
    Returns max_lifetime if nodes are closing or holding distance.
    """
    rel_pos = pos_j - pos_i
    rel_vel = vel_j - vel_i

    dist = np.linalg.norm(rel_pos)
    if dist < 1e-6:
        return max_lifetime

    unit_pos = rel_pos / dist
    radial_speed = float(np.dot(rel_vel, unit_pos))  # >0 means moving apart

    if radial_speed <= 0:
        return max_lifetime

    remaining = comm_range - dist
    if remaining <= 0:
        return 0.0

    return float(min(remaining / radial_speed, max_lifetime))
