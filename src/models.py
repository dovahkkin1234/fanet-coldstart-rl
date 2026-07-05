"""
models.py
Energy model and packet queue model for each drone.
"""

from collections import deque


class EnergyModel:
    """Communication energy model (radio only, not propulsion)."""

    TX_COST = 0.02     # energy per transmitted packet
    RX_COST = 0.01     # energy per received packet
    IDLE_COST = 0.001  # energy per second idle

    @staticmethod
    def consume_tx(energy):
        return max(0.0, energy - EnergyModel.TX_COST)

    @staticmethod
    def consume_rx(energy):
        return max(0.0, energy - EnergyModel.RX_COST)

    @staticmethod
    def consume_idle(energy, dt):
        return max(0.0, energy - EnergyModel.IDLE_COST * dt)

    @staticmethod
    def is_alive(energy):
        return energy > 0.01


class NodeQueue:
    """Bounded FIFO packet buffer for a drone."""

    def __init__(self, max_size=50):
        self.buffer = deque()
        self.max_size = max_size
        self.dropped_overflow = 0

    def enqueue(self, packet):
        if len(self.buffer) >= self.max_size:
            self.dropped_overflow += 1
            return False
        self.buffer.append(packet)
        return True

    def dequeue(self):
        return self.buffer.popleft() if self.buffer else None

    @property
    def occupancy(self):
        return len(self.buffer) / self.max_size

    @property
    def length(self):
        return len(self.buffer)
