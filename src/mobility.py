"""
mobility.py
Random Waypoint 3D mobility model for FANET simulation.

Each drone moves independently. No coordination between drones.
"""

import numpy as np


class DroneRWP:
    """A single drone following the Random Waypoint 3D mobility model."""

    def __init__(self, drone_id, area_x, area_y, z_min, z_max,
                 speed_min, speed_max, pause_max, seed=0):
        # Each drone gets a unique, reproducible random stream
        self.rng = np.random.default_rng(seed * 10000 + drone_id)
        self.id = drone_id

        self.area_x = area_x
        self.area_y = area_y
        self.z_min = z_min
        self.z_max = z_max
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.pause_max = pause_max

        # Start at a random position inside the area
        self.x = self.rng.uniform(0, area_x)
        self.y = self.rng.uniform(0, area_y)
        self.z = self.rng.uniform(z_min, z_max)

        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

        self.pause_remaining = 0.0
        self._pick_new_waypoint()

    def _pick_new_waypoint(self):
        """Pick a new random destination and travel speed."""
        self.dest_x = self.rng.uniform(0, self.area_x)
        self.dest_y = self.rng.uniform(0, self.area_y)
        self.dest_z = self.rng.uniform(self.z_min, self.z_max)
        self.current_speed = self.rng.uniform(self.speed_min, self.speed_max)

    def step(self, dt):
        """Advance the drone one timestep of length dt seconds."""
        # If paused (hovering at a waypoint), count down and stay still
        if self.pause_remaining > 0:
            self.pause_remaining -= dt
            self.vx = self.vy = self.vz = 0.0
            return

        dx = self.dest_x - self.x
        dy = self.dest_y - self.y
        dz = self.dest_z - self.z
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)

        # Arrived at destination -> pause, then pick a new waypoint
        if dist < 1.0:
            self.pause_remaining = self.rng.uniform(0, self.pause_max)
            self.vx = self.vy = self.vz = 0.0
            self._pick_new_waypoint()
            return

        # Limit vertical motion: vz capped to 30% of total motion
        vertical_fraction = min(abs(dz) / max(dist, 1e-6), 0.3)
        scale = self.current_speed / dist

        self.vx = dx * scale * (1.0 - vertical_fraction)
        self.vy = dy * scale * (1.0 - vertical_fraction)
        self.vz = dz * scale * vertical_fraction

        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        # Clamp to boundaries; if we hit one, pick a fresh waypoint
        hit_boundary = False
        if self.x < 0 or self.x > self.area_x:
            self.x = float(np.clip(self.x, 0, self.area_x))
            hit_boundary = True
        if self.y < 0 or self.y > self.area_y:
            self.y = float(np.clip(self.y, 0, self.area_y))
            hit_boundary = True
        if self.z < self.z_min or self.z > self.z_max:
            self.z = float(np.clip(self.z, self.z_min, self.z_max))
            hit_boundary = True
        if hit_boundary:
            self._pick_new_waypoint()

    @property
    def pos(self):
        return np.array([self.x, self.y, self.z])

    @property
    def vel(self):
        return np.array([self.vx, self.vy, self.vz])

    @property
    def speed(self):
        return float(np.linalg.norm(self.vel))

    @property
    def heading(self):
        return float(np.arctan2(self.vy, self.vx))

    @property
    def pitch(self):
        horiz = np.sqrt(self.vx ** 2 + self.vy ** 2)
        return float(np.arctan2(self.vz, max(horiz, 1e-9)))
