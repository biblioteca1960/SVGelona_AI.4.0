"""
AngularDefectDynamics.py
------------------------
Dynamics of Angular Defects and Vector Transformations (abc → t3-) 
for SVGelona_IA 2.0.

This module models:
- Conversion of central vector abc into spiral trajectories (t3-)
- Angular defect evolution between rotated icosahedra
- Integration with ChronNet for temporal coherence
"""

import numpy as np
from dataclasses import dataclass
from typing import List
from geometry import Icosahedron, CriticalPoint, AngularDefect, CentralVector
from chronnet import ChronNet, ChronNode

# -----------------------------
# Spiral Transformation
# -----------------------------
@dataclass
class SpiralVector:
    """
    Represents the vector after passing the critical line and becoming t3-.
    """
    vector: np.ndarray
    origin: CentralVector
    z0: CriticalPoint
    step_size: float = 0.01

    def evolve_step(self):
        """
        Perform one step of the spiral evolution.
        """
        # Rotate slightly around z-axis (simplified model)
        angle = self.step_size
        rot_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1]
        ])
        self.vector = rot_matrix @ self.vector
        return self.vector

# -----------------------------
# Angular Defect Dynamics
# -----------------------------
class AngularDefectDynamics:
    """
    Handles the dynamics of angular defects between two rotated icosahedra.
    """
    def __init__(self, icosa_a: Icosahedron, icosa_b: Icosahedron, z0: CriticalPoint):
        self.defect = AngularDefect(icosa_a, icosa_b, z0)
        self.spiral_vectors: List[SpiralVector] = []

    def initialize_spirals(self, central_vector: CentralVector):
        """
        Convert central vector abc into initial t3- vectors along defect direction.
        """
        direction = self.defect.defect_direction()
        initial_vec = central_vector.normalized() + direction
        spiral = SpiralVector(vector=initial_vec, origin=central_vector, z0=self.defect.z0)
        self.spiral_vectors.append(spiral)

    def step(self):
        """
        Evolve all spiral vectors by one step.
        """
        for spiral in self.spiral_vectors:
            spiral.evolve_step()

    def integrate_with_chronnet(self, chronnet: ChronNet, timestamp: float):
        """
        Insert current spiral vectors as nodes in the temporal network.
        """
        for spiral in self.spiral_vectors:
            node = ChronNode(vector=spiral.vector.copy(), timestamp=timestamp)
            chronnet.root_nodes.append(node)

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    from geometry import Icosahedron, CriticalPoint, CentralVector

    # Create two rotated icosahedra
    ico1 = Icosahedron()
    ico2 = ico1.rotate(np.eye(3))  # Identity rotation for example

    # Critical point
    z0 = CriticalPoint(t=0.5)

    # Central vector
    abc = CentralVector(direction=np.array([1.0, 0.0, 0.0]))

    # Initialize dynamics
    dynamics = AngularDefectDynamics(ico1, ico2, z0)
    dynamics.initialize_spirals(abc)

    # Temporal network
    cn = ChronNet()
    
    # Evolve spirals and integrate
    for t in np.linspace(0, 1, 10):
        dynamics.step()
        dynamics.integrate_with_chronnet(cn, timestamp=t)

    print(f"ChronNet now has {len(cn.root_nodes)} spiral nodes.")
