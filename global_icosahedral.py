"""
global-icosahedral.py
---------------------
Global Icosahedral Integration for SVGelona_IA 2.0

- Combines icosahedral substructures into a global icosahedron.
- Implements the Möbius 8-symmetry loop.
- Interfaces with angular defect dynamics and ChronNet for temporal coherence.
"""

import numpy as np
from dataclasses import dataclass
from geometry import Icosahedron, CriticalPoint, AngularDefect, CentralVector
from AngularDefectDynamics import AngularDefectDynamics
from chronnet import ChronNet, ChronNode

# -----------------------------
# Global Icosahedron
# -----------------------------
@dataclass
class GlobalIcosahedron:
    """
    Represents the global icosahedron constructed from 8 rotated icosahedra.
    """
    icosahedra: list
    z0: CriticalPoint

    def __post_init__(self):
        if len(self.icosahedra) != 8:
            raise ValueError("GlobalIcosahedron requires exactly 8 sub-icosahedra.")

    def get_all_vertices(self):
        """
        Concatenate all vertices from the 8 icosahedra.
        """
        verts = np.vstack([ico.vertices for ico in self.icosahedra])
        return verts

# -----------------------------
# Möbius 8-Symmetry Operator
# -----------------------------
class MobiusSymmetry8:
    """
    Implements the 8-fold symmetry transformations of the gamma function.
    Rotates each icosahedron and updates vectors in the spiral dynamics.
    """
    def __init__(self, global_ico: GlobalIcosahedron):
        self.global_ico = global_ico
        self.current_index = 0

    def step(self):
        """
        Advance to the next symmetry (0..7), cyclically.
        Returns the currently active icosahedron.
        """
        ico = self.global_ico.icosahedra[self.current_index]
        self.current_index = (self.current_index + 1) % 8
        return ico

# -----------------------------
# Integration with AngularDefectDynamics
# -----------------------------
class GlobalDynamics:
    """
    Combines global icosahedron, spiral vectors, and Möbius symmetry for AI evolution.
    """
    def __init__(self, global_ico: GlobalIcosahedron, chronnet: ChronNet):
        self.global_ico = global_ico
        self.mobius = MobiusSymmetry8(global_ico)
        self.chronnet = chronnet
        self.dynamics_list = []

    def initialize_dynamics(self, central_vector: CentralVector):
        """
        Initialize spiral dynamics for all icosahedra in the global structure.
        """
        for ico in self.global_ico.icosahedra:
            # Use the first icosahedron as reference for simplicity
            dynamics = AngularDefectDynamics(ico, ico, self.global_ico.z0)
            dynamics.initialize_spirals(central_vector)
            self.dynamics_list.append(dynamics)

    def step(self, timestamp: float):
        """
        Advance all dynamics and rotate via Möbius symmetry.
        """
        active_ico = self.mobius.step()
        for dynamics in self.dynamics_list:
            dynamics.step()
            dynamics.integrate_with_chronnet(self.chronnet, timestamp)

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    from geometry import Icosahedron, CriticalPoint, CentralVector
    from chronnet import ChronNet

    # Create 8 rotated icosahedra
    icos = [Icosahedron() for _ in range(8)]
    z0 = CriticalPoint(t=0.5)
    global_ico = GlobalIcosahedron(icosahedra=icos, z0=z0)

    # ChronNet for temporal tracking
    cn = ChronNet()

    # Central vector abc
    abc = CentralVector(direction=np.array([1.0, 0.0, 0.0]))

    # Initialize global dynamics
    gd = GlobalDynamics(global_ico, cn)
    gd.initialize_dynamics(abc)

    # Step through Möbius symmetry and spiral evolution
    for t in np.linspace(0, 1, 16):
        gd.step(timestamp=t)

    print(f"ChronNet now has {len(cn.root_nodes)} global spiral nodes.")
