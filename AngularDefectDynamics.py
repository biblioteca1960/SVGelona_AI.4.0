# AngularDefectDynamics.py
import numpy as np
from geometry import Icosahedron, CriticalPoint, CentralVector

# ----------------------------
# SpiralVector definit al nivell del mòdul
# ----------------------------
class SpiralVector:
    """Vector que evoluciona en una trajectòria espiral"""
    def __init__(self, vector, origin=None, z0=None, step_size=0.01):
        self.vector = np.array(vector, dtype=float)
        self.origin = origin
        self.z0 = z0
        self.step_size = step_size

    def evolve_step(self):
        """Evolució simple: rotació al voltant de Z"""
        theta = self.step_size
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0],
                      [s,  c, 0],
                      [0,  0, 1]])
        self.vector = R @ self.vector

# ----------------------------
# Classe principal AngularDefectDynamics
# ----------------------------
class AngularDefectDynamics:
    """Dinàmica de defectes angulars sobre icosaedres"""
    def __init__(self, ico1: Icosahedron, ico2: Icosahedron, z0: CriticalPoint):
        self.ico1 = ico1
        self.ico2 = ico2
        self.z0 = z0
        self.spiral_vectors = []

    def initialize_spirals(self, central_vector: CentralVector):
        """Inicialitza vectors espirals a partir de central_vector"""
        for v in self.ico1.vertices:
            spiral = SpiralVector(vector=v.copy(), origin=None, z0=self.z0, step_size=0.01)
            self.spiral_vectors.append(spiral)

    def step(self):
        """Executa un pas d’evolució sobre tots els vectors"""
        for spiral in self.spiral_vectors:
            spiral.evolve_step()

# ----------------------------
# Export explícit per import directe
# ----------------------------
__all__ = ["AngularDefectDynamics", "SpiralVector"]
