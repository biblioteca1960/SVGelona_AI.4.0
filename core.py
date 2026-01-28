"""
core.py
-------
Fundamental Tetrahedron, Critical Trajectories, ChronNet, Holonetic System, Mobius 8-Symmetry Loop
"""

import numpy as np
from geometry import Icosahedron, CriticalPoint, CriticalLine, AngularDefect
from chronnet import ChronNet
from tesseract_module import FixedTesseract, FloatingTesseract, regulate
from mobius_symmetry import apply_mobius_8sym

# ─────────────────────────────────────────────────────────────
# Fundamental Tetrahedron
# ─────────────────────────────────────────────────────────────
class FundamentalTetrahedron:
    """
    Defines the fundamental tetrahedron for all critical trajectories.
    """
    def __init__(self, vertices=None):
        # Default: unit tetrahedron in 3D space
        if vertices is None:
            self.vertices = np.array([
                [1, 1, 1],
                [-1, -1, 1],
                [-1, 1, -1],
                [1, -1, -1]
            ], dtype=float)
        else:
            self.vertices = np.array(vertices, dtype=float)
    
    def contains_point(self, point: np.ndarray) -> bool:
        """
        Checks if a point is inside the tetrahedron (approximate convex hull test)
        """
        # Using barycentric coordinates
        A = self.vertices
        v0 = A[1] - A[0]
        v1 = A[2] - A[0]
        v2 = A[3] - A[0]
        vp = point - A[0]
        mat = np.column_stack((v0, v1, v2))
        try:
            coeffs = np.linalg.solve(mat, vp)
            return np.all(coeffs >= 0) and np.sum(coeffs) <= 1
        except np.linalg.LinAlgError:
            return False

# ─────────────────────────────────────────────────────────────
# Core AI System
# ─────────────────────────────────────────────────────────────
class HoloneticSystem:
    """
    Integrates all modules:
    - Critical Tetrahedron
    - ChronNet temporal network
    - Mobius 8-symmetry rotations
    - Tesseract regulation
    """
    def __init__(self):
        # Initialize fundamental geometry
        self.tetrahedron = FundamentalTetrahedron()
        
        # Initialize ChronNet
        self.chronnet = ChronNet()
        
        # Initialize fixed and floating tesseracts
        self.F = [FixedTesseract(i) for i in range(8)]
        self.G = [FloatingTesseract(j) for j in range(8)]
        regulate(self.G, self.F)
        
        # Initialize critical points list
        self.critical_points = []
        
    # ─────────────── Trajectories ───────────────
    def add_critical_point(self, t: float):
        """Add a critical point along the line Re(s)=1/2"""
        z0 = CriticalPoint(t)
        self.critical_points.append(z0)
        return z0
    
    def generate_trajectory(self, icosa: Icosahedron, angle: float):
        """
        Generates a fractal trajectory from an icosahedron vertex along a critical line
        """
        traj = []
        for cp in self.critical_points:
            # Simple mapping: rotate vertex towards z0
            for v in icosa.vertices:
                direction = cp.complex.real * np.array([1,0,0]) + cp.complex.imag * np.array([0,1,0])
                traj.append(v + 0.1 * direction)
        return np.array(traj)
    
    # ─────────────── Mobius 8-Symmetry ───────────────
    def apply_mobius_loop(self):
        """
        Applies the 8-symmetry Möbius loop to all icosahedra in the system
        """
        for i in range(len(self.G)):
            self.G[i].vector = apply_mobius_8sym(self.G[i].vector)
    
    # ─────────────── Update / Self-Regulation ───────────────
    def update(self, new_input: np.ndarray):
        """
        Adds a new floating tesseract, regulates, applies Mobius loop, updates ChronNet
        """
        new_G = FloatingTesseract(len(self.G))
        new_G.vector = new_input / np.linalg.norm(new_input)
        self.G.append(new_G)
        
        # Self-regulation
        regulate(self.G, self.F)
        
        # Mobius 8-symmetry rotations
        self.apply_mobius_loop()
        
        # Update ChronNet with new trajectory
        self.chronnet.update_network(self.G)
        
        return new_G.vector
