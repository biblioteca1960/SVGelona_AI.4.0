"""
geometry.py
------------
Fundamental geometry for SVGelona_IA 2.0

This module defines:
- Icosahedron geometry
- Critical line and critical points (z0)
- Angular defect between rotated icosahedra
- Central vectors and basic geometric utilities

All is rigorous geometric mathematics with optional physical interpretation.
"""

import numpy as np
from dataclasses import dataclass

# -----------------------------
# Basic Utilities
# -----------------------------
def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

# -----------------------------
# Critical Point z0
# -----------------------------
@dataclass(frozen=True)
class CriticalPoint:
    """
    Shared critical point z0 = 0.5 + i*t
    Geometrically represented as a point connecting icosahedral domains.
    """
    t: float

    @property
    def complex(self) -> complex:
        return 0.5 + 1j * self.t

# -----------------------------
# Critical Line
# -----------------------------
class CriticalLine:
    """
    Critical line Re(s) = 0.5
    Acts as a topological boundary in the geometric model.
    """
    @staticmethod
    def contains(z: complex, tol: float = 1e-9) -> bool:
        return abs(z.real - 0.5) < tol

# -----------------------------
# Icosahedron
# -----------------------------
class Icosahedron:
    """
    Regular icosahedron oriented in 3D space.
    Orientation defines its relationship with z0 and other icosahedra.
    """

    def __init__(self, rotation_matrix: np.ndarray | None = None):
        self.rotation = rotation_matrix if rotation_matrix is not None else np.eye(3)
        self.vertices = self._generate_vertices()

    def _generate_vertices(self) -> np.ndarray:
        """
        Generate the 12 vertices of a unit icosahedron.
        """
        phi = (1 + np.sqrt(5)) / 2
        verts = []
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                verts.append([0, s1, s2 * phi])
                verts.append([s1, s2 * phi, 0])
                verts.append([s1 * phi, 0, s2])
        verts = np.array(verts, dtype=float)
        verts = np.array([normalize(v) for v in verts])
        return (self.rotation @ verts.T).T

    def rotate(self, R: np.ndarray) -> "Icosahedron":
        """Return a new rotated icosahedron."""
        return Icosahedron(rotation_matrix=R @ self.rotation)

# -----------------------------
# Angular Defect
# -----------------------------
@dataclass
class AngularDefect:
    """
    Angular defect between two rotated icosahedra sharing z0.

    Geometrically:
    - Space between non-coincident faces
    - Channel through which central vector abc -> t3- passes
    """
    icosa_a: Icosahedron
    icosa_b: Icosahedron
    z0: CriticalPoint

    def defect_measure(self) -> float:
        """
        Simple measure of angular defect based on the mean
        misalignment between corresponding vertices.
        """
        dots = []
        for va, vb in zip(self.icosa_a.vertices, self.icosa_b.vertices):
            dots.append(np.clip(np.dot(va, vb), -1.0, 1.0))
        angles = [np.arccos(d) for d in dots]
        return float(np.mean(angles))

    def defect_direction(self) -> np.ndarray:
        """
        Mean direction of the angular defect.
        This is the geometric direction of the channel.
        """
        diff = self.icosa_b.vertices - self.icosa_a.vertices
        return normalize(np.mean(diff, axis=0))

# -----------------------------
# Central Vector abc
# -----------------------------
@dataclass
class CentralVector:
    """
    Central vector abc originating from the icosahedron center
    pointing towards z0.
    """
    direction: np.ndarray

    def normalized(self) -> np.ndarray:
        return normalize(self.direction)

# -----------------------------
# Visualization helper (optional)
# -----------------------------
def plot_icosahedron(ico: Icosahedron, z0: CriticalPoint = None):
    """
    Simple 3D plot of icosahedron and optional critical point.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    verts = ico.vertices
    ax.scatter(verts[:,0], verts[:,1], verts[:,2], color='blue')
    for i, v in enumerate(verts):
        ax.text(v[0], v[1], v[2], f"{i}", color='red')
    if z0 is not None:
        ax.scatter([0.0], [0.0], [z0.t], color='green', s=100, label="z0")
    ax.set_box_aspect([1,1,1])
    plt.show()
