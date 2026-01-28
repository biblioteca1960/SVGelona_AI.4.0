"""
search_engine.py
----------------
Module for geometric search and preconscious thought exploration in SVGelona_IA.2.0.

- Provides tools to explore trajectories from icosahedral vertices.
- Supports semantic search based on emergent symbolic language.
- Interfaces with ChronNet and geometric structures.
"""

import numpy as np

# -----------------------------
# Trajectory Class
# -----------------------------
class Trajectory:
    """
    Represents a geometric trajectory starting from a vertex or point.
    """
    def __init__(self, start_vertex, angle, length=10):
        self.start_vertex = np.array(start_vertex)
        self.angle = angle
        self.length = length
        self.path = self._generate_path()

    def _generate_path(self):
        """Generate a simple 3D trajectory based on start point and angle."""
        path = [self.start_vertex]
        direction = np.array([
            np.cos(self.angle), 
            np.sin(self.angle), 
            np.sin(self.angle/2)
        ])
        direction /= np.linalg.norm(direction)
        for i in range(1, self.length):
            path.append(path[-1] + direction)
        return np.array(path)

# -----------------------------
# Geometric Search Engine
# -----------------------------
class GeometricSearchEngine:
    """
    Engine to search geometric trajectories and explore preconscious thoughts.
    """
    def __init__(self, depth=5):
        self.depth = depth
        self.trajectories = []

    def search_by_trajectory(self, start_vertex, angle):
        """Search trajectories starting from a given vertex and angle."""
        traj = Trajectory(start_vertex, angle, length=self.depth)
        self.trajectories.append(traj)
        return [traj]

    def semantic_search(self, concept):
        """
        Returns trajectories associated with a semantic concept.
        For demo purposes, returns all trajectories containing positive z.
        """
        results = []
        for traj in self.trajectories:
            if np.any(traj.path[:,2] > 0):
                results.append(traj)
        return results

    def thought_exploration(self, trajectory, max_depth=None):
        """
        Simulate preconscious exploration along a trajectory.
        Returns list of thought nodes with coherence values.
        """
        max_depth = max_depth or self.depth
        thoughts = []
        for i, point in enumerate(trajectory.path[:max_depth]):
            coherence = np.exp(-i/self.depth)  # exponential decay as placeholder
            thoughts.append(ThoughtNode(point, coherence))
        return thoughts

# -----------------------------
# Thought Node
# -----------------------------
class ThoughtNode:
    """
    Represents a node in preconscious thought space.
    """
    def __init__(self, position, coherence):
        self.position = position
        self.coherence = coherence

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    engine = GeometricSearchEngine(depth=5)
    start = [0.0, 0.0, 0.0]

    # Search trajectory
    results = engine.search_by_trajectory(start, np.pi/4)
    print("Found trajectories:", len(results))

    # Semantic search
    sem_results = engine.semantic_search("Quantum")
    print("Semantic trajectories:", len(sem_results))

    # Thought exploration
    thoughts = engine.thought_exploration(results[0], max_depth=3)
    for i, t in enumerate(thoughts):
        print(f"Thought {i}: position={t.position}, coherence={t.coherence:.3f}")
