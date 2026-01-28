"""
chronnet.py
------------
Temporal Geometric Network (ChronNet) for SVGelona_IA 2.0

This module encodes temporal depth and relationships between trajectories.
It allows navigation using coherence-based predicates to explore
"preconscious thought" paths in the AI model.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional

# -----------------------------
# Node in the ChronNet
# -----------------------------
@dataclass
class ChronNode:
    """
    Node representing a state in the temporal network.
    """
    vector: np.ndarray
    timestamp: float
    children: List["ChronNode"] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

# -----------------------------
# ChronNet structure
# -----------------------------
class ChronNet:
    """
    Fractal temporal network storing trajectories as nodes.
    """
    def __init__(self):
        self.root_nodes: List[ChronNode] = []

    def add_trajectory(self, vectors: List[np.ndarray], timestamps: List[float]):
        """
        Add a trajectory as a sequence of nodes.
        """
        if len(vectors) != len(timestamps):
            raise ValueError("Vectors and timestamps must have the same length")
        prev_node: Optional[ChronNode] = None
        for vec, t in zip(vectors, timestamps):
            node = ChronNode(vector=vec, timestamp=t)
            if prev_node is not None:
                prev_node.children.append(node)
            else:
                self.root_nodes.append(node)
            prev_node = node

    def traverse(self, max_depth: int = 5):
        """
        Generator to traverse all nodes up to a given depth.
        """
        def _traverse(node: ChronNode, depth: int):
            if depth > max_depth:
                return
            yield node
            for child in node.children:
                yield from _traverse(child, depth + 1)

        for root in self.root_nodes:
            yield from _traverse(root, 1)

    def search_by_coherence(self, target_vector: np.ndarray, threshold: float = 0.95):
        """
        Find nodes whose vector has high coherence (cosine similarity) with target_vector.
        """
        target_norm = target_vector / np.linalg.norm(target_vector)
        results = []
        for node in self.traverse():
            vec_norm = node.vector / np.linalg.norm(node.vector)
            similarity = np.dot(vec_norm, target_norm)
            if similarity >= threshold:
                results.append(node)
        return results

    def explore_paths(self, start_node: ChronNode, max_depth: int = 3):
        """
        Explore sequences of nodes starting from a node, up to max_depth.
        Returns lists of node sequences representing preconscious paths.
        """
        paths = []

        def _explore(node: ChronNode, current_path: List[ChronNode], depth: int):
            current_path.append(node)
            if depth >= max_depth or not node.children:
                paths.append(current_path.copy())
            else:
                for child in node.children:
                    _explore(child, current_path, depth + 1)
            current_path.pop()

        _explore(start_node, [], 1)
        return paths

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Sample temporal network
    cn = ChronNet()
    t = np.linspace(0, 1, 5)
    vectors = [np.random.rand(3) for _ in t]
    cn.add_trajectory(vectors, t)

    # Traverse
    print("All nodes:")
    for node in cn.traverse():
        print(f"Time {node.timestamp:.2f}, vector {node.vector}")

    # Search by coherence
    target = np.array([0.5, 0.5, 0.5])
    matching_nodes = cn.search_by_coherence(target)
    print(f"\nNodes coherent with {target}: {len(matching_nodes)} found")
