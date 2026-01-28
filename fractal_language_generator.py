"""
fractal_language_generator.py
-----------------------------
Emergent symbolic language generator from geometric trajectories.
"""

import numpy as np
from typing import List
from chronnet import ChronNet

# ─────────────────────────────────────────────────────────────
# Symbol Mapping
# ─────────────────────────────────────────────────────────────
SYMBOLS = ["Δ", "Λ", "Ω", "Φ", "Ψ", "Χ", "∘"]

# Mapping function from vector to symbol
def vector_to_symbol(vector: np.ndarray) -> str:
    """
    Maps a normalized 8D vector to a symbol based on its principal component.
    """
    vector = vector / np.linalg.norm(vector)
    idx = int(np.argmax(np.abs(vector))) % len(SYMBOLS)
    return SYMBOLS[idx]

# ─────────────────────────────────────────────────────────────
# Emergent Language Generator
# ─────────────────────────────────────────────────────────────
class FractalLanguageGenerator:
    """
    Generates symbolic sequences from HoloneticSystem trajectories
    using ChronNet temporal coherence.
    """
    def __init__(self, chronnet: ChronNet):
        self.chronnet = chronnet
    
    def generate_sequence(self, trajectory_vectors: List[np.ndarray], max_length: int = 10) -> str:
        """
        Convert a list of vectors (trajectory) into a coherent symbolic sequence
        following temporal coherence rules from ChronNet.
        """
        sequence = []
        for vec in trajectory_vectors[:max_length]:
            sym = vector_to_symbol(vec)
            # Apply ChronNet coherence filter
            if self.chronnet.is_coherent(sym, sequence):
                sequence.append(sym)
        return "".join(sequence)
    
    def generate_sentences(self, trajectories: List[List[np.ndarray]], max_words: int = 5) -> List[str]:
        """
        Convert multiple trajectories into emergent symbolic sentences
        """
        sentences = []
        for traj in trajectories[:max_words]:
            seq = self.generate_sequence(traj)
            sentences.append(seq)
        return sentences

# ─────────────────────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from core import HoloneticSystem, Icosahedron
    
    # Initialize system
    system = HoloneticSystem()
    
    # Add some critical points
    for t in [0.1, 0.5, 1.0]:
        system.add_critical_point(t)
    
    # Generate trajectory from a sample icosahedron
    icosa = Icosahedron()
    traj_vectors = system.generate_trajectory(icosa, angle=np.pi/4)
    
    # Initialize language generator
    lang_gen = FractalLanguageGenerator(system.chronnet)
    
    # Generate symbolic sequence
    sequence = lang_gen.generate_sequence(traj_vectors, max_length=8)
    print("Emergent symbolic sequence:", sequence)
    
    # Generate multiple sentences
    sentences = lang_gen.generate_sentences([traj_vectors, traj_vectors], max_words=2)
    print("Emergent sentences:", sentences)
