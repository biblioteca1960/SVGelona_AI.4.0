"""
tesseract_module.py
-------------------
Module for fixed and floating tesseracts used in SVGelona_IA.2.0.

- FixedTesseract: Represents an invariant axis (F_i)
- FloatingTesseract: Represents a modifiable axis (G_j)
- regulate(): Self-regulation function aligning floating tesseracts to fixed ones
"""

import numpy as np

# -----------------------------
# Tesseract Classes
# -----------------------------
class FixedTesseract:
    """Represents a fixed, invariant tesseract axis."""
    def __init__(self, index):
        self.index = index
        # Initialize as a unit vector along standard basis
        self.vector = np.zeros(8)
        self.vector[index % 8] = 1.0

class FloatingTesseract:
    """Represents a floating tesseract axis subject to self-regulation."""
    def __init__(self, index):
        self.index = index
        # Initialize as a small random perturbation vector
        self.vector = np.random.rand(8)
        self.vector /= np.linalg.norm(self.vector)

# -----------------------------
# Regulation Function
# -----------------------------
def regulate(floating_list, fixed_list, k=0.1, max_iterations=100, tolerance=1e-6):
    """
    Aligns floating tesseracts with fixed ones using angular deviation minimization.

    Parameters:
    - floating_list: list of FloatingTesseract
    - fixed_list: list of FixedTesseract
    - k: learning rate
    - max_iterations: maximum iterations for convergence
    - tolerance: convergence threshold

    Returns:
    - Updated list of floating tesseracts
    """
    # Compute fixed axes span
    F_matrix = np.stack([f.vector for f in fixed_list])
    
    for _ in range(max_iterations):
        converged = True
        for g in floating_list:
            # Project onto fixed span
            proj = F_matrix.T @ (F_matrix @ g.vector)
            proj /= np.linalg.norm(proj) + 1e-12
            # Compute angular deviation
            delta = g.vector - proj
            if np.linalg.norm(delta) > tolerance:
                # Correct vector towards projection
                g.vector -= k * delta
                g.vector /= np.linalg.norm(g.vector)
                converged = False
        if converged:
            break
    return floating_list

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    F = [FixedTesseract(i) for i in range(8)]
    G = [FloatingTesseract(i) for i in range(8)]

    print("Initial floating vectors:")
    for g in G:
        print(g.vector)

    G = regulate(G, F)

    print("\nAligned floating vectors:")
    for g in G:
        print(g.vector)
