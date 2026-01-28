"""
mobius_symmetry.py
------------------
Module implementing the Möbius 8-symmetry loop for SVGelona_IA.2.0

- Applies sequential Möbius transformations to simulate motion through 8 symmetries.
- Can operate on tesseract vectors or geometric points in 3D/complex space.
"""

import numpy as np

# -----------------------------
# Möbius Transformation Class
# -----------------------------
class MobiusTransform:
    """
    Represents a single Möbius transformation:
        f(z) = (a*z + b) / (c*z + d)
    where a,b,c,d are complex numbers.
    """
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def apply(self, z):
        """Apply the Möbius transformation to a complex number or array."""
        return (self.a * z + self.b) / (self.c * z + self.d + 1e-12)

# -----------------------------
# 8-Symmetry Loop
# -----------------------------
class Mobius8Loop:
    """
    Generates and applies 8 Möbius transformations to simulate full symmetry loop.
    """
    def __init__(self):
        # Generate 8 canonical Möbius transformations
        self.transforms = self._generate_transforms()
        self.current_index = 0

    def _generate_transforms(self):
        """
        Generate 8 Möbius transformations representing the symmetries of the system.
        Here, simple rotations/scalings in complex plane are used as examples.
        """
        transforms = []
        for k in range(8):
            angle = k * (np.pi / 4)
            a = np.exp(1j * angle)
            b = 0.0
            c = 0.0
            d = 1.0
            transforms.append(MobiusTransform(a, b, c, d))
        return transforms

    def next(self, z):
        """
        Apply the next transformation in the loop to z.
        """
        t = self.transforms[self.current_index]
        result = t.apply(z)
        self.current_index = (self.current_index + 1) % 8
        return result

    def apply_all(self, z):
        """
        Apply all 8 transformations sequentially to z, returning a list of results.
        """
        results = []
        for i in range(8):
            results.append(self.next(z))
        return results

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    loop = Mobius8Loop()
    z0 = 0.5 + 1j * 1.0  # example complex input
    print("Original z0:", z0)

    transformed = loop.apply_all(z0)
    print("\nAfter 8 Möbius symmetries:")
    for idx, z in enumerate(transformed):
        print(f"Symmetry {idx+1}: {z}")
