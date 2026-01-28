import numpy as np
from core import SVGelonaCore
from global_icosahedral import GlobalIcosahedron, GlobalDynamics
from angular_defect_dynamics import AngularDefectDynamics

if __name__ == '__main__':
    # Inicialitzar core AI
    ai = SVGelonaCore()

    # Inicialitzar icosaedre global i dinàmica
    ico = GlobalIcosahedron()
    global_dyn = GlobalDynamics(ico, k=0.05)

    # Generar vectors objectiu aleatoris
    target_vectors = [np.random.rand(3) for _ in range(len(ico.vertices))]

    # Aplicar la dinàmica global als vèrtexs
    updated_vertices = global_dyn.step(target_vectors)

    # Simular inputs al core
    input_vectors = [np.random.rand(8) for _ in range(3)]
    outputs = ai.simulate_inputs(input_vectors)

    print("--- SVGelona_AI 4.0 Full Integration Test ---")
    for idx, out in enumerate(outputs):
        print(f"\nInput vector {idx}: {input_vectors[idx]}")
        print(f"Aligned vector: {out['aligned_vector']}")
        print(f"Mobius 8-loop vectors:")
        for m_idx, m_vec in enumerate(out['mobius_vectors']):
            print(f"  Symmetry {m_idx}: {m_vec}")
        print(f"Emergent symbols: {out['symbols']}")

    print("\nUpdated icosahedron vertices:")
    for v_idx, v in enumerate(updated_vertices):
        print(f"Vertex {v_idx}: {v}")

    print("\nFull integration test complete.")
