# benchmark_and_test_final.py
import time
import numpy as np
import sys
import os
import importlib.util

# Desactivar l'execució d'exemples en imports
os.environ['SKIP_EXAMPLES'] = '1'

# Afegir directori actual al path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ----------------------------
# Funció d'import dinàmic
# ----------------------------
def import_module_from_path(module_name, file_path):
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"✅ Imported {module_name} successfully")
        return module
    except Exception as e:
        print(f"❌ Failed to import {module_name}: {e}")
        return None

# ----------------------------
# Tests simples
# ----------------------------
def run_simple_tests():
    print("\n" + "="*50)
    print("RUNNING SIMPLE TESTS")
    print("="*50)
    
    results = []

    # Geometry
    try:
        from geometry import Icosahedron, CriticalPoint
        cp = CriticalPoint(t=1.0)
        assert cp.complex == 0.5 + 1j
        results.append(("Geometry: CriticalPoint", True))
        
        ico = Icosahedron()
        assert ico.vertices.shape == (12, 3)
        results.append(("Geometry: Icosahedron", True))
    except Exception as e:
        results.append((f"Geometry: {e}", False))

    # ChronNet
    try:
        from chronnet import ChronNet, ChronNode
        cn = ChronNet()
        assert len(cn.root_nodes) == 0
        results.append(("ChronNet: Empty creation", True))
        
        node = ChronNode(vector=np.array([1,2,3]), timestamp=0.0)
        assert node.timestamp == 0.0
        results.append(("ChronNet: Node creation", True))
    except Exception as e:
        results.append((f"ChronNet: {e}", False))

    # AngularDefectDynamics / SpiralVector
    SpiralVector = None
    module_path = os.path.join(BASE_DIR, "AngularDefectDynamics.py")
    module = import_module_from_path("AngularDefectDynamics", module_path)
    if module:
        SpiralVector = getattr(module, "SpiralVector", None)
        if SpiralVector:
            try:
                spiral = SpiralVector(vector=np.array([1.0, 0.0, 0.0]),
                                      origin=None, z0=None, step_size=0.01)
                original = spiral.vector.copy()
                spiral.evolve_step()
                assert not np.array_equal(spiral.vector, original)
                results.append(("SpiralVector: evolve_step", True))
            except Exception as e:
                results.append((f"SpiralVector runtime error: {e}", False))
        else:
            results.append(("SpiralVector not found in AngularDefectDynamics", False))
    else:
        results.append(("AngularDefectDynamics module not loaded", False))

    # Mostrar resultats
    print("\nTEST RESULTS:")
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name} [{status}]")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    print(f"\n📊 {passed_count}/{total_count} tests passed")
    
    return passed_count == total_count

# ----------------------------
# Benchmarks
# ----------------------------
def run_benchmarks():
    print("\n" + "="*50)
    print("RUNNING CONTROLLED BENCHMARKS")
    print("="*50)
    
    try:
        from geometry import Icosahedron
        from chronnet import ChronNet, ChronNode
        module_path = os.path.join(BASE_DIR, "AngularDefectDynamics.py")
        module = import_module_from_path("AngularDefectDynamics", module_path)
        SpiralVector = getattr(module, "SpiralVector", None) if module else None
    except Exception as e:
        print(f"❌ Cannot import modules for benchmark: {e}")
        return

    # Benchmark 1: Icosahedron
    print("\n1. 🎲 ICOSAHEDRON BATCH CREATION")
    times = []
    for _ in range(5):
        start = time.time()
        for _ in range(50):
            Icosahedron()
        times.append(time.time() - start)
    avg_time = np.mean(times)
    print(f"   Average per icosahedron: {avg_time/50:.6f}s")

    # Benchmark 2: SpiralVector evolution
    if SpiralVector:
        print("\n2. 🌀 SPIRAL EVOLUTION")
        spirals = [SpiralVector(vector=np.random.randn(3),
                                origin=None, z0=None, step_size=0.01)
                   for _ in range(20)]
        start = time.time()
        for spiral in spirals:
            for _ in range(50):
                spiral.evolve_step()
        spiral_time = time.time() - start
        print(f"   20 spirals × 50 steps each: {spiral_time:.4f}s")
    else:
        print("⚠️ SpiralVector not available, skipping spiral benchmark")

    # Benchmark 3: ChronNet
    print("\n3. ⏳ CHRONNET OPERATIONS")
    cn = ChronNet()
    nodes = [ChronNode(vector=np.random.rand(3), timestamp=i*0.1) for i in range(100)]
    start = time.time()
    prev_node = None
    for node in nodes:
        if prev_node:
            prev_node.children.append(node)
        else:
            cn.root_nodes.append(node)
        prev_node = node
    linking_time = time.time() - start
    print(f"   Linked 100 nodes: {linking_time:.4f}s")
    print(f"   Total ChronNet size: {len(list(cn.traverse()))} nodes")

# ----------------------------
# Funció principal
# ----------------------------
def main():
    print("="*60)
    print("SVGELONA IA 2.0 - TEST & BENCHMARK FINAL")
    print("="*60)
    
    print("\n🚀 Starting tests...")
    all_passed = run_simple_tests()
    
    print("\n⚡ Starting benchmarks...")
    run_benchmarks()
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"✅ ALL TESTS PASSED" if all_passed else "⚠️ SOME TESTS FAILED")

    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
