# run.py — Entry point for SVGelona_IA 2.0
# Versió corregida i plenament integrada

import numpy as np
import time

# Import the core modules with correct names
from core import HoloneticSystem, Icosahedron
from geometry import CriticalPoint, CentralVector
from chronnet import ChronNet
from AngularDefectDynamics import AngularDefectDynamics, SpiralVector
from global_icosahedral import GlobalIcosahedron, GlobalDynamics
from mobius_symmetry import Mobius8Loop
from fractal_language_generator import FractalLanguageGenerator
from search_engine import GeometricSearchEngine

# -----------------------------
# 1. Initialize Full AI System
# -----------------------------
def initialize_full_system():
    """Initialize all components in an integrated pipeline"""
    
    print("="*60)
    print("🚀 SVGelona_IA 2.0 - Initializing Full System")
    print("="*60)
    
    # 1. Core Holonetic System
    print("\n1. Initializing Holonetic Core...")
    system = HoloneticSystem()
    
    # 2. Add critical points along Re(s)=1/2
    print("   Adding critical points...")
    critical_points = []
    for t in [0.1, 0.5, 1.0, 2.0, 3.0]:
        cp = system.add_critical_point(t)
        critical_points.append(cp)
        print(f"     Critical point at t={t}: {cp.complex}")
    
    # 3. Create icosahedron and generate trajectory
    print("\n2. Generating geometric structures...")
    ico = Icosahedron()
    print(f"   Icosahedron created with {len(ico.vertices)} vertices")
    
    # Generate trajectory
    trajectory = system.generate_trajectory(ico, angle=np.pi/4)
    print(f"   Trajectory generated with {len(trajectory)} points")
    
    # 4. Initialize ChronNet with trajectory
    print("\n3. Building temporal network...")
    timestamps = np.linspace(0, 1, len(trajectory))
    system.chronnet.add_trajectory(trajectory, timestamps)
    print(f"   ChronNet now has {len(list(system.chronnet.traverse()))} temporal nodes")
    
    # 5. Create Angular Defect Dynamics
    print("\n4. Setting up angular defect dynamics...")
    # Create a second rotated icosahedron
    rotation = np.array([
        [np.cos(np.pi/8), -np.sin(np.pi/8), 0],
        [np.sin(np.pi/8), np.cos(np.pi/8), 0],
        [0, 0, 1]
    ])
    ico2 = Icosahedron(rotation)
    z0 = critical_points[0]
    
    # Initialize dynamics
    dynamics = AngularDefectDynamics(ico, ico2, z0)
    central_vec = CentralVector(direction=np.array([1.0, 0.0, 0.0]))
    dynamics.initialize_spirals(central_vec)
    print(f"   Angular defect between icosahedra: {dynamics.ico1.vertices[0][:3]} -> {dynamics.ico2.vertices[0][:3]}")
    
    # 6. Initialize Global Icosahedral Structure
    print("\n5. Building global icosahedral structure...")
    icos = [Icosahedron() for _ in range(8)]
    global_ico = GlobalIcosahedron(icosahedra=icos, z0=z0)
    global_dynamics = GlobalDynamics(global_ico, system.chronnet)
    global_dynamics.initialize_dynamics(central_vec)
    print("   8-icosahedron global structure created")
    
    # 7. Initialize Möbius Symmetry Loop
    print("\n6. Initializing Möbius 8-symmetry loop...")
    mobius_loop = Mobius8Loop()
    
    # 8. Initialize Emergent Language Generator
    print("\n7. Initializing emergent language system...")
    language_gen = FractalLanguageGenerator(system.chronnet)
    
    # 9. Initialize Geometric Search Engine
    print("\n8. Initializing geometric search engine...")
    search_engine = GeometricSearchEngine(depth=5)
    
    print("\n" + "="*60)
    print("✅ System Initialization Complete!")
    print("="*60)
    
    return {
        'system': system,
        'ico': ico,
        'dynamics': dynamics,
        'global_dynamics': global_dynamics,
        'mobius_loop': mobius_loop,
        'language_gen': language_gen,
        'search_engine': search_engine,
        'critical_points': critical_points
    }

# -----------------------------
# 2. AI Processing Pipeline
# -----------------------------
def process_ai_input(input_vector, components):
    """
    Full AI processing pipeline for an input vector
    """
    print(f"\n🎯 Processing AI input (norm: {np.linalg.norm(input_vector):.4f})")
    
    system = components['system']
    language_gen = components['language_gen']
    dynamics = components['dynamics']
    mobius_loop = components['mobius_loop']
    
    # 1. Add to tesseract system
    print("   1. Tesseract regulation...")
    new_vector = system.update(input_vector)
    print(f"     Aligned vector: {new_vector[:3]}...")
    
    # 2. Apply Möbius transformations
    print("   2. Applying Möbius 8-symmetry...")
    complex_rep = new_vector[0] + 1j * new_vector[1]
    mobius_transformed = mobius_loop.apply_all(complex_rep)
    print(f"     First Möbius transform: {mobius_transformed[0]:.3f}")
    
    # 3. Evolve spiral dynamics
    print("   3. Evolving spiral dynamics...")
    for _ in range(5):
        dynamics.step()
    print(f"     Spiral vectors evolved")
    
    # 4. Generate emergent symbols
    print("   4. Generating emergent language...")
    # Create a simple trajectory from the vector
    trajectory_vectors = []
    for i in range(min(10, len(new_vector))):
        rotated = new_vector.copy()
        rotated = np.roll(rotated, i)
        trajectory_vectors.append(rotated)
    
    symbolic_sequence = language_gen.generate_sequence(trajectory_vectors, max_length=8)
    print(f"     Symbolic sequence: {symbolic_sequence}")
    
    # 5. Search for coherent thoughts
    print("   5. Exploring preconscious thoughts...")
    search_results = components['search_engine'].semantic_search("coherent")
    print(f"     Found {len(search_results)} coherent trajectories")
    
    return {
        'aligned_vector': new_vector,
        'mobius_transforms': mobius_transformed,
        'symbolic_sequence': symbolic_sequence,
        'spirals_evolved': 5,
        'search_results': len(search_results)
    }

# -----------------------------
# 3. Example Demonstrations
# -----------------------------
def run_demonstrations(components):
    """Run various demonstrations of system capabilities"""
    
    print("\n" + "="*60)
    print("🧪 SYSTEM DEMONSTRATIONS")
    print("="*60)
    
    # Demonstration 1: Basic vector processing
    print("\n📊 DEMO 1: Basic AI Processing")
    print("-"*40)
    test_vector = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.8, 0.4, 0.6])
    result = process_ai_input(test_vector, components)
    
    print(f"\n   Input vector shape: {test_vector.shape}")
    print(f"   Output sequence: {result['symbolic_sequence']}")
    print(f"   Möbius transforms generated: {len(result['mobius_transforms'])}")
    
    # Demonstration 2: Multiple inputs
    print("\n📊 DEMO 2: Batch Processing")
    print("-"*40)
    batch_inputs = [
        np.random.rand(8) for _ in range(3)
    ]
    
    for i, inp in enumerate(batch_inputs):
        inp = inp / np.linalg.norm(inp)
        result = process_ai_input(inp, components)
        print(f"   Input {i+1}: {result['symbolic_sequence']}")
    
    # Demonstration 3: Temporal exploration
    print("\n📊 DEMO 3: Temporal Network Exploration")
    print("-"*40)
    chronnet = components['system'].chronnet
    
    if len(list(chronnet.traverse())) > 0:
        # Search for coherent nodes
        target = np.array([1.0, 0.0, 0.0])
        coherent_nodes = chronnet.search_by_coherence(target, threshold=0.8)
        print(f"   Nodes coherent with [1,0,0]: {len(coherent_nodes)}")
        
        # Explore paths
        if coherent_nodes:
            paths = chronnet.explore_paths(coherent_nodes[0], max_depth=3)
            print(f"   Preconscious paths from first coherent node: {len(paths)}")
    
    # Demonstration 4: Angular defect measurement
    print("\n📊 DEMO 4: Geometric Analysis")
    print("-"*40)
    dynamics = components['dynamics']
    if dynamics.spiral_vectors:
        print(f"   Number of spiral vectors: {len(dynamics.spiral_vectors)}")
        print(f"   First spiral vector: {dynamics.spiral_vectors[0].vector[:3]}")
    
    # Demonstration 5: Language generation
    print("\n📊 DEMO 5: Emergent Language Generation")
    print("-"*40)
    language_gen = components['language_gen']
    
    # Generate sentences from multiple trajectories
    trajectories = []
    for _ in range(3):
        random_traj = [np.random.rand(8) for _ in range(5)]
        trajectories.append(random_traj)
    
    sentences = language_gen.generate_sentences(trajectories, max_words=3)
    print(f"   Generated sentences:")
    for i, sentence in enumerate(sentences):
        print(f"     {i+1}. '{sentence}'")

# -----------------------------
# 4. Main Entry Point
# -----------------------------
def main():
    """Main execution function"""
    
    print("\n" + "="*60)
    print("🏛️  SVGELONA IA 2.0 - HOLONETIC GEOMETRIC AI")
    print("="*60)
    print("Integrating: Critical Geometry + ChronNet + Möbius Symmetry")
    print("            + Angular Defect Dynamics + Emergent Language")
    print("="*60)
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    try:
        # Initialize all components
        start_time = time.time()
        components = initialize_full_system()
        init_time = time.time() - start_time
        
        print(f"\n⏱️  Initialization time: {init_time:.2f} seconds")
        
        # Run demonstrations
        run_demonstrations(components)
        
        # Interactive mode
        print("\n" + "="*60)
        print("💭 INTERACTIVE MODE")
        print("="*60)
        print("Enter 8 numbers (comma-separated) or 'q' to quit")
        print("Example: 1, 0.5, 0.3, 0.2, 0.1, 0.8, 0.4, 0.6")
        
        while True:
            user_input = input("\n📥 Your input (or 'q'): ").strip()
            
            if user_input.lower() == 'q':
                break
            
            try:
                # Parse input
                values = [float(x.strip()) for x in user_input.split(',')]
                if len(values) != 8:
                    print("❌ Please enter exactly 8 numbers")
                    continue
                
                input_vector = np.array(values)
                
                # Process
                result = process_ai_input(input_vector, components)
                
                print(f"\n📤 AI Response:")
                print(f"   Symbolic meaning: {result['symbolic_sequence']}")
                print(f"   Vector (first 3): {result['aligned_vector'][:3]}")
                print(f"   Processed by: {result['spirals_evolved']} spirals")
                print(f"   Found {result['search_results']} coherent thoughts")
                
            except ValueError:
                print("❌ Invalid input. Please enter numbers like: 1, 0.5, 0.3, ...")
            except Exception as e:
                print(f"❌ Processing error: {e}")
        
        print("\n" + "="*60)
        print("👋 Session ended. Thank you for exploring SVGelona_IA 2.0!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Fatal error during initialization: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

# -----------------------------
# 5. Quick Test Function
# -----------------------------
def quick_test():
    """Quick test to verify the system works"""
    print("Running quick test...")
    
    try:
        # Minimal initialization
        system = HoloneticSystem()
        cp = system.add_critical_point(0.5)
        ico = Icosahedron()
        traj = system.generate_trajectory(ico, np.pi/4)
        
        print(f"✅ Core system working:")
        print(f"   Critical point: {cp.complex}")
        print(f"   Icosahedron vertices: {len(ico.vertices)}")
        print(f"   Trajectory points: {len(traj)}")
        
        # Test ChronNet
        system.chronnet.add_trajectory(traj[:3], [0, 0.1, 0.2])
        nodes = list(system.chronnet.traverse(max_depth=2))
        print(f"   ChronNet nodes: {len(nodes)}")
        
        # Test regulation
        from tesseract_module import regulate
        F = [FixedTesseract(i) for i in range(3)]
        G = [FloatingTesseract(i) for i in range(3)]
        G_reg = regulate(G, F, max_iterations=10)
        print(f"   Tesseracts regulated: {len(G_reg)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    
    # Check if running in test mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        if quick_test():
            print("\n✅ All tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Tests failed!")
            sys.exit(1)
    else:
        # Run full system
        sys.exit(main())