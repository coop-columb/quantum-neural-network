import sys
sys.path.append('src')

print("Testing step-by-step imports...")

try:
    print("1. Testing quantum_nn base imports...")
    from quantum_nn.circuits import ParameterizedCircuit
    print("   ✅ circuits imported")
    
    from quantum_nn.models import QuantumModel
    print("   ✅ models imported")
    
    from quantum_nn.optimizers import ParameterShiftOptimizer
    print("   ✅ optimizers imported")
    
except Exception as e:
    print(f"   ❌ Base imports failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("2. Testing preprocessing imports...")
    from quantum_nn.applications.medical_imaging.preprocessing.image_processor import ImageProcessor
    print("   ✅ ImageProcessor imported directly")
    
    from quantum_nn.applications.medical_imaging.preprocessing.data_pipeline import DataPipeline
    print("   ✅ DataPipeline imported directly")
    
except Exception as e:
    print(f"   ❌ Direct preprocessing imports failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("3. Testing preprocessing module import...")
    from quantum_nn.applications.medical_imaging.preprocessing import ImageProcessor, DataPipeline
    print("   ✅ Preprocessing module imports successful")
    
except Exception as e:
    print(f"   ❌ Preprocessing module imports failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("4. Testing model imports...")
    from quantum_nn.applications.medical_imaging.models.quantum_classifier import MedicalQuantumClassifier
    print("   ✅ MedicalQuantumClassifier imported directly")
    
    from quantum_nn.applications.medical_imaging.models.hybrid_model import MedicalHybridModel
    print("   ✅ MedicalHybridModel imported directly")
    
except Exception as e:
    print(f"   ❌ Direct model imports failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("5. Testing full application import...")
    from quantum_nn.applications.medical_imaging import (
        ImageProcessor, DataPipeline,
        MedicalQuantumClassifier, MedicalHybridModel
    )
    print("   ✅ Full application imports successful")
    
except Exception as e:
    print(f"   ❌ Full application imports failed: {e}")
    import traceback
    traceback.print_exc()

print("Debug complete!")
