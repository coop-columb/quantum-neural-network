"""
Test script to verify the medical imaging component fixes.
"""
import sys
sys.path.append('src')

import numpy as np
import tensorflow as tf

# Test imports
print("🔍 Testing imports...")
try:
    from quantum_nn.layers import QuantumLayer
    print("✅ QuantumLayer imported")
    
    from quantum_nn.applications.medical_imaging.preprocessing import ImageProcessor, DataPipeline
    print("✅ Preprocessing imports successful")
    
    from quantum_nn.applications.medical_imaging.models import (
        create_medical_quantum_classifier,
        create_hybrid_medical_model
    )
    print("✅ Model imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    exit(1)

# Test basic components
print("\n🧪 Testing components...")

# Test preprocessing
print("📸 Testing image preprocessing...")
processor = ImageProcessor(target_size=(32, 32), quantum_target_size=(8, 8))
test_images = np.random.random((5, 64, 64, 3))
processed = processor.preprocess_batch(test_images)
features = processor.extract_features(processed)
quantum_features = processor.prepare_for_quantum(features)
print(f"✅ Preprocessing works! Quantum features: {quantum_features.shape}")

# Test quantum classifier
print("\n⚛️  Testing quantum classifier...")
quantum_model = create_medical_quantum_classifier(
    input_shape=quantum_features.shape[1:],
    n_classes=2,
    n_qubits=4,
    circuit_complexity='simple'
)
print("✅ Quantum classifier created")

# Test compilation
quantum_model.compile_for_medical_imaging()
print("✅ Quantum classifier compiled")

# Test prediction
predictions = quantum_model.predict(quantum_features[:2], verbose=0)
print(f"✅ Quantum predictions work! Shape: {predictions.shape}")

# Test hybrid model
print("\n🔀 Testing hybrid model...")
hybrid_model = create_hybrid_medical_model(
    input_shape=(32, 32, 3),
    n_classes=2,
    model_size='small',
    use_pretrained=False
)
print("✅ Hybrid model created")

# Test compilation
hybrid_model.compile_for_medical_imaging()
print("✅ Hybrid model compiled")

# Test prediction
hybrid_predictions = hybrid_model.predict(processed[:2], verbose=0)
print(f"✅ Hybrid predictions work! Shape: {hybrid_predictions.shape}")

print("\n🎉 All tests passed! Medical imaging components are working correctly.")
