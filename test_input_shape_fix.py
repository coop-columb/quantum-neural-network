"""
Quick test to verify the input shape fix works.
"""
import sys
sys.path.append('src')

import numpy as np
import tensorflow as tf

try:
    from quantum_nn.applications.medical_imaging.preprocessing import ImageProcessor
    from quantum_nn.applications.medical_imaging.models import create_medical_quantum_classifier
    
    # Create test data
    processor = ImageProcessor(target_size=(32, 32), quantum_target_size=(8, 8))
    test_images = np.random.random((5, 64, 64, 3))
    processed = processor.preprocess_batch(test_images)
    features = processor.extract_features(processed)
    quantum_features = processor.prepare_for_quantum(features)
    
    print(f"✅ Quantum features shape: {quantum_features.shape}")
    
    # Create and test quantum classifier
    quantum_model = create_medical_quantum_classifier(
        input_shape=quantum_features.shape[1:],
        n_classes=2,
        n_qubits=4,
        circuit_complexity='simple'
    )
    print("✅ Quantum classifier created successfully!")
    
    # Compile and test prediction
    quantum_model.compile_for_medical_imaging()
    print("✅ Model compiled!")
    
    prediction = quantum_model.predict(quantum_features[:2], verbose=0)
    print(f"✅ Prediction successful! Shape: {prediction.shape}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
