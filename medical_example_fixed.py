"""
Quick demonstration of medical imaging quantum neural networks.
Shows the complete workflow from preprocessing to quantum classification.
"""
import sys
sys.path.append('src')

import numpy as np

from quantum_nn.applications.medical_imaging.preprocessing import ImageProcessor
from quantum_nn.applications.medical_imaging.models import (
    create_medical_quantum_classifier,
    create_hybrid_medical_model
)


def main():
    """Run a quick medical imaging demo."""
    print("🧬 Medical Imaging Quantum Neural Networks - Quick Demo")
    print("=" * 55)
    
    # Create synthetic medical images
    print("📸 Creating synthetic medical images...")
    n_samples = 20
    images = np.random.random((n_samples, 64, 64, 3))
    labels = np.random.randint(0, 2, (n_samples,))
    print(f"   Created {n_samples} images with shape {images.shape}")
    
    # Preprocess images
    print("🔧 Preprocessing images...")
    processor = ImageProcessor(target_size=(32, 32), quantum_target_size=(8, 8))
    processed_images = processor.preprocess_batch(images)
    features = processor.extract_features(processed_images)
    quantum_features = processor.prepare_for_quantum(features)
    print(f"   Quantum features shape: {quantum_features.shape}")
    
    # Create quantum classifier
    print("⚛️  Creating quantum classifier...")
    quantum_model = create_medical_quantum_classifier(
        input_shape=quantum_features.shape[1:],
        n_classes=2,
        n_qubits=4,
        circuit_complexity='simple'
    )
    quantum_model.compile_for_medical_imaging()
    print("   Quantum classifier ready!")
    
    # Create hybrid model
    print("🔀 Creating hybrid model...")
    hybrid_model = create_hybrid_medical_model(
        input_shape=(32, 32, 3),
        n_classes=2,
        model_size='small',
        use_pretrained=False
    )
    hybrid_model.compile_for_medical_imaging()
    print("   Hybrid model ready!")
    
    # Test predictions
    print("🔮 Testing predictions...")
    
    # Quantum model prediction
    quantum_pred = quantum_model.predict(quantum_features[:5], verbose=0)
    print(f"   Quantum predictions: {quantum_pred.flatten()}")
    
    # Hybrid model prediction
    hybrid_pred = hybrid_model.predict(processed_images[:5], verbose=0)
    print(f"   Hybrid predictions: {hybrid_pred.flatten()}")
    
    # Quick training demo (few epochs)
    print("🏋️  Quick training demo...")
    
    # Split data
    split_idx = int(0.8 * len(quantum_features))
    x_train_q = quantum_features[:split_idx]
    y_train_q = labels[:split_idx]
    x_val_q = quantum_features[split_idx:]
    y_val_q = labels[split_idx:]
    
    x_train_h = processed_images[:split_idx]
    y_train_h = labels[:split_idx]
    x_val_h = processed_images[split_idx:]
    y_val_h = labels[split_idx:]
    
    # Train quantum model
    print("   Training quantum model...")
    quantum_history = quantum_model.fit(
        x_train_q, y_train_q,
        validation_data=(x_val_q, y_val_q),
        epochs=3,
        batch_size=4,
        verbose=0
    )
    
    # Train hybrid model
    print("   Training hybrid model...")
    hybrid_history = hybrid_model.fit(
        x_train_h, y_train_h,
        validation_data=(x_val_h, y_val_h),
        epochs=3,
        batch_size=4,
        verbose=0
    )
    
    # Final evaluation
    print("📊 Final Results:")
    q_loss, q_acc = quantum_model.evaluate(x_val_q, y_val_q, verbose=0)
    h_loss, h_acc = hybrid_model.evaluate(x_val_h, y_val_h, verbose=0)
    
    print(f"   Quantum Model - Loss: {q_loss:.4f}, Accuracy: {q_acc:.4f}")
    print(f"   Hybrid Model  - Loss: {h_loss:.4f}, Accuracy: {h_acc:.4f}")
    
    print("\n✅ Demo completed successfully!")
    print("\n🎯 Key Features Demonstrated:")
    print("   • PennyLane-based quantum neural networks")
    print("   • Medical image preprocessing pipeline")
    print("   • Quantum feature extraction and encoding")
    print("   • Hybrid classical-quantum architectures")
    print("   • End-to-end training and evaluation")
    print("   • Apple Silicon native execution")


if __name__ == "__main__":
    main()
