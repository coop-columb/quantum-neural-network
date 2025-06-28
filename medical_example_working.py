"""
Working medical imaging example with gradient-safe architecture.
"""
import sys
sys.path.append('src')

import numpy as np
import tensorflow as tf

from quantum_nn.layers import QuantumLayer
from quantum_nn.applications.medical_imaging.preprocessing import ImageProcessor

print("🧬 Medical Imaging Quantum Neural Networks - Working Demo")
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

# Create simplified quantum model (avoiding gradient issues)
print("⚛️  Creating gradient-safe quantum model...")

def create_safe_quantum_model(input_shape, n_qubits=4):
    """Create a quantum model that avoids gradient flow issues."""
    inputs = tf.keras.Input(shape=input_shape)
    
    # Pre-processing without problematic activations
    x = tf.keras.layers.Dense(16)(inputs)
    x = tf.keras.layers.LayerNormalization()(x)  # Use LayerNorm instead of BatchNorm
    
    # Quantum layer
    x = QuantumLayer(n_qubits=n_qubits)(x)
    
    # Post-processing
    x = tf.keras.layers.Dense(8)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    
    # Output
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

quantum_model = create_safe_quantum_model(quantum_features.shape[1:])
quantum_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print("   Gradient-safe quantum model created!")

# Create simplified hybrid model
print("🔀 Creating gradient-safe hybrid model...")

def create_safe_hybrid_model(input_shape, n_qubits=4):
    """Create a hybrid model that avoids gradient flow issues."""
    inputs = tf.keras.Input(shape=input_shape)
    
    # Simple CNN
    x = tf.keras.layers.Conv2D(16, 3, padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Classical processing
    x = tf.keras.layers.Dense(16)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    
    # Quantum layer
    x = QuantumLayer(n_qubits=n_qubits)(x)
    
    # Final processing
    x = tf.keras.layers.Dense(8)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

hybrid_model = create_safe_hybrid_model((32, 32, 3))
hybrid_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print("   Gradient-safe hybrid model created!")

# Test predictions
print("🔮 Testing predictions...")
quantum_pred = quantum_model.predict(quantum_features[:5], verbose=0)
print(f"   Quantum predictions: {quantum_pred.flatten()}")

hybrid_pred = hybrid_model.predict(processed_images[:5], verbose=0)
print(f"   Hybrid predictions: {hybrid_pred.flatten()}")

# Training demo
print("🏋️  Training demonstration...")

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
    epochs=5,
    batch_size=4,
    verbose=0
)
print(f"   Quantum model - Final loss: {quantum_history.history['loss'][-1]:.4f}")

# Train hybrid model
print("   Training hybrid model...")
hybrid_history = hybrid_model.fit(
    x_train_h, y_train_h,
    validation_data=(x_val_h, y_val_h),
    epochs=5,
    batch_size=4,
    verbose=0
)
print(f"   Hybrid model - Final loss: {hybrid_history.history['loss'][-1]:.4f}")

# Final evaluation
print("\n📊 Final Results:")
q_loss, q_acc = quantum_model.evaluate(x_val_q, y_val_q, verbose=0)
h_loss, h_acc = hybrid_model.evaluate(x_val_h, y_val_h, verbose=0)

print(f"   Quantum Model - Loss: {q_loss:.4f}, Accuracy: {q_acc:.4f}")
print(f"   Hybrid Model  - Loss: {h_loss:.4f}, Accuracy: {h_acc:.4f}")

print("\n✅ Demo completed successfully!")
print("\n🎯 Key Insights:")
print("   • Avoided ReLU activations before QuantumLayer")
print("   • Used LayerNormalization instead of BatchNormalization")
print("   • Simplified architecture for stable gradients")
print("   • Both quantum and hybrid models train successfully")
