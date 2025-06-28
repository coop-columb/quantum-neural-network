"""
Test gradient flow through QuantumLayer in a model.
"""
import sys
sys.path.append('src')

import numpy as np
import tensorflow as tf
from quantum_nn.layers import QuantumLayer

print("🔍 Testing gradient flow through layers...")

# Test 1: Dense -> QuantumLayer -> Dense
print("\n1️⃣ Testing Dense -> QuantumLayer -> Dense...")
try:
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(4)(inputs)  # No activation first
    x = QuantumLayer(n_qubits=4)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Test with small batch
    x_train = np.random.random((4, 10))
    y_train = np.random.randint(0, 2, (4, 1))
    
    history = model.fit(x_train, y_train, epochs=2, batch_size=2, verbose=0)
    print(f"✅ Model without ReLU trains! Loss: {history.history['loss'][-1]:.4f}")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Add ReLU activation
print("\n2️⃣ Testing with ReLU activation...")
try:
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(4, activation='relu')(inputs)
    x = QuantumLayer(n_qubits=4)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Test prediction first
    pred = model.predict(x_train[:2], verbose=0)
    print(f"   Prediction works: {pred.shape}")
    
    # Test single gradient computation
    with tf.GradientTape() as tape:
        predictions = model(x_train[:2], training=True)
        loss = tf.keras.losses.binary_crossentropy(y_train[:2], predictions)
        scalar_loss = tf.reduce_mean(loss)
    
    print(f"   Loss shape: {loss.shape}, Scalar loss: {scalar_loss.shape}")
    
    gradients = tape.gradient(scalar_loss, model.trainable_variables)
    print(f"   Computed {len(gradients)} gradients")
    
    # Try training
    history = model.fit(x_train, y_train, epochs=2, batch_size=2, verbose=0)
    print(f"✅ Model with ReLU trains! Loss: {history.history['loss'][-1]:.4f}")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Check if the issue is batch normalization
print("\n3️⃣ Testing with BatchNormalization...")
try:
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(4, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = QuantumLayer(n_qubits=4)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    history = model.fit(x_train, y_train, epochs=2, batch_size=2, verbose=0)
    print(f"✅ Model with BatchNorm trains! Loss: {history.history['loss'][-1]:.4f}")
    
except Exception as e:
    print(f"❌ Failed with BatchNorm: {e}")

# Test 4: Simplified medical model architecture
print("\n4️⃣ Testing simplified medical architecture...")
try:
    inputs = tf.keras.Input(shape=(20,))
    x = tf.keras.layers.Dense(8)(inputs)  # No activation
    x = QuantumLayer(n_qubits=4)(x)
    x = tf.keras.layers.Dense(4)(x)  # No activation
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Create data
    x_train = np.random.random((10, 20))
    y_train = np.random.randint(0, 2, (10, 1))
    
    history = model.fit(x_train, y_train, epochs=3, batch_size=4, verbose=1)
    print(f"✅ Simplified medical model trains! Final loss: {history.history['loss'][-1]:.4f}")
    
except Exception as e:
    print(f"❌ Simplified model failed: {e}")
    import traceback
    traceback.print_exc()
