"""
Direct test of QuantumLayer gradient behavior.
"""
import sys
sys.path.append('src')

import numpy as np
import tensorflow as tf
import pennylane as qml

from quantum_nn.layers import QuantumLayer

print("🔍 Testing QuantumLayer directly...")

# Test 1: Basic forward pass
print("\n1️⃣ Testing forward pass...")
try:
    layer = QuantumLayer(n_qubits=4)
    
    # Build the layer
    dummy_input = tf.random.normal((2, 4))
    layer.build(dummy_input.shape)
    
    # Forward pass
    output = layer(dummy_input)
    print(f"✅ Forward pass successful! Output shape: {output.shape}")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")

# Test 2: Gradient computation
print("\n2️⃣ Testing gradient computation...")
try:
    layer = QuantumLayer(n_qubits=4)
    dummy_input = tf.Variable(tf.random.normal((2, 4)))
    
    with tf.GradientTape() as tape:
        output = layer(dummy_input)
        loss = tf.reduce_mean(output)
    
    grads = tape.gradient(loss, dummy_input)
    if grads is None:
        print("❌ Gradients are None - layer might not be differentiable")
    else:
        print(f"✅ Input gradients computed! Shape: {grads.shape}")
    
    # Check layer parameter gradients
    layer_grads = tape.gradient(loss, layer.trainable_variables)
    print(f"   Layer has {len(layer.trainable_variables)} trainable variables")
    print(f"   Layer gradients: {[g.shape if g is not None else 'None' for g in layer_grads]}")
    
except Exception as e:
    print(f"❌ Gradient test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Check if QuantumLayer implements @tf.custom_gradient
print("\n3️⃣ Checking QuantumLayer implementation...")
try:
    # Create a minimal quantum function with gradient
    @tf.custom_gradient
    def quantum_function(x):
        # Simple quantum circuit simulation
        # This is a placeholder - the actual implementation should use PennyLane
        output = tf.reduce_sum(x, axis=-1, keepdims=True)
        
        def grad_fn(upstream):
            # Gradient should flow back properly
            return upstream * tf.ones_like(x)
        
        return output, grad_fn
    
    # Test the custom gradient
    x = tf.Variable([[1.0, 2.0, 3.0, 4.0]])
    with tf.GradientTape() as tape:
        y = quantum_function(x)
    
    grad = tape.gradient(y, x)
    print(f"✅ Custom gradient works! Grad shape: {grad.shape}")
    
except Exception as e:
    print(f"❌ Custom gradient test failed: {e}")

# Test 4: Simple model with proper gradient flow
print("\n4️⃣ Testing simple model without QuantumLayer...")
try:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,)),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    x = np.random.random((5, 10))
    y = np.random.randint(0, 2, (5, 1))
    
    history = model.fit(x, y, epochs=2, batch_size=2, verbose=0)
    print(f"✅ Classical model trains fine! Final loss: {history.history['loss'][-1]:.4f}")
    
except Exception as e:
    print(f"❌ Classical model failed: {e}")
