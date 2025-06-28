"""
Test quantum layer gradient propagation.
"""
import sys
sys.path.append('src')

import numpy as np
import tensorflow as tf

from quantum_nn.layers import QuantumLayer

# Create a simple model with quantum layer
def create_simple_quantum_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation='relu', input_shape=(10,)),
        QuantumLayer(n_qubits=4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Test gradient flow
print("🧪 Testing quantum layer gradient flow...")

try:
    # Create model
    model = create_simple_quantum_model()
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Create dummy data
    x = np.random.random((5, 10))
    y = np.random.randint(0, 2, (5, 1))
    
    # Test prediction
    pred = model.predict(x[:2], verbose=0)
    print(f"✅ Prediction works: {pred.shape}")
    
    # Test single training step
    with tf.GradientTape() as tape:
        predictions = model(x[:2], training=True)
        loss = tf.keras.losses.binary_crossentropy(y[:2], predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    print(f"✅ Gradient computation works: {len(gradients)} gradients")
    
    # Check for None gradients
    none_grads = [i for i, g in enumerate(gradients) if g is None]
    if none_grads:
        print(f"❌ None gradients at indices: {none_grads}")
        for i in none_grads:
            print(f"   Variable: {model.trainable_variables[i].name}")
    else:
        print("✅ All gradients are non-None")
    
    # Try actual training
    print("\n🏋️ Testing actual training...")
    history = model.fit(x, y, epochs=2, batch_size=2, verbose=0)
    print(f"✅ Training completed! Final loss: {history.history['loss'][-1]:.4f}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
