"""
Check QuantumLayer shape handling.
"""
import sys
sys.path.append('src')

import tensorflow as tf
from quantum_nn.layers import QuantumLayer

print("🔍 Checking QuantumLayer shape behavior...")

# Create layer
layer = QuantumLayer(n_qubits=4)

# Test different input shapes
test_inputs = [
    tf.random.normal((1, 4)),    # Single sample
    tf.random.normal((5, 4)),    # Batch
    tf.random.normal((10, 8)),   # Different feature size
]

for i, inp in enumerate(test_inputs):
    print(f"\nTest {i+1}: Input shape {inp.shape}")
    try:
        # Build layer if needed
        if not layer.built:
            layer.build(inp.shape)
        
        output = layer(inp)
        print(f"✅ Output shape: {output.shape}")
        print(f"   Output dtype: {output.dtype}")
        print(f"   Output sample: {output[0, :5]}...")  # First 5 values
        
        # Check if output is differentiable
        if tf.is_tensor(output) and output.dtype in [tf.float32, tf.float64]:
            print("   ✅ Output is differentiable")
        else:
            print("   ❌ Output might not be differentiable")
            
    except Exception as e:
        print(f"❌ Failed: {e}")

# Check the actual circuit inside QuantumLayer
print("\n🔬 Checking QuantumLayer internals...")
layer = QuantumLayer(n_qubits=4)
print(f"Layer attributes: {[attr for attr in dir(layer) if not attr.startswith('_')]}")
