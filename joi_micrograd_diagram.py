from micrograd.engine import Value
from micrograd.utils import draw_dot

# Token embedding (simplified 2D example from 768)
x1 = Value(0.5, label='embed[0]')
x2 = Value(-1.2, label='embed[1]')

# First MLP layer (1 of 12 layers)
w1 = Value(0.8, label='W1 (Layer 1 of 12)')
w2 = Value(-0.5, label='W2')
b1 = Value(0.1, label='Bias1')

# Linear → activation
h = x1 * w1 + x2 * w2 + b1
h.label = 'Linear → Layer 1'
h_relu = h.relu()
h_relu.label = 'ReLU → Layer 1'

# Output projection (1 of 12 attention heads)
w3 = Value(1.5, label='W3 (Head 1 of 12)')
b2 = Value(0.0, label='Bias2')
logit = h_relu * w3 + b2
logit.label = 'Logit (Vocab Score)'

# Save the graph
dot = draw_dot(logit)
dot.render("joi_micrograd_diagram", format="png", cleanup=True)
