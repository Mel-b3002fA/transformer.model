from torchviz import make_dot
import torch
from model.gpt import GPT, GPTConfig

# Create a GPTConfig instance
config = GPTConfig(vocab_size=50257, block_size=64)

# Pass the config object to GPT
model = GPT(config)

# Create a random input tensor (e.g., for testing the forward pass)
x = torch.randint(0, 50257, (1, 64))  # Batch size 1, sequence length 64

# Forward pass to get the computation graph
logits, _ = model(x)

# Generate the computation graph (visualization)
dot = make_dot(logits, params=dict(model.named_parameters()))

# Save the graph to a file
dot.render("model_architecture", format="png")

print("Model architecture visualization saved as model_architecture.png")
