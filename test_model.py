""" from model.transformer import GPT, GPTConfig

# Define a small test config
config = GPTConfig(
    n_layer=2,
    n_head=2,
    n_embd=128,
    block_size=64
)

model = GPT(config)
import torch

# Batch of 4 sequences, each 64 tokens long
x = torch.randint(0, config.vocab_size, (4, config.block_size))

# Forward pass
out = model(x)

print("Output shape:", out.shape) """






















""" import torch
from model.transformer import GPT, GPTConfig
from data.toy_data.toy_data import toy_data 

config = GPTConfig(
    vocab_size=10,
    block_size=4,
    n_layer=2,
    n_head=2,
    n_embd=64
)

model = GPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


model.train()
for step in range(100):
    total_loss = 0.0
    for seq in toy_data:
        x = torch.tensor(seq[:-1]).unsqueeze(0)  
        y = torch.tensor(seq[1:]).unsqueeze(0)  

        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if step % 10 == 0:
        print(f"Step {step}, Loss: {total_loss:.4f}")


""" """


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model.transformer import GPT, GPTConfig


config = GPTConfig(
    vocab_size=1000,
    n_embd=64,
    n_head=4,
    n_layer=4
)

model = GPT(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


batch_size = 8
seq_len = 16
x = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
y = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)


losses = []

for step in range(100):
    model.train()
    optimizer.zero_grad()

    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")


plt.plot(losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.grid(True)
plt.savefig("loss_curve.png")  


model.eval()
with torch.no_grad():
    test_input = x[0].unsqueeze(0) 
    logits, _ = model(test_input)
    prediction = torch.argmax(logits, dim=-1)

    print("\nSample Prediction:")
    print("Input:     ", test_input[0].tolist())
    print("Predicted: ", prediction[0].tolist())
 """

""" 


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model.transformer import GPT, GPTConfig
from data.toy_data.toy_data import toy_data  # Comment if using real data

# === Settings ===
USE_TOY_DATA = True  # Toggle between toy data and random data
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Model Configuration ===
def create_config(use_toy=True):
    if use_toy:
        return GPTConfig(
            vocab_size=10,
            block_size=4,
            n_layer=2,
            n_head=2,
            n_embd=64
        )
    else:
        return GPTConfig(
            vocab_size=1000,
            block_size=16,
            n_layer=4,
            n_head=4,
            n_embd=64
        )


# === Training Function ===
def train_model(model, optimizer, config, use_toy=True, steps=100):
    model.train()
    all_losses = []

    for step in range(steps):
        total_loss = 0.0

        if use_toy:
            for seq in toy_data:
                x = torch.tensor(seq[:-1]).unsqueeze(0).to(DEVICE)
                y = torch.tensor(seq[1:]).unsqueeze(0).to(DEVICE)

                logits, loss = model(x, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        else:
            batch_size = 8
            seq_len = config.block_size
            x = torch

# Your imports, config functions, training, evaluation, plotting go here...

def main():
    config = create_config(USE_TOY_DATA)
    model = GPT(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = train_model(model, optimizer, config, use_toy=USE_TOY_DATA)
    evaluate_model(model, config, use_toy=USE_TOY_DATA)
    plot_losses(losses)


if __name__ == "__main__":
    main() """




import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder

def toy_data():
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=7, n_informative=5, 
                               n_redundant=1, n_classes=5, n_clusters_per_class=1)

    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y

    # Convert categorical features into numbers
    df['cat_feature_1'] = np.random.choice(['A', 'B', 'C'], size=500)
    df['cat_feature_2'] = np.random.choice(['X', 'Y'], size=500)

    # Use LabelEncoder to convert categorical features into integers
    label_encoder_1 = LabelEncoder()
    df['cat_feature_1'] = label_encoder_1.fit_transform(df['cat_feature_1'])
    
    label_encoder_2 = LabelEncoder()
    df['cat_feature_2'] = label_encoder_2.fit_transform(df['cat_feature_2'])

    df['interaction'] = df['feature_0'] * df['feature_1'] + np.sin(df['feature_2'])
    df['random_noise'] = np.random.normal(0, 1, size=500)

    return df
