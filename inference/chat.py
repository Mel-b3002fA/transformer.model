""" from transformer import Transformer  # or model.py if you're using nanoGPT
import torch

# load your trained model
model = Transformer(config)
model.load_state_dict(torch.load("path/to/model.pt"))
model.eval()
from tokenizers import Tokenizer  # if using HuggingFace's Tokenizers
tokenizer = Tokenizer.from_file("tokenizer.json")

def generate(model, tokenizer, prompt, max_new_tokens=50):
    model.eval()
    input_ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([input_ids])

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)[:, -1, :]  # last token logits
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

    return tokenizer.decode(input_ids[0].tolist())
 """


from model.transformer import Transformer
from model.tokenizer import Tokenizer
import torch

# Load your trained model
model = Transformer(config)
model.load_state_dict(torch.load("assets/best_model.pt"))
model.eval()

# Load tokenizer
tokenizer = Tokenizer.from_file("assets/tokenizer.json")

# Chat loop
print("🧠 LLM is online. Type 'exit' to quit.")
while True:
    prompt = input("You: ")
    if prompt.lower() in ["exit", "quit"]: break

    input_ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([input_ids])

    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=100)

    response = tokenizer.decode(output[0].tolist())
    print("AI:", response)
