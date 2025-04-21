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


""" from model.transformer import Transformer
from model.tokenizers import Tokenizer
import torch

# Load your trained model
model = Transformer(config)
model.load_state_dict(torch.load("assets/best_model.pt"))
model.eval()

# Load tokenizer
tokenizer = Tokenizer.from_file("assets/tokenizer.json")

# Chat loop
print("JOI is online. Type 'exit' to quit.")
while True:
    prompt = input("You: ")
    if prompt.lower() in ["exit", "quit"]: break

    input_ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([input_ids])

    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=100)

    response = tokenizer.decode(output[0].tolist())
    print("AI:", response) """



import torch
from model.transformer import Transformer
from tokenizers import Tokenizer  
tokenizer = Tokenizer.from_file("assets/tokenizer.json")

from training.config import config     # make sure this exists and defines all model hyperparameters


model = Transformer(config)
model.load_state_dict(torch.load("assets/best_model.pt", map_location="cpu"))  # add map_location for CPU compatibility
model.eval()


tokenizer = Tokenizer.from_file("assets/tokenizer.json")

# Generation function (you must define this if it's not in transformer.py or generation.py)
def generate(model, input_ids, max_new_tokens=50):
    model.eval()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)[:, -1, :] 
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids


print("JOI is online. Type 'exit' to quit.")
while True:
    prompt = input("You: ")
    if prompt.lower() in ["exit", "quit"]:
        break

    input_ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([input_ids], dtype=torch.long)

    output_ids = generate(model, input_ids, max_new_tokens=50)
    response = tokenizer.decode(output_ids[0].tolist())

    print("JOI:", response)

