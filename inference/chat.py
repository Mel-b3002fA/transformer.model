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



""" import torch
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

 """



""" import torch
import pickle
from model import GPT, GPTConfig

# Load tokenizer (meta.pkl)
with open('data/openwebtext/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']

def encode(s):
    return [stoi.get(c, stoi['<unk>']) for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Load model
checkpoint = torch.load('out/ckpt.pt', map_location='cpu')
model_args = checkpoint['model_args']
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.load_state_dict(checkpoint['model'])
model.eval()
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Define generate() function
def generate(model, prompt, max_new_tokens=100, temperature=1.0, top_k=None):
    device = next(model.parameters()).device
    model.eval()

    idx = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

    return decode(idx[0].tolist())

# --- New! --- Conversation history buffer
conversation = ""

# Chat loop
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['exit', 'quit']:
        break

    # Add user input to conversation
    conversation += f"You: {user_input}\nBot:"

    # Generate response
    response = generate(model, conversation, max_new_tokens=100)

    # Cut off the model's text after it finishes its sentence (optional)
    response = response.split('\nYou:')[0].strip()

    print(f"Bot: {response}")

    # Add bot response to conversation
    conversation += f" {response}\n"
 """