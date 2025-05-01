""" from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=10000)

files = ["./data/iris.txt"]
tokenizer.train(files, trainer)
tokenizer.save("tokenizer.json")

class Tokenizer:
    def __init__(self):
        # Example setup; load actual vocab file or build from data
        self.itos = {i: chr(i) for i in range(256)}   # dummy: index to character
        self.stoi = {chr(i): i for i in range(256)}   # dummy: character to index

    def encode(self, text):
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, tokens):
        return ''.join([self.itos.get(t, '') for t in tokens])

    def get_itos(self):
        return self.itos

    def get_stoi(self):
        return self.stoi """



import pickle
from model.tokenizer import Tokenizer  # Adjust if your tokenizer file or class is named differently

tokenizer = Tokenizer()

# Try common tokenizer vocab properties
try:
    stoi = tokenizer.stoi
    itos = tokenizer.itos
    vocab_size = len(stoi)
except AttributeError:
    try:
        stoi = tokenizer.encoder
        itos = tokenizer.decoder
        vocab_size = len(stoi)
    except AttributeError:
        raise ValueError("Tokenizer must have 'stoi' and 'itos' or 'encoder' and 'decoder'.")

meta = {
    'vocab_size': vocab_size,
    'stoi': stoi,
    'itos': itos
}

with open('data/openwebtext/meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

print("✅ meta.pkl successfully written.")

