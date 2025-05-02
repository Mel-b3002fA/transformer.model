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
import re
from collections import Counter

class Tokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0

    def train(self, text, vocab_size=5000):
        words = re.findall(r'\w+|\S', text)
        counter = Counter(words)
        most_common = counter.most_common(vocab_size)
        self.itos = {i: word for i, (word, _) in enumerate(most_common)}
        self.stoi = {word: i for i, word in self.itos.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, text):
        words = re.findall(r'\w+|\S', text)
        return [self.stoi.get(word, self.vocab_size) for word in words]  # unknowns mapped to vocab_size

    def decode(self, tokens):
        return ' '.join(self.itos.get(token, '<unk>') for token in tokens)
    

    import pickle
from model.tokenizer import Tokenizer  # adjust if needed

tokenizer = Tokenizer()

meta = {
    'vocab_size': len(tokenizer.stoi),
    'stoi': tokenizer.stoi,
    'itos': tokenizer.itos
}

with open('data/openwebtext/meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

print("✅ meta.pkl successfully saved.")

