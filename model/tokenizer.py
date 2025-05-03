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

