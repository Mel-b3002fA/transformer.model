import pickle
import re  # Ensure that 're' is imported
from collections import Counter

class Tokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0

    def train(self, text, vocab_size=50257):  # Update default vocab_size to 50257
        words = re.findall(r'\w+|\S', text)  # Tokenize the text
        counter = Counter(words)  # Count occurrences of each token
        most_common = counter.most_common(vocab_size)  # Get the most common tokens up to vocab_size
        self.itos = {i: word for i, (word, _) in enumerate(most_common)}  # Create 'itos' dictionary
        self.stoi = {word: i for i, word in self.itos.items()}  # Create 'stoi' dictionary
        self.vocab_size = len(self.stoi)  # Set the vocab_size

    def encode(self, text):
        words = re.findall(r'\w+|\S', text)
        # Ensure unknown tokens are mapped to vocab_size (50257 or `vocab_size - 1`)
        return [self.stoi.get(word, self.vocab_size) for word in words]  # Ensure the vocab_size here is correct

    def decode(self, tokens):
        return ' '.join(self.itos.get(token, '<unk>') for token in tokens)  # Convert token IDs back to text
