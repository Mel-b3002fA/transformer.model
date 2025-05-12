from collections import Counter

with open("your_dataset.txt") as f:
    lines = f.readlines()

token_counts = Counter()
for line in lines:
    token_counts.update(line.strip().split())

print(token_counts.most_common(20))
