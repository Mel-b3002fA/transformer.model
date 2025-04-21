from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=10000)

files = ["./data/iris.txt"]
tokenizer.train(files, trainer)
tokenizer.save("tokenizer.json")
