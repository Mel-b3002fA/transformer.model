import json

input_path = 'data/joi.jsonl'
output_path = 'data/joi.txt'

with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        try:
            obj = json.loads(line)
            # Replace 'text' with your actual key
            text = obj.get("text") or obj.get("dialogue") or ""
            if text:
                fout.write(text.strip() + "\n")
        except json.JSONDecodeError:
            print("Skipping bad line:", line)
