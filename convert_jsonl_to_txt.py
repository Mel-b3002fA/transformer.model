import json

input_path = 'data/joi.jsonl'
output_path = 'data/joi.txt'

with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        line = line.strip()
        if not line or line.startswith("/*"):  
            continue
        try:
            obj = json.loads(line)
            input_text = obj.get("input")
            output_text = obj.get("output")
            if input_text and output_text:
                outfile.write(f"{input_text.strip()} {output_text.strip()}\n")
        except json.JSONDecodeError:
            print(f"⚠️ Skipping malformed line: {line}")


