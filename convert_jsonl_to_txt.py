import json


input_path = 'data/joi.jsonl'  
output_path = 'data/joi.txt'  

with open(input_path, 'r', encoding='utf-8') as infile, \
     open(output_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        obj = json.loads(line)
        context = obj.get("context", "")
        response = obj.get("response", "")
        full_text = f"{context} {response}".strip()
        outfile.write(full_text + "\n")

with open(input_path, 'r', encoding='utf-8') as f:
    for i in range(5):  # Read first 5 lines
        line = f.readline()
        if not line:
            print("🚨 Reached end of file unexpectedly. Is it empty?")
            break
        obj = json.loads(line)
        print(f"\n🔍 Line {i+1}:")
        print(obj)
