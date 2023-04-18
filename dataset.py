import json
import torch
import transformers

prompt = """Instruction:{INSTRUCTION}

Input:{INPUT}

Output:{OUTPUT}"""

class Mydatasets(torch.utils.data.Dataset):
    def __init__(self, tokenizer, path="alpaca_cleaned_ja.jsonl"):
        self.data = []
        for line in open(path, encoding="utf-8").readlines():
            try:
                self.data.append(json.loads(line))
            except:
                pass

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        input_text = prompt.replace("{INSTRUCTION}", d["instruction"]).replace("{INPUT}", d["input"]).replace("{OUTPUT}", d["output"])
        tokenized = self.tokenizer(input_text, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
        for key in tokenized: tokenized[key] = tokenized[key].squeeze()
        
        labels = tokenized["input_ids"].clone()
        labels[labels==self.tokenizer.pad_token_id] = -100
        tokenized["labels"] = labels
        return tokenized
    

# tokenizer = transformers.AutoTokenizer.from_pretrained("bigscience/bloom-560m")
# dataset = Mydatasets(tokenizer=tokenizer)
# print(dataset[0])
# print(len(dataset))
