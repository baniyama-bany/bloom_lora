
import torch
device = torch.device("cuda")
from lora import Linear, LoRALayer
import transformers
from tqdm import tqdm
from print_pram import print_trainable_parameters

model = transformers.AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m").to(device)
tokenizer = transformers.AutoTokenizer.from_pretrained("bigscience/bloom-560m")
for i in range(model.config.n_layer):
    lora_linear = Linear(model.config.hidden_size, model.config.hidden_size*4)
    lora_linear.weight = model.transformer.h[i].self_attention.query_key_value.weight
    lora_linear.bias = model.transformer.h[i].self_attention.query_key_value.bias
    model.transformer.h[i].self_attention.query_key_value = lora_linear


########################################
for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False

for m in model.modules():
    if isinstance(m, LoRALayer) and hasattr(m, 'bias') and m.bias is not None:
        m.bias.requires_grad = True
########################################

from dataset import Mydatasets
dataset = Mydatasets(tokenizer)
trainloader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True)

print_trainable_parameters(model)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

for i in range(4):
    for batch in tqdm(trainloader):
        for key in batch: batch[key] = batch[key].to(device)
        optimizer.zero_grad()
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
