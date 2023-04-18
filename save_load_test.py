
import torch
device = torch.device("cuda")
from lora import Linear, LoRALayer
import transformers
from print_pram import print_trainable_parameters

model = transformers.AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m").to(device)
tokenizer = transformers.AutoTokenizer.from_pretrained("bigscience/bloom-560m")
for i in range(model.config.n_layer):
    lora_linear = Linear(model.config.hidden_size, model.config.hidden_size*4)
    lora_linear.weight = model.transformer.h[i].self_attention.query_key_value.weight
    lora_linear.bias = model.transformer.h[i].self_attention.query_key_value.bias
    model.transformer.h[i].self_attention.query_key_value = lora_linear


########## save
name = "lora_bloom_0419"
import os
if not os.path.exists(name):
    os.mkdir(name)
torch.save(model.state_dict(), os.path.join(name, "model.bin"))

########## load
model.load_state_dict(torch.load(os.path.join(name, "model.bin")))
