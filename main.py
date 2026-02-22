import GPT_CONFIG
from DummyGPTModel import DummyGPTModel
from GPTDatasetV1 import create_dataloader_v1
from LayerNorm import LayerNorm
import tiktoken
import torch
import torch.nn as nn



if __name__ == '__main__':
    # #PRE TRAINING WITH BOOK VERDICT
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    max_length = 4 #constant value
    dataLoader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
    data_iter = iter(dataLoader)
    inputs, targets = next(data_iter)

    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)

    torch.manual_seed(123)
    batch_example = torch.randn(2,5)
    layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())
    out = layer(batch_example)


    torch.set_printoptions(sci_mode=False)

    model = DummyGPTModel(GPT_CONFIG.GPT_CONFIG_124M)
    logits = model(batch)

    ln = LayerNorm(emb_dim = 5)
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    print("Mean:\n", mean)
    print("Variance:\n", var)








