import re

from GPTDatasetV1 import create_dataloader_v1
from SimpleTokenizer import SimpleTokenizerV1
from importlib.metadata import version
import tiktoken
import torch



if __name__ == '__main__':
    # #PRE TRAINING WITH BOOK VERDICT
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    max_length = 4 #constant value
    dataLoader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
    data_iter = iter(dataLoader)
    inputs, targets = next(data_iter)
    vocab_size = 50257
    output_dim = 256 #number of dimsensions for the token embeddings
  #  torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim) #creating an embedding layer using pytorch
    token_embeddings = embedding_layer(inputs)
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    # positional embedding layer to keep track of position of the tokens
    pos_embedding = pos_embedding_layer(torch.arange(context_length))
    #final input embeddings
    input_embeddings = token_embeddings + pos_embedding




