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

    #sample data for computing attention scores
    inputs2 = torch.tensor(
        [[0.43, 0.15, 0.89],  [0.55, 0.87, 0.66],
         [0.57, 0.85, 0.64],
         [0.22, 0.58, 0.33],  [0.77, 0.25, 0.10], [0.05, 0.80, 0.55]] )
    query = inputs2[1]
    attn_scores_2 = torch.empty(inputs2.shape[0])
    for i, x_i in enumerate(inputs2):
        attn_scores_2[i] = torch.dot(x_i, query)

    #naive implementation of converting attention score to weights by normalization
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()


    #using torch softmax for normalization
    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)


    query = inputs2[1]
    context_vec_2 = torch.zeros(query.shape)
    for i, x_i in enumerate(inputs2):
        context_vec_2 += attn_weights_2[i] * x_i
    print("context_vec_2", context_vec_2)

