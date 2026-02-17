# LLM-from-scratch
Creating an LLM from scratch following the book " Build a Large Language Model (From Scratch)" by Sebastian Raschka

Things done so far 
**CHAPTER 1 & 2**
1. Get a raw text file
2. Convert to tokens using byte-pair encoding from OpenAI's open source model
3. Convert tokens to embeddings using pytorch
4. Add positional information to create final embeddings for input to LLM

**CHAPTER 3**
1. calculate attention scores for the vector embeddings
2. convert attention scores to normalized weights
3. use the weights to convert the vectors to context vectors

