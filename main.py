import re
from SimpleTokenizer import SimpleTokenizerV1
from importlib.metadata import version
import tiktoken



if __name__ == '__main__':
    # #PRE TRAINING WITH BOOK VERDICT
    # with open("the-verdict.txt", "r", encoding="utf-8") as f:
    #     raw_text = f.read()
    #     preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    # preprocessed = [item for item in preprocessed if item.strip()]
    # all_tokens = sorted(set(preprocessed)) #using set to remove duplicates
    #
    # all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    #
    # vocab_size = len(all_tokens)
    #
    # #PRE TRAINED VOCAB
    # vocab = {token:integer for integer, token in enumerate(all_tokens)}
    #
    # #TESTING PRE TRAINED VOCAB WITH NEW TEXT
    # tokenizer = SimpleTokenizerV1(vocab)
    text1 = "Conor wins this fight by knockout"
    text2 = "He is the new double champ"
    text = " <|endoftext|> ".join((text1, text2))

    tokenizerGPT = tiktoken.get_encoding("gpt2")
    integers = tokenizerGPT.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)
    strings = tokenizerGPT.decode(integers)
    print(strings)





