import re
from SimpleTokenizer import SimpleTokenizerV1


if __name__ == '__main__':

    #PRE TRAINING WITH BOOK VERDICT
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item for item in preprocessed if item.strip()]
    all_words = sorted(set(preprocessed)) #using set to remove duplicates
    vocab_size = len(all_words)
    #PRE TRAINED VOCAB
    vocab = {token:integer for integer, token in enumerate(all_words)}

    #TESTING PRE TRAINED VOCAB WITH NEW TEXT
    tokenizer = SimpleTokenizerV1(vocab)
    text = "Hello, Do you like tea?"
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))




