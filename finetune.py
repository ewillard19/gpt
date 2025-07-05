

dataset_ = load_dataset("daily_dialog")
texts = dataset_["train"]["dialog"]

def get_corpus(texts):

    corpus = []
    for dialog in texts:
        for sentence in dialog:
            if sentence.strip():
                corpus.append(sentence.strip().lower())

    def clean(text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)  # supprime ponctuation
        text = re.sub(r"\s+", " ", text).strip()
        return text

    corpus = [clean(s) for s in corpus if len(s.strip()) > 0]


    tokens = set(" ".join(corpus).split())
    vocab = {word: i+1 for i, word in enumerate(tokens)}  # +1 pour réserver 0 = padding
    vocab["<PAD>"] = 0
    inv_vocab = {i: w for w, i in vocab.items()}

    encoded_corpus = []
    for lines in corpus:
        encoded_corpus.append([vocab[word] for word in lines.split()])
    
    return encoded_corpus, vocab, inv_vocab, tokens
