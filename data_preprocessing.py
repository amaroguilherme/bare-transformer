import random

from math import sqrt, sin, cos

class DataPreprocessor:
    def __init__(self, vocab=None, special_tokens=None, embedding_dim=512):
        self.vocab = vocab or {}
        self.special_tokens = special_tokens or {"PAD": 0, "UNK": 1, "SOS": 2, "EOS": 3}
        self.embedding_dim = embedding_dim
        self.embedding_matrix = None


    def _tokenize(self, text):
        return text.lower().split()


    def build_vocab(self, texts):
        idx = len(self.special_tokens)
        for text in texts:
            for token in self._tokenize(text):
                if token not in self.vocab:
                    self.vocab[token] = idx
                    idx += 1


    def encode(self, text):
        return [self.vocab.get(token, self.special_tokens["UNK"])
                for token in self._tokenize(text)]


    def decode(self, indices):
        inv_vocab = {v: k for k, v in self.vocab.items()}
        return [inv_vocab.get(i, "<UNK>") for i in indices]


    def build_embedding_matrix(self):
        vocab_size = len(self.vocab) + len(self.special_tokens)
        lim = 1/(sqrt(self.embedding_dim))
        self.embedding_matrix = [
            [random.uniform(-lim, lim) for _ in range(self.embedding_dim)]
                for _ in range(vocab_size)
        ]
        return self.embedding_matrix

    
    def calculate_positional_encoding(self, encoded_text):
        pe = []
        for pos in range(encoded_text):
            row = []
            for i in range(self.embedding_dim):
                if i % 2 == 0:
                    value = sin(pos / (10000 ** (i / self.embedding_dim)))
                else:
                    value = cos(pos / (10000 ** ((i - 1) / self.embedding_dim)))
                row.append(value)
            pe.append(row)
        return pe
    