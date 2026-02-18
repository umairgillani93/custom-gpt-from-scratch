import os
import sys 
import torch
import torch.nn as nn 

data = [
        ['I', 'like', 'tea'],
        ['Tea', 'is', 'fantastic'],
        ['Abhi', 'was', 'treated', 'awesomely']
        ]

vocab = {}

count = 0;
for val in data:
    for v in val:
        if v not in vocab.keys():
            vocab[v] = count
            count += 1


token_ids = [vocab[val] for d in data for val in d]

vocab_size = len(vocab)
emb_dim = 512
emb_layer = nn.Embedding(vocab_size, emb_dim)

input_tensor = torch.tensor(token_ids, dtype = int)

emb_res = emb_layer(input_tensor)
print(emb_res.shape)
print(type(emb_res))

