import torch
import torch.nn as nn
import numpy as np


# Self-attention layer
class SelfAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, dropout_ratio):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, query, key, value, mask=None):
        key_dim = key.size(-1)
        attn = torch.matmul(query / np.sqrt(key_dim), key.transpose(2, 3))
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(torch.softmax(attn, dim=-1))
        output = torch.matmul(attn, value)

        return output


# Multi-head attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout_ratio):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.self_attention = SelfAttention(dropout_ratio)
        # The number of heads
        self.num_heads = num_heads
        # The dimension of each head
        self.dim_per_head = embedding_dim // num_heads
        # The linear projections
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)

        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x, mask=None):
        # Apply the linear projections
        batch_size = x.size(0)
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)
        # Reshape the input
        query = query.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        # Calculate the attention
        scores = self.self_attention(query, key, value, mask)
        # Reshape the output
        output = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)
        # Apply the linear projection
        output = self.out(output)
        return output


# Transformer decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout_ratio):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout_ratio)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embedding_dim)
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.dropout2 = nn.Dropout(dropout_ratio)

    def forward(self, x, mask):
        x = x + self.dropout1(self.self_attention(self.norm1(x), mask))
        x = x + self.dropout2(self.feed_forward(self.norm2(x)))
        return x


# Decoder transformer
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, block_size, num_heads, num_layers, dropout_ratio):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, block_size, embedding_dim))
        # self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, num_heads, embedding_dim*4, dropout_ratio) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x, mask):
        B, T = x.size()
        # embed the source
        x_emb = self.embedding(x)
        # position embeddings
        # x_pos = self.position_embedding(torch.arange(x.shape[1], device=x.device))
        x_pos = self.position_embedding[:, :T, :]
        # add
        x = self.dropout(x_emb + x_pos)

        # Propagate through the layers
        for layer in self.layers:
            x = layer(x, mask)
        # Normalize
        x = self.norm(x)
        return x


# Transformers
class GPT_1(nn.Module):
    def __init__(self, vocab_size, block_size, embedding_dim, num_heads, num_layers, dropout_ratio):
        super(GPT_1, self).__init__()
        self.decoder = Decoder(vocab_size, embedding_dim, block_size, num_heads, num_layers, dropout_ratio)
        self.final_linear = nn.Linear(embedding_dim, vocab_size)
        self.apply(self._init_weights)
    
    def forward(self, x):
        mask = self._make_mask(x.shape[1], x.device)
        # Decoder forward pass
        output = self.decoder(x, mask)
        # Final linear layer
        output = self.final_linear(output)
        return output

    def _make_mask(self, block_size, device):
        subsequent_mask = (1 - torch.triu(torch.ones((1, block_size, block_size), device=device), diagonal=1)).bool()
        return subsequent_mask

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(0, 0.02)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm)):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()