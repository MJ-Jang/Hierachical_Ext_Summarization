from torch import nn
import torch.nn.functional as F
import torch
import math
import copy


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class TransformerSentEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, max_len):
        super().__init__()
        self.N = N
        self.d_model = d_model
        
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, max_len)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    
    def forward(self, src, mask, sent_len):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        x = self.norm(x)
        #x = x.mean(dim=1)
        
        # mean by sequence length
        mean_x = []
        for i in range(len(x)):
            if sent_len[i] > 0:
                tmp_x = x[i][:int(sent_len[i]),:].mean(dim=0)
            elif sent_len[i] == 0:
                tmp_x = x[i].mean(dim=0)
            mean_x.append(tmp_x)
        mean_x = torch.stack(mean_x)
        return mean_x
    
    
class TransformerDocEncoder(nn.Module):
    def __init__(self, d_model, N, heads, max_len):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(d_model, max_len)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, x, mask):
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 3) Summarizer
class Classifier(nn.Module):
    def __init__(self, d_model, class_num, d_ff=512, dropout=0.5):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, class_num)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

    
class HierSumTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, d_model, N, heads, max_sent_len, max_doc_len):
        super(HierSumTransformer, self).__init__()

        self.d_model = d_model
        self.max_sent_len = max_sent_len

        self.SentEncoder = TransformerSentEncoder(vocab_size, d_model, N, heads, max_sent_len)
        self.DocEncoder = TransformerDocEncoder(d_model, N, heads, max_doc_len)
        self.classifier = Classifier(d_model, 2, d_ff=256)

    def forward(self, input, sent_mask, doc_mask, sent_len):
        input = input.transpose(1,0)
        sent_mask = sent_mask.transpose(1,0)
        sent_len = sent_len.transpose(1,0)
        output_list = []
        for i, m, l in zip(input, sent_mask, sent_len):
            output = self.SentEncoder(i, m, l)
            output_list.append(output)

        output = torch.stack(output_list).transpose(1,0)
        output = self.DocEncoder(output, doc_mask)
        logits = self.classifier(output)
        return logits
