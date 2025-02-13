import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) # we use nn's Embedding thing to map vocab id to the embedding
    
    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_model) # paper says to multiply weights of embeddings by sqrt(d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model) # 0 matrix
        # [[0, 0, 0, ...d_model times],
        #   .
        #   .  
        #   seq_len times ]
        # we use the position encoding using sins and cosine formula
        # in even positions, it is sin, odd poistions it is cosine
        
        #create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # this is like one-d tensor in this format - 
        #
        # [[0], [1], [2], [3], [4], ... [seq_len-1]]
        #
        #
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000)/d_model))
        #this is the div_term which is just a one-d tensor in this format for every position
        # Apply the sin to the even positions
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)

        #add new dimension with unsquueze
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
######################################ENCODER STARTS HERE###########################################
    
class LayerNormalization(nn.Module):
    #this calculates mean and variance of the sentence embeddings after position and is adjust by two linear parameters (weights and bias)
    #takes in epsilon as parameter
    # x_cap = (x_j-u_j)/root(sigma_j^2+epsilon)
    # we need epsilon because if sigma_j^2 is close to zero, then we will get very large value(need to avoid div by zero)
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.ones(1)) # makes it learnable parameter
        self.bias = nn.Parameter(torch.zeros(1)) # added parameter (learnable as well)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True) # (batch, seq_len, 1) (calculate mean for last dimension)
        # keep dimesnion removes the dimesnion, but we want to keep it
        std = x.std(dim = -1, keepdim=True)
        return self.alpha*(x-mean)/(std+self.eps) + self.bias # this output is actually (batch, seq_len, d_model) which solves my doubt on dimension mismatch
    
class FeedForward(nn.Module):
    #this is used after add and norm
    #this is applied to every position seperately and identically, consists of two linear transformation with ReLU in between
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        # we go from d_model to d_ff and then back to d_model
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and b1, bias enabled by default
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and b2, here also bias enabled by default
    
    def forward(self, x):
        #input is x: (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttention(nn.Module):
    # before add and norm and after positions embeddings
    # take in query key value which are basicallly duplicates of the positional embeddings (seq_len, d_model)
    # we multiply query by W_q, key by W_k and value by W_v and get Q', K', V'
    # then split Q', K', V' into h heads of dimension d_model/h
    # each head access to full sentence, but it's like part of the embedding only (split across embedding dimension)
    # then we apply attention(q,k,v) = softmax(q.k^T/sqrt(d_k))v for all heads
    # then we concatenate all the heads and apply linear transformation (learnable parameters)
    # then we get multi head attention output as (seq, d_model)

    #above dimensions were for theory we did not include multi sentence example..so there will be one more thing called
    # batch, so dimesnion is (batch, seq, d_model)

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        # check if d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"
        # d_k = d_model/h
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model) 
        # we don;t use the same for all as they can be learned seperately
        self.w_o = nn.Linear(d_model, d_model)
        #d_v and d_k are same, it is just called d_v because last multiplication is of v in attention
        self.dropout = nn.Dropout(dropout)
    #create function to calculate attention
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = torch.matmul(query, key.transpose(-2, -1))/(math.sqrt(d_k))
        #now apply softmax and multiply by the value matrix 
        if mask is not None:
            #put things (future words) and other stuff to -inf which is -1e9
            attention_scores = attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            #attention scores is just softmax(q.k^T/sqrt(d_k))
        return torch.matmul(attention_scores, value), attention_scores
        
    def forward(self, q, k, v, mask=None):
        #if we don't want some words to interact with other words, we use mask
        # like, we don't want future words to interact with past words
        query = self.w_q(q) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k) [this after transpose]
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) #why do we transpose?, we make the h as the second dimension
        #do the same for key and value
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        # x is of shape (bathc, h, seq_len, d_k)
        # we change it to (batch, seq_len, h, d_k) -> (batch, seq_len, h*d_k=d_model)
        #contiguous transposes the tensor and makes it contiguous in memory
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h*self.d_k)
        # (batch, seq_len, d_model)->(btach, seq_len, d_model)
        return self.w_o(x)
    

# Now let's build the skip connections
class ResidualConnection(nn.Module):
    def __init__(self, dropout) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        #sublayer is previous layer
        # return x + self.dropout(sublayer(self.norm(x)))
        #most papers apply sublayer first and then the normalization
        #in umar jamil's video, it did not make sense to me so i am following what the paper did
        #return x + self.dropout(self.norm(sublayer(x)))
        # ok i realized doing it the paper's way meant all inputs were not b/w 0 and 1 to the layers for matmul...so sticking back to umar jamil
        return x + self.dropout(sublayer(self.norm(x)))

#now finally create the encoder block

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # we apply mask so that padding doesn't affect our input
        # first step is send x to mha and the add and norm
        x = self.residual_connections[0](x, lambda: self.self_attention_block(x, x, x, src_mask)) # calls forward function of the mha
        x = self.residual_connections[1](x, lambda: self.feed_forward_block(x))
        return x
    
# we can have many encoders, idk why many and why one is not sufficient

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

#This finishes out encoder..the input we should send to the encoder is the positional embeddings
######################################ENCODER ENDS HERE###########################################
######################################DECODER STARTS HERE###########################################

# we can just use the classes we created earlier to make the decoder

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # we have two kinds of attention which is self and not self 
        #first one is self and other one is cross attention
        # takes the key and value from the encoder
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # we have two masks, one from encoder and decoder -> source and target
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        #now claculate cross attention
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.norm = LayerNormalization()
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
#decoder is done
######################################DECODER ENDS HERE###########################################
#now we project the embedding into the vocabulary

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        #(batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,  src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_embed = tgt_embed
        self.projection = projection_layer
        self.tgt_pos = tgt_pos

    # we keep them seperate to visualize results
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    def decode(self,encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    def project(self, x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048):
    #create embedding layers
    src_embedding = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    #then positional encoding
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    #create another positionalencoding even thought its same forunderstanding purpose
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # create encoder
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    #create decoder block
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    #create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    #create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    #create transformer
    transformer = Transformer(encoder, decoder, src_embedding, tgt_embed, src_pos, tgt_pos, projection_layer)

    #initialize the transformer parameters (using xavier)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer