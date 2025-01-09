import torch
import torch.nn as nn
import math

# Define a class for image embeddings
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        """
        Initialize the embedding layer for images.
        
        Parameters:
        d_model: int - Dimension of the model embeddings.
        vocab_size: int - Size of the vocabulary or the number of unique tokens.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embedding layer to map input tokens to d_model-dimensional vectors
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        """
        Forward pass for the image embeddings.
        
        Parameters:
        x: Tensor - Input tensor containing token indices.
        
        Returns:
        Tensor - Scaled embeddings for the input tokens.
        """
        # Scale the embeddings by sqrt(d_model) for stability in transformer models
        return self.embedding(x) * math.sqrt(self.d_model)


# Define a class for positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        """
        Initialize the positional encoding layer.
        
        Parameters:
        d_model: int - Dimension of the model embeddings.
        seq_len: int - Maximum sequence length.
        dropout: float - Dropout rate to apply to the positional encodings.
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Dropout layer to regularize positional encodings
        self.dropout = nn.Dropout(dropout)
        
        # Create a positional encoding matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        
        # Create a vector of positions (0 to seq_len-1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # Shape: (seq_len, 1)
        
        # Calculate the divisors for the positional encoding formula
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices (0, 2, 4, ...)
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices (1, 3, 5, ...)
        
        # Add a batch dimension (1, seq_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register `pe` as a buffer to ensure it is not updated during backpropagation
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Forward pass for adding positional encodings.
        
        Parameters:
        x: Tensor - Input tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
        Tensor - Input tensor with positional encodings added, followed by dropout.
        """
        # Add positional encoding (no gradients required for `pe`)
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        
        # Apply dropout for regularization
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        """
        Initialize the Layer Normalization module.
        
        Parameters:
        eps: float - A small constant added to the denominator for numerical stability (default is 1e-6).
        """
        super().__init__()
        
        # Small constant for numerical stability
        self.eps = eps
        
        # Learnable scaling parameter (initialized to 1), multiplicative
        self.alpha = nn.Parameter(torch.ones(1))
        
        # Learnable bias parameter (initialized to 0), additive
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """
        Forward pass for layer normalization.
        
        Parameters:
        x: Tensor - Input tensor of shape (..., feature_dim), where the last dimension is normalized.
        
        Returns:
        Tensor - The normalized tensor with the same shape as input.
        """
        # Compute the mean across the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        
        # Compute the standard deviation across the last dimension
        std = x.std(dim=-1, keepdim=True)
        
        # Normalize the input and apply learnable scale (alpha) and bias
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """
        Initialize the feed-forward block.
        
        Parameters:
        d_model: int - The input and output dimensionality of the model.
        d_ff: int - The hidden dimensionality of the feed-forward network.
        dropout: float - Dropout rate for regularization.
        """
        super().__init__()
        
        # First linear layer (projection from d_model to d_ff)
        self.Linear_1 = nn.Linear(d_model, d_ff)  # Weights W1 and Bias B1
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Second linear layer (projection back from d_ff to d_model)
        self.Linear_2 = nn.Linear(d_ff, d_model)  # Weights W2 and Bias B2
        
    def forward(self, x):
        """
        Forward pass for the feed-forward block.
        
        Parameters:
        x: Tensor - Input tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
        Tensor - Output tensor of the same shape as input.
        """
        # Apply the first linear transformation: (batch, seq_len, d_model) -> (batch, seq_len, d_ff)
        x = self.Linear_1(x)
        
        # Apply ReLU activation to introduce non-linearity
        x = torch.relu(x)
        
        # Apply dropout for regularization
        x = self.dropout(x)
        
        # Apply the second linear transformation: (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        x = self.Linear_2(x)
        
        return x

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        """
        Initialize the multi-head attention block.
        
        Parameters:
        d_model: int - Total dimensionality of the input and output.
        h: int - Number of attention heads.
        dropout: float - Dropout rate for regularization.
        """
        super().__init__()
        self.h = h
        self.d_model = d_model
        
        # Ensure d_model is divisible by the number of heads
        assert d_model % h == 0, "d_model must be divisible by the number of heads (h)"
        
        # Dimensionality per head
        self.d_k = d_model // h
        
        # Linear layers for query, key, and value projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Output linear layer
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod    
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Compute scaled dot-product attention.
        
        Parameters:
        query: Tensor - Query tensor of shape (batch, h, seq_len, d_k).
        key: Tensor - Key tensor of shape (batch, h, seq_len, d_k).
        value: Tensor - Value tensor of shape (batch, h, seq_len, d_k).
        mask: Tensor or None - Mask tensor of shape (batch, 1, seq_len, seq_len) or None.
        dropout: nn.Dropout - Dropout layer.
        
        Returns:
        Tensor - Attention-weighted values.
        Tensor - Attention scores.
        """
        # Scale query by the square root of d_k
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)  # Shape: (batch, h, seq_len, seq_len)
        
        # Apply mask (if provided) by setting masked positions to a very low value
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention probabilities
        attention_scores = attention_scores.softmax(dim=-1)  # Shape: (batch, h, seq_len, seq_len)
        
        # Apply dropout (if specified)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # Compute the weighted sum of values
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        """
        Forward pass for the multi-head attention block.
        
        Parameters:
        q: Tensor - Query tensor of shape (batch, seq_len, d_model).
        k: Tensor - Key tensor of shape (batch, seq_len, d_model).
        v: Tensor - Value tensor of shape (batch, seq_len, d_model).
        mask: Tensor or None - Mask tensor of shape (batch, 1, seq_len, seq_len) or None.
        
        Returns:
        Tensor - Output tensor of shape (batch, seq_len, d_model).
        """
        # Linear projections for query, key, and value
        query = self.w_q(q)  # Shape: (batch, seq_len, d_model)
        key = self.w_k(k)    # Shape: (batch, seq_len, d_model)
        value = self.w_v(v)  # Shape: (batch, seq_len, d_model)
        
        # Reshape and transpose for multi-head attention
        # Shape: (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        # Compute scaled dot-product attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Reshape the output back to (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        # Apply the final output linear layer
        return self.w_o(x)
    
    
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        """
        Initialize the ResidualConnection block.
        
        Parameters:
        dropout: float - Dropout rate for regularization.
        """
        super().__init__()
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        # Layer normalization to stabilize training
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        """
        Forward pass for the ResidualConnection block.
        
        Parameters:
        x: Tensor - Input tensor of shape (batch, seq_len, d_model).
        sublayer: Callable - A sublayer (e.g., feedforward block or multi-head attention block).
        
        Returns:
        Tensor - Output tensor with residual connection and normalization applied.
        """
        # Apply layer normalization to the input
        normalized_x = self.norm(x)
        
        # Apply the sublayer to the normalized input and then dropout
        sublayer_output = self.dropout(sublayer(normalized_x))
        
        # Add the residual connection (input + sublayer output)
        return x + sublayer_output
    
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        """
        Represents a single block in the Transformer encoder.

        Parameters:
        self_attention_block: MultiHeadAttentionBlock - Self-attention mechanism.
        feed_forward_block: FeedForwardBlock - Fully connected feed-forward network.
        dropout: float - Dropout rate for regularization.
        """
        super().__init__()
        # Self-attention block
        self.self_attention_block = self_attention_block
        # Feed-forward block
        self.feed_forward_block = feed_forward_block
        # Two residual connections for self-attention and feed-forward blocks
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        """
        Forward pass for the EncoderBlock.

        Parameters:
        x: Tensor - Input tensor of shape (batch, seq_len, d_model).
        src_mask: Tensor - Mask for the input sequence, shape (batch, 1, seq_len, seq_len).

        Returns:
        Tensor - Output tensor after self-attention and feed-forward layers.
        """
        # Apply self-attention with residual connection
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # Apply feed-forward block with residual connection
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        """
        Represents the Transformer encoder, composed of multiple encoder blocks.

        Parameters:
        layers: nn.ModuleList - List of EncoderBlock layers.
        """
        super().__init__()
        # Sequence of encoder blocks
        self.layers = layers
        # Final layer normalization
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        """
        Forward pass for the Encoder.

        Parameters:
        x: Tensor - Input tensor of shape (batch, seq_len, d_model).
        mask: Tensor - Mask for the input sequence, shape (batch, 1, seq_len, seq_len).

        Returns:
        Tensor - Encoded output tensor after all layers and normalization.
        """
        # Pass the input through each encoder block
        for layer in self.layers:
            x = layer(x, mask)
        # Apply layer normalization to the final output
        return self.norm(x)
    
    
    
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        """
        Represents a single block in the Transformer decoder.

        Parameters:
        self_attention_block: MultiHeadAttentionBlock - Self-attention mechanism for the decoder.
        cross_attention_block: MultiHeadAttentionBlock - Cross-attention mechanism with the encoder output.
        feed_forward_block: FeedForwardBlock - Fully connected feed-forward network.
        dropout: float - Dropout rate for regularization.
        """
        super().__init__()
        # Decoder self-attention block
        self.self_attention_block = self_attention_block
        # Cross-attention block (attends to encoder output)
        self.cross_attention_block = cross_attention_block
        # Feed-forward block
        self.feed_forward_block = feed_forward_block
        # Three residual connections: for self-attention, cross-attention, and feed-forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass for the DecoderBlock.

        Parameters:
        x: Tensor - Input tensor of shape (batch, tgt_seq_len, d_model).
        encoder_output: Tensor - Output from the encoder of shape (batch, src_seq_len, d_model).
        src_mask: Tensor - Mask for the source sequence, shape (batch, 1, 1, src_seq_len).
        tgt_mask: Tensor - Mask for the target sequence, shape (batch, 1, tgt_seq_len, tgt_seq_len).

        Returns:
        Tensor - Output tensor after self-attention, cross-attention, and feed-forward layers.
        """
        # Self-attention with residual connection
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # Cross-attention with residual connection
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        # Feed-forward block with residual connection
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        """
        Represents the Transformer decoder, consisting of a stack of decoder blocks.

        Parameters:
        layers: nn.ModuleList - A list of `DecoderBlock` modules.
        """
        super().__init__()
        # Stack of decoder blocks
        self.layers = layers
        # Layer normalization applied at the end of the decoder
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass for the Decoder.

        Parameters:
        x: Tensor - Input tensor of shape (batch, tgt_seq_len, d_model).
        encoder_output: Tensor - Output from the encoder of shape (batch, src_seq_len, d_model).
        src_mask: Tensor - Mask for the source sequence, shape (batch, 1, 1, src_seq_len).
        tgt_mask: Tensor - Mask for the target sequence, shape (batch, 1, tgt_seq_len, tgt_seq_len).

        Returns:
        Tensor - Normalized output tensor after processing through all decoder blocks.
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        A projection layer that transforms the model's output to the vocabulary space
        and applies a log softmax for probability distribution over the vocabulary.

        Parameters:
        d_model: int - The dimensionality of the model's output features.
        vocab_size: int - The size of the target vocabulary.
        """
        super().__init__()
        # Linear layer that maps the d_model dimensions to the vocab_size
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        Forward pass for the ProjectionLayer.

        Parameters:
        x: Tensor - Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
        Tensor - Output tensor of shape (batch_size, seq_len, vocab_size) where
                 each element is a log-probability over the vocabulary.
        """
        # Apply the linear projection
        x = self.proj(x)
        # Apply log softmax along the last dimension (vocabulary dimension)
        return torch.log_softmax(x, dim=-1)
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        """
        Initializes a Transformer model with separate encoder and decoder components.

        Parameters:
        encoder: Encoder - The encoder part of the Transformer.
        decoder: Decoder - The decoder part of the Transformer.
        src_embed: InputEmbeddings - Embedding layer for source input.
        tgt_embed: InputEmbeddings - Embedding layer for target input.
        src_pos: PositionalEncoding - Positional encoding for the source input.
        tgt_pos: PositionalEncoding - Positional encoding for the target input.
        projection_layer: ProjectionLayer - Layer to project decoder output to the vocabulary size.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        """
        Encodes the source input sequence.

        Parameters:
        src: Tensor - Source sequence input.
        src_mask: Tensor - Mask for the source sequence.

        Returns:
        Tensor - Encoded source output.
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        """
        Decodes the target input sequence using the encoded source.

        Parameters:
        encoder_output: Tensor - Output from the encoder.
        src_mask: Tensor - Mask for the source sequence.
        tgt: Tensor - Target sequence input.
        tgt_mask: Tensor - Mask for the target sequence.

        Returns:
        Tensor - Decoded output.
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        """
        Projects the model's output to the vocabulary space.

        Parameters:
        x: Tensor - Input tensor from the decoder.

        Returns:
        Tensor - Projected tensor in vocabulary space.
        """
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    """
    Builds and initializes a Transformer model.

    Parameters:
    src_vocab_size: int - Size of the source vocabulary.
    tgt_vocab_size: int - Size of the target vocabulary.
    src_seq_len: int - Maximum sequence length for the source.
    tgt_seq_len: int - Maximum sequence length for the target.
    d_model: int - Dimensionality of the model's layers.
    N: int - Number of layers in both encoder and decoder.
    h: int - Number of heads in the multi-head attention mechanism.
    dropout: float - Dropout rate.
    d_ff: int - Dimensionality of the feed-forward layer.

    Returns:
    Transformer - An initialized Transformer model.
    """
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the Transformer model
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters using Xavier uniform initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


