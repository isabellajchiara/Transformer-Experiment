
# ============================================================================
# TF(-)RR: the same transformer architecture as transformerBlocks.py, but
# with the RR-derived feature_weights / pooling_weights injection removed.
#
# This file is meant to be a clean ablation of transformerBlocks.py: every
# class/function here is identical to its TF(+)RR counterpart EXCEPT for the
# specific lines that touch feature_weights/pooling_weights, which are
# removed rather than replaced with some other weighting scheme. That's
# deliberate -- the point of TF(-)RR is to isolate the effect of the RR
# injection itself, so nothing else about the architecture, training loop,
# or quirks should differ between the two.
#
# One quirk carried over on purpose: scaled_dot_product_attention() below
# does not apply a softmax to attn_scores before the matmul with V, matching
# transformerBlocks.py exactly. That looks like it's probably not intended
# (standard scaled dot-product attention normalizes with softmax), but fixing
# it here and not in the TF(+)RR version would confound the RR-injection
# ablation with an unrelated architecture change. Flagging it rather than
# silently fixing it -- worth a decision from whoever owns the tensor
# methodology before either version changes.
# ============================================================================

# Multi-Head Attention (no RR feature weighting)
class MultiHeadAttentionNoRR(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttentionNoRR, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # No RR feature_weights reweighting here -- this is the only
        # substantive difference from MultiHeadAttention in transformerBlocks.py.
        output = torch.matmul(attn_scores, V)

        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V)
        attn_output = self.combine_heads(attn_output)
        output = self.W_o(attn_output)

        return output


# Position-Wise Feed Forward -- identical to transformerBlocks.py
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


# Positional Encoding -- identical to transformerBlocks.py
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# Encoder Layer (uses MultiHeadAttentionNoRR, no feature_weights to pass through)
class EncoderLayerNoRR(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayerNoRR, self).__init__()
        self.self_attn = MultiHeadAttentionNoRR(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


def initialize_attention_weights(module):
    if isinstance(module, MultiHeadAttentionNoRR):
        nn.init.xavier_uniform_(module.W_q.weight)
        nn.init.xavier_uniform_(module.W_k.weight)
        nn.init.xavier_uniform_(module.W_v.weight)
        nn.init.xavier_uniform_(module.W_o.weight)
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
    elif isinstance(module, nn.Embedding):
        nn.init.xavier_uniform_(module.weight)


def focalLoss(beta, gamma, batch_y, estimations):
    beta = beta
    gamma = gamma
    abs_error = torch.abs(estimations - batch_y)
    loss = (torch.tanh(beta * abs_error) ** gamma * abs_error)
    return loss.mean()


# Transformer Model -- no feature_weights/pooling_weights parameters at all,
# and plain (unweighted) mean pooling over the sequence instead of RR-weighted
# pooling.
class TransformerNoRR(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(TransformerNoRR, self).__init__()

        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayerNoRR(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))

        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(src_embedded)

        # plain mean pooling -- no RR pooling_weights to multiply in
        pooled_output = enc_output.mean(dim=1)
        output = self.fc(pooled_output)
        return output.squeeze(-1)


# NOTE: there is no getWeights() in this file. TF(-)RR never fits an RR model
# at all, so there is no alpha search, no RR coefficients, and therefore no
# way for the train/test independence issue fixed in transformerBlocks.py's
# getWeights() to apply here -- this architecture structurally has nothing to
# leak through.


def preprocess(data):
    data = data.sample(frac=1)
    X = data.drop(["Unnamed: 0", "0", "1", "2", "3"], axis=1)
    threshold = 0.01
    X = X.drop(X.std()[X.std() < threshold].index.values, axis=1)
    y = data["3"]
    unique = X.stack().nunique()
    return X, y, unique


def evaluateModel(model, loader):
    model.eval()
    predictions = []
    true_vals = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_y = batch_y.squeeze(-1)
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            predictions.extend(preds.numpy())
            true_vals.extend(batch_y.numpy())

    accuracy, _ = pearsonr(predictions, true_vals)
    values = np.column_stack((predictions, true_vals))
    values = pd.DataFrame(values)
    values.columns = ["pred", "true"]

    return accuracy, values


def createTensors(xTrain, X_test, yTrain, y_test):
    xTrain_tensor = torch.tensor(xTrain.values, dtype=torch.long)
    yTrain_tensor = torch.tensor(yTrain.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    train_dataset = TensorDataset(xTrain_tensor, yTrain_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader
