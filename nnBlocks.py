
# ============================================================================
# NN: a plain feedforward network (MLP) benchmark.
#
# Unlike the two transformer variants, this model takes the marker matrix
# as plain numeric features (float), the same representation RR-BLUP uses --
# not token indices through an embedding layer. That keeps the comparison
# clean: RR is the linear model on these features, NN is a non-linear model
# on the SAME features, and the two transformer variants are the
# attention-based alternative (with and without RR injected into it).
# ============================================================================

class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super(FeedForwardNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def initialize_nn_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def focalLoss(beta, gamma, batch_y, estimations):
    beta = beta
    gamma = gamma
    abs_error = torch.abs(estimations - batch_y)
    loss = (torch.tanh(beta * abs_error) ** gamma * abs_error)
    return loss.mean()


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


def createTensorsFloat(xTrain, X_test, yTrain, y_test):
    # Float features (not long/token indices) -- the NN reads raw marker
    # values directly, the same representation RR-BLUP uses, so the two are
    # directly comparable; only the transformer variants use an embedding.
    xTrain_tensor = torch.tensor(xTrain.values, dtype=torch.float32)
    yTrain_tensor = torch.tensor(yTrain.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    train_dataset = TensorDataset(xTrain_tensor, yTrain_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader
