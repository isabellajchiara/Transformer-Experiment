exec(open("dependencies.py").read())
exec(open("nnBlocks.py").read())

import concurrent.futures
np.random.seed(126)

# ============================================================================
# Leave-one-location-out CV for the NN baseline (plain feedforward MLP on
# raw marker values -- no RR, no attention). Same experimental design
# (fold definition, taxa-overlap reporting, output format) as
# LocationsCV.py / LocationsCV_TFnoRR.py so all four methods' results
# (RR, TF(+)RR, TF(-)RR, NN) line up for direct comparison.
#
# Optimizer note: this uses Adam rather than the SGD(lr=0.0001) the
# transformer scripts use. That's a deliberate choice, not an oversight --
# the NN is a separate model class being tuned on its own terms, not an
# ablation of the transformer, so there's no requirement to match its
# optimizer. If you'd rather hold the optimizer/LR constant across every
# method for the writeup, say so and this can be changed to match.
# ============================================================================

def train_fold(loc, fold_num, data, unique):
    global criterion, batch_size

    test = data[data["loc"] == loc]
    train = data[data["loc"] != loc]
    xTrain = train.drop(["Unnamed: 0", "x", "taxa", "loc", "pheno"], axis=1)
    yTrain = train["pheno"]
    xTest = test.drop(["Unnamed: 0", "x", "taxa", "loc", "pheno"], axis=1)
    yTest = test["pheno"]

    train_taxa = set(train["taxa"])
    test_taxa = set(test["taxa"])
    shared_taxa = train_taxa & test_taxa
    print(
        f"[NN] Fold (held-out loc={loc}): {len(test_taxa)} taxa in test, "
        f"{len(shared_taxa)} of them ({len(shared_taxa)/max(len(test_taxa),1):.1%}) "
        f"also appear in the training set (other locations)."
    )

    batch_size = 15

    xTrain = pd.DataFrame(xTrain)
    yTrain = pd.DataFrame(yTrain)
    xTest = pd.DataFrame(xTest)
    yTest = pd.DataFrame(yTest)

    train_loader, test_loader = createTensorsFloat(xTrain, xTest, yTrain, yTest)

    input_dim = X.shape[1]
    hidden_dims = [512, 128]
    dropout = 0.1
    criterion = lambda estimations, batch_y: focalLoss(beta=0.5, gamma=1, batch_y=batch_y, estimations=estimations)

    model = FeedForwardNN(input_dim, hidden_dims, dropout)
    model.apply(initialize_nn_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    loss_values, val_losses, val_accuracies = [], [], []

    # Inner train/validation split carved out of the training fold only --
    # same independence principle as the fixed getWeights()/TF(-)RR scripts.
    xTrainInner, xValid, yTrainInner, yValid = train_test_split(xTrain, yTrain, test_size=0.33, shuffle=True)

    train_inner_x = torch.tensor(xTrainInner.values, dtype=torch.float32)
    train_inner_y = torch.tensor(yTrainInner.values, dtype=torch.float32)
    train_inner_loader = DataLoader(TensorDataset(train_inner_x, train_inner_y), batch_size=batch_size, shuffle=False, drop_last=True)

    x_valid_tensor = torch.tensor(xValid.values, dtype=torch.float32)
    y_valid_tensor = torch.tensor(yValid.values, dtype=torch.float32)
    valid_loader = DataLoader(TensorDataset(x_valid_tensor, y_valid_tensor), batch_size=batch_size, shuffle=False, drop_last=True)

    for epoch in range(40):
        model.train()
        train_loss = 0

        for batch_x, batch_y in train_inner_loader:
            batch_y = batch_y.squeeze(-1)
            optimizer.zero_grad()
            estimations = model(batch_x)
            loss = criterion(estimations, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_inner_loader)
        loss_values.append(avg_train_loss)

        model.eval()
        val_loss = 0
        predictions, true_vals = [], []

        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_y = batch_y.squeeze(-1)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item()
                predictions.extend(preds.numpy())
                true_vals.extend(batch_y.numpy())

        avg_val_loss = val_loss / len(valid_loader)
        val_losses.append(avg_val_loss)
        val_accuracy, _ = pearsonr(predictions, true_vals)
        val_accuracies.append(val_accuracy)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"[NN] Fold {fold_num} (loc={loc}), Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | LR: {current_lr:.6f}")

    loss_csv_file = f"SY_optim_loss_fold{fold_num}_CDBN_Locations_NN.csv"
    with open(loss_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Validation Accuracy"])
        for i in range(len(loss_values)):
            writer.writerow([i + 1, loss_values[i], val_losses[i], val_accuracies[i]])

    fold_result, values = evaluateModel(model, test_loader)
    values = pd.DataFrame(values)
    values.to_csv(f"pred_true_SY_BLB_{loc}_NN.csv")

    print(f"[NN] Finished fold {fold_num} (loc={loc})")

    return loc, fold_result


data = pd.read_csv("fullDatasetSY_Updated.csv")
IDS = data["taxa"]

X = data.drop(["Unnamed: 0", "x", "taxa", "loc", "pheno"], axis=1)
unique = X.stack().nunique()

locations = data["loc"].unique()
cv_results = []

with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = {executor.submit(train_fold, loc, i + 1, data, unique): loc for i, loc in enumerate(locations)}
    for future in concurrent.futures.as_completed(futures):
        cv_results.append(future.result())

final_df = pd.DataFrame(cv_results, columns=["Location", "Accuracy"])
final_df.to_csv("SY_CV_Accuracies_CDBN_Locations_NN.csv", index=False)
