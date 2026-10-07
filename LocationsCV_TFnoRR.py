exec(open("dependencies.py").read())
exec(open("transformerBlocksNoRR.py").read())

import concurrent.futures
np.random.seed(126)

# ============================================================================
# Leave-one-location-out CV for TF(-)RR: same experimental design as
# LocationsCV.py (TF(+)RR), but training TransformerNoRR -- no RR fit, no
# feature_weights/pooling_weights anywhere in this script. This is the
# ablation counterpart: everything about the CV loop (fold definition, taxa
# overlap reporting, training loop, output format) is kept as close to
# LocationsCV.py as possible so a TF(+)RR vs TF(-)RR comparison isolates the
# RR-injection effect rather than incidental procedural differences.
#
# Two small fixes relative to the current LocationsCV.py that were worth
# making in a new file rather than carrying forward: (1) each fold's training
# loop here trains on the actual training-fold loader and evaluates on the
# inner validation split -- LocationsCV.py currently trains AND "validates"
# on the same inner valid_loader each epoch (the train_loader it builds is
# never actually used in the training loop), which likely wasn't intended;
# (2) `criterion` is declared global here so evaluateModel() (which reads a
# module-level `criterion`) can actually see it -- as written, LocationsCV.py
# would hit a NameError there since its `criterion` is only ever a local
# variable inside train_fold. Both are flagged separately for LocationsCV.py
# too, since they'd affect the TF(+)RR numbers in the same way.
# ============================================================================

def train_fold(loc, fold_num, data, unique):
    global criterion

    test = data[data["loc"] == loc]
    train = data[data["loc"] != loc]
    xTrain = train.drop(["Unnamed: 0", "x", "taxa", "loc", "pheno"], axis=1)
    yTrain = train["pheno"]
    xTest = test.drop(["Unnamed: 0", "x", "taxa", "loc", "pheno"], axis=1)
    yTest = test["pheno"]

    # Same germplasm-overlap reporting as LocationsCV.py, for consistency
    # across all CV scripts in this comparison.
    train_taxa = set(train["taxa"])
    test_taxa = set(test["taxa"])
    shared_taxa = train_taxa & test_taxa
    print(
        f"[TF(-)RR] Fold (held-out loc={loc}): {len(test_taxa)} taxa in test, "
        f"{len(shared_taxa)} of them ({len(shared_taxa)/max(len(test_taxa),1):.1%}) "
        f"also appear in the training set (other locations)."
    )

    batch_size = 15

    xTrain = pd.DataFrame(xTrain)
    yTrain = pd.DataFrame(yTrain)
    xTest = pd.DataFrame(xTest)
    yTest = pd.DataFrame(yTest)

    train_loader, test_loader = createTensors(xTrain, xTest, yTrain, yTest)

    # Model definition -- same hyperparameters as the TF(+)RR LocationsCV.py
    # run, so d_model/num_heads/num_layers/etc. aren't a confound either.
    src_vocab_size = int(unique)
    tgt_vocab_size = 1
    d_model = 300
    d_ff = 150
    num_heads = 5
    num_layers = 5
    max_seq_length = X.shape[1]
    dropout = 0.05
    criterion = lambda estimations, batch_y: focalLoss(beta=0.5, gamma=1, batch_y=batch_y, estimations=estimations)

    transformer = TransformerNoRR(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

    optimizer = torch.optim.SGD(transformer.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    transformer.apply(initialize_attention_weights)

    loss_values, val_losses, val_accuracies = [], [], []

    # Inner train/validation split carved out of the training fold -- the
    # held-out location (xTest/yTest) is never touched here.
    xTrainInner, xValid, yTrainInner, yValid = train_test_split(xTrain, yTrain, test_size=0.33, shuffle=True)

    train_inner_x = torch.tensor(xTrainInner.values, dtype=torch.long)
    train_inner_y = torch.tensor(yTrainInner.values, dtype=torch.float32)
    train_inner_loader = DataLoader(TensorDataset(train_inner_x, train_inner_y), batch_size=batch_size, shuffle=False, drop_last=True)

    x_valid_tensor = torch.tensor(xValid.values, dtype=torch.long)
    y_valid_tensor = torch.tensor(yValid.values, dtype=torch.float32)
    valid_loader = DataLoader(TensorDataset(x_valid_tensor, y_valid_tensor), batch_size=batch_size, shuffle=False, drop_last=True)

    # Training loop
    for epoch in range(40):
        transformer.train()
        train_loss = 0

        for batch_x, batch_y in train_inner_loader:
            batch_y = batch_y.squeeze(-1)
            optimizer.zero_grad()
            estimations = transformer(batch_x)
            loss = criterion(estimations, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_inner_loader)
        loss_values.append(avg_train_loss)

        transformer.eval()
        val_loss = 0
        predictions, true_vals = [], []

        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_y = batch_y.squeeze(-1)
                preds = transformer(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item()
                predictions.extend(preds.numpy())
                true_vals.extend(batch_y.numpy())

        avg_val_loss = val_loss / len(valid_loader)
        val_losses.append(avg_val_loss)
        val_accuracy, _ = pearsonr(predictions, true_vals)
        val_accuracies.append(val_accuracy)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"[TF(-)RR] Fold {fold_num} (loc={loc}), Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | LR: {current_lr:.6f}")

    loss_csv_file = f"SY_optim_loss_fold{fold_num}_CDBN_Locations_TFnoRR.csv"
    with open(loss_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Validation Accuracy"])
        for i in range(len(loss_values)):
            writer.writerow([i + 1, loss_values[i], val_losses[i], val_accuracies[i]])

    # Final test on the held-out location
    fold_result, values = evaluateModel(transformer, test_loader)
    values = pd.DataFrame(values)
    values.to_csv(f"pred_true_SY_BLB_{loc}_TFnoRR.csv")

    print(f"[TF(-)RR] Finished fold {fold_num} (loc={loc})")

    return loc, fold_result


# Load and preprocess the data -- same input file as LocationsCV.py
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
final_df.to_csv("SY_CV_Accuracies_CDBN_Locations_TFnoRR.csv", index=False)
