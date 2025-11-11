# The following code enables training and evaluation on "data_1000.npz"
# Assumes the .npz is in the current directory and contains:
# X (N, ...) and y (N, )
import numpy as np
import os
import torch
import wandb
import torch.optim as optim
from regress_lm import core
from regress_lm import rlm

from sklearn.model_selection import train_test_split

def make_text_from_X(X, func_name="custom", func_id=None):
    # X is already text, just convert to list if necessary
    # return list(X)
    return [str(x) for x in X]

def load_npz_data_and_split(npz_path="data_1000.npz", test_size=0.2, seed=42):
    data = np.load(npz_path)
    X = data["X"]          # (N, ...)
    y = data["y"][:, 0]          # (N,)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return X_train, y_train, X_test, y_test

def main_with_npz():
    wandb.init(
        project="regress_lm_custom",
        name="rlm_custom_data",
        config={
            "d_model": 256,
            "num_encoder_layers": 4,
            "num_decoder_layers": 4,
            "batch_size": 128,
            "num_epochs": 200,
            "num_samples": 16,
            "npz_data": "results/N12_L50.0/R20_Trandom_MIN300_MAX600/data/preprocessed/data_1000.npz",
            "test_size": 0.2,
        }
    )
    config = wandb.config
    d_model = config.d_model
    num_encoder_layers = config.num_encoder_layers
    num_decoder_layers = config.num_decoder_layers
    batch_size = config.batch_size
    num_epochs = config.num_epochs
    num_samples = config.num_samples
    npz_data = config.npz_data
    test_size = config.test_size

    X_train, y_train, X_test, y_test = load_npz_data_and_split(
        npz_data, test_size=test_size
    )

    # Build training and test examples
    train_texts = make_text_from_X(X_train, func_name="custom", func_id=0)
    train_examples = [core.Example(text, round(float(y), 4)) for text, y in zip(train_texts, y_train)]
    test_texts = make_text_from_X(X_test, func_name="custom", func_id=None)
    test_examples = [core.ExampleInput(text) for text in test_texts]
    y_test_true = y_test

    print(f"Loaded npz data: X_train {X_train.shape}, X_test {X_test.shape}")

    # Set up model
    model = rlm.RegressLM.from_scratch(
        max_input_len=1024,
        d_model=d_model,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        compile_model=False,
    )
    optimizer = optim.Adafactor(
        filter(lambda p: p.requires_grad, model.model.parameters()),
        lr=5e-4,
    )

    num_train = len(train_examples)
    num_batches = (num_train + batch_size - 1) // batch_size

    for epoch in range(num_epochs):
        # Shuffle the training data at the outset of each epoch
        perm = np.random.permutation(num_train)
        avg_loss = 0.0
        for batch_idx in range(num_batches):
            batch_indices = perm[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            batch_examples = [train_examples[i] for i in batch_indices]
            print(batch_examples[:4])
            print(kyle)
            optimizer.zero_grad()
            tensor_examples = model.model.converter.convert_examples(batch_examples)
            loss, _ = model.model.compute_losses_and_metrics(tensor_examples)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        avg_loss /= num_batches

        # After each epoch, evaluate RMSE on test set using batch_size for batching
        pred_samples = []
        for i in range(0, len(test_examples), batch_size):
            batch_examples = test_examples[i : i + batch_size]
            batch_samples = model.sample(batch_examples, num_samples=num_samples)
            pred_samples.append(batch_samples)
        pred_ys_mean = np.concatenate(
            [np.stack(samples).mean(axis=1) for samples in pred_samples], axis=0
        )
        pred_ys_mean = np.squeeze(pred_ys_mean, axis=-1)
        true_ys = y_test_true.flatten()
        rmse = np.sqrt(np.mean((pred_ys_mean - true_ys) ** 2))
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Test RMSE: {rmse:.4f}")
        wandb.log({"epoch": epoch, "train/avg_loss": avg_loss, "eval/test_rmse": rmse})

    # Save model
    if not os.path.exists("models"):
        os.makedirs("models")
    model_path = f"models/rlm_custom_{num_epochs}.pt"
    torch.save(model.model.state_dict(), model_path)
    wandb.save(model_path)

    # Reload model and final test RMSE
    model = rlm.RegressLM.from_scratch(
        max_input_len=1024,
        d_model=d_model,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        compile_model=False,
    )
    model.model.load_state_dict(torch.load(model_path))
    # Test on all test samples individually, no batching
    pred_samples = []
    for example in train_examples + test_examples:
        # Get num_samples predictions for this example
        samples = model.sample([example], num_samples=16)  # shape: (16, 1)
        pred_samples.append(samples)
    pred_samples = [np.array(s).reshape(-1) for s in pred_samples]  # shape: (test_size, 16)
    pred_ys_mean = np.array([s.mean() for s in pred_samples])
    print("Test predictions (first 5):", pred_ys_mean[:5])
    # true_ys = y_test_true.flatten()
    true_ys = np.concatenate([y_train, y_test_true])
    print("Test true values (first 5):", true_ys[:5])
    rmse = np.sqrt(np.mean((pred_ys_mean - true_ys) ** 2))
    print(f"Test RMSE: {rmse:.4f}")
    wandb.log({"final/test_rmse": rmse})

    # Save prediction results for all test set to npz
    np.savez(
        "all_preds.npz",
        pred_ys_mean=pred_ys_mean,
        true_ys=true_ys,
    )

if __name__ == "__main__":
    main_with_npz()
