"""Provides a detailed example of fine-tuning a TabPFNRegressor model.

This script demonstrates the complete workflow, including data loading and preparation
for the Bike Sharing Demand dataset, model configuration, the fine-tuning loop,
and performance evaluation for a regression task.
"""

from functools import partial

import numpy as np
import sklearn.datasets
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabpfn import TabPFNRegressor
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.utils import meta_dataset_collator

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
# 添加项目根目录到Python路径
DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(DIR_PATH)
from utils.early_stopping import AdaptiveES


def prepare_data(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads, subsets, and splits the California Housing dataset."""
    print("--- 1. Data Preparation ---")
    # # Fetch Ames housing data from OpenML
    # bike_sharing = sklearn.datasets.fetch_openml(
    #     data_id=44973, as_frame=True, parser="auto"
    # )

    # # Separate features (X) and target (y)
    # X_df = bike_sharing.data
    # y_df = bike_sharing.target

    DATA = pd.read_csv("/home/fit/zhangcs/WORK/lzx/TabDiff1/eval/report_runs/learnable_schedule/grid_stability/1886/samples.csv")
    X_df = DATA.iloc[:, :-1]
    y_df = DATA.iloc[:, -1]

    # Select only numeric features for simplicity
    X_numeric = X_df.select_dtypes(include=np.number)

    X_all, y_all = X_numeric.values, y_df.values

    rng = np.random.default_rng(config["random_seed"])
    num_samples_to_use = min(config["num_samples_to_use"], len(y_all))
    indices = rng.choice(np.arange(len(y_all)), size=num_samples_to_use, replace=False)
    X, y = X_all[indices], y_all[indices]

    splitter = partial(
        train_test_split,
        test_size=config["valid_set_ratio"],
        random_state=config["random_seed"],
    )
    X_train, X_test, y_train, y_test = splitter(X, y)

    print(
        f"Loaded and split data: {X_train.shape[0]} train, {X_test.shape[0]} test samples."
    )
    print("---------------------------\n")
    return X_train, X_test, y_train, y_test


def setup_regressor(config: dict) -> tuple[TabPFNRegressor, dict]:
    """Initializes the TabPFN regressor and its configuration."""
    print("--- 2. Model Setup ---")
    regressor_config = {
        "ignore_pretraining_limits": True,
        "device": config["device"],
        "n_estimators": 2,
        "random_state": config["random_seed"],
        "inference_precision": torch.float32,
    }
    regressor = TabPFNRegressor(
        **regressor_config, fit_mode="batched", differentiable_input=False,
        model_path="/home/fit/zhangcs/WORK/chenkq/project/ckpt/tabpfn-v2-regressor.ckpt"
    )

    print(f"Using device: {config['device']}")
    print("----------------------\n")
    return regressor, regressor_config


def evaluate_regressor(
    regressor: TabPFNRegressor,
    eval_config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float, float]:
    """Evaluates the regressor's performance on the test set."""
    eval_regressor = clone_model_for_evaluation(regressor, eval_config, TabPFNRegressor)
    eval_regressor.fit(X_train, y_train)

    try:
        predictions = eval_regressor.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        mse, mae, r2 = np.nan, np.nan, np.nan

    return mse, mae, r2


def plot_results(plot_dict: dict):
    # Create dual y-axis plot
    _, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    plot_dict['epoch'] = range(len(plot_dict['train_loss']))
    
    # Plot training loss (left y-axis)
    color1 = 'tab:blue'
    ax1.plot(plot_dict['epoch'], plot_dict['train_loss'], color=color1, linewidth=3, label='Training Loss')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Training Loss', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Plot validation loss (right y-axis)
    color2 = 'tab:orange'
    ax2.plot(plot_dict['epoch'], plot_dict['validation_loss'], color=color2, linewidth=3, label='Validation Loss')
    ax2.set_ylabel('Validation Loss', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add best step marker
    best_step = np.argmax(plot_dict['validation_loss'])
    ax1.axvline(x=best_step, color="red", linestyle="--", linewidth=2, label="Best Step")
    ax2.axhline(y=plot_dict['initial_validation_loss'], color="green", linestyle="--", linewidth=2, label="Initial Validation Loss")
    ax1.text(best_step + 0.5, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.1, 
             f"Best Epoch R2: {plot_dict['validation_loss'][best_step]:.4f}", color="red", ha="left", va="bottom")
    ax2.text(best_step + 0.5, ax2.get_ylim()[0] + (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.2, 
             f"Initial R2: {plot_dict['initial_validation_loss']:.4f}", color="green", ha="left", va="bottom")
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()

    plt.savefig(os.path.join(DIR_PATH, f"figs/fine_tuning_loss_plot.png"))    


def main():
    """Main function to configure and run the finetuning workflow."""
    # --- Master Configuration ---
    # This improved structure separates general settings from finetuning hyperparameters.
    config = {
        # Sets the computation device ('cuda' for GPU if available, otherwise 'cpu').
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        # The total number of samples to draw from the full dataset. This is useful for
        # managing memory and computation time, especially with large datasets.
        # For very large datasets the entire dataset is preprocessed and then
        # fit in memory, potentially leading to OOM errors.
        "num_samples_to_use": 100_000,
        # A seed for random number generators to ensure that data shuffling, splitting,
        # and model initializations are reproducible.
        "random_seed": 42,
        # The proportion of the dataset to allocate to the valid set for final evaluation.
        "valid_set_ratio": 0.3,
        # During evaluation, this is the number of samples from the training set given to the
        # model as context before it makes predictions on the test set.
        "n_inference_context_samples": 5_000,
    }
    config["finetuning"] = {
        # The total number of passes through the entire fine-tuning dataset.
        "epochs": 200,
        # A small learning rate is crucial for fine-tuning to avoid catastrophic forgetting.
        "learning_rate": 1.5e-6,
        # Meta Batch size for finetuning, i.e. how many datasets per batch. Must be 1 currently.
        "meta_batch_size": 1,
        # The number of samples within each training data split. It's capped by
        # n_inference_context_samples to align with the evaluation setup.
        "batch_size": int(
            min(
                config["n_inference_context_samples"],
                config["num_samples_to_use"] * (1 - config["valid_set_ratio"]),
            )
        ),
    }
    config["adptive_es"] = {
        "adaptive_rate": 0.2,
        "adaptive_offset": 5,
        "min_patience": 5,
        "max_patience": 100,
    }
    adaptive_es = AdaptiveES(**config["adptive_es"])

    # --- Setup Data, Model, and Dataloader ---
    X_train, X_test, y_train, y_test = prepare_data(config)
    regressor, regressor_config = setup_regressor(config)

    splitter = partial(train_test_split, test_size=config["valid_set_ratio"])
    # Note: `max_data_size` corresponds to the finetuning `batch_size` in the config
    training_datasets = regressor.get_preprocessed_datasets(
        X_train, y_train, splitter, max_data_size=config["finetuning"]["batch_size"]
    )
    finetuning_dataloader = DataLoader(
        training_datasets,
        batch_size=config["finetuning"]["meta_batch_size"],
        collate_fn=meta_dataset_collator,
    )

    # Optimizer must be created AFTER get_preprocessed_datasets, which initializes the model
    optimizer = Adam(
        regressor.model_.parameters(), lr=config["finetuning"]["learning_rate"]
    )
    print(
        f"--- Optimizer Initialized: Adam, LR: {config['finetuning']['learning_rate']} ---\n"
    )

    # Create evaluation config, linking it to the master config
    eval_config = {
        **regressor_config,
        "inference_config": {
            "SUBSAMPLE_SAMPLES": config["n_inference_context_samples"]
        },
    }

    # --- Finetuning and Evaluation Loop ---
    print("--- 3. Starting Finetuning & Evaluation ---")
    plot_dict = {
        "train_loss": [],
        "validation_loss": [],
        "initial_validation_loss": 0,
    }
    best_validation_loss = -1
    for epoch in range(config["finetuning"]["epochs"] + 1):
        if epoch > 0:
            # Create a tqdm progress bar to iterate over the dataloader
            progress_bar = tqdm(finetuning_dataloader, desc=f"Finetuning Epoch {epoch}")
            total_loss = []
            for data_batch in progress_bar:
                optimizer.zero_grad()
                (
                    X_trains_p,
                    X_tests_p,
                    y_trains_p,
                    y_test_std,
                    cat_ixs,
                    confs,
                    norm_bardist,
                    bardist,
                    _,
                    batch_y_test_raw,
                ) = data_batch

                regressor.normalized_bardist_ = norm_bardist[0]
                regressor.fit_from_preprocessed(X_trains_p, y_trains_p, cat_ixs, confs)
                logits, _, _ = regressor.forward(X_tests_p)

                # For regression, the loss function is part of the preprocessed data
                loss_fn = norm_bardist[0]
                y_target = y_test_std

                loss = loss_fn(logits, y_target.to(config["device"])).mean()
                loss.backward()
                optimizer.step()

                # Set the postfix of the progress bar to show the current loss
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                total_loss.append(loss.item())

        # Evaluation Step (runs before finetuning and after each epoch)
        mse, mae, r2 = evaluate_regressor(
            regressor, eval_config, X_train, y_train, X_test, y_test
        )
        if epoch > 0:
            plot_dict["validation_loss"].append(r2)
            plot_dict["train_loss"].append(np.mean(total_loss))
        else:
            plot_dict["initial_validation_loss"] = r2
        is_best = r2 > best_validation_loss
        if is_best:
            best_validation_loss = r2
        early_stop_no_imp = adaptive_es.update(
            cur_round=epoch, is_best=is_best,
        )

        status = "Initial" if epoch == 0 else f"Epoch {epoch}"
        patience_left = adaptive_es.remaining_patience(cur_round=epoch)
        print(
            f"📊 {status} Evaluation | Test MSE: {mse:.4f}, Test MAE: {mae:.4f}, Test R2: {r2:.4f} | Patience: {patience_left}\n"
        )
        if early_stop_no_imp:
            break

    print("--- ✅ Finetuning Finished ---")
    plot_results(plot_dict)


if __name__ == "__main__":
    main()
