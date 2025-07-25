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
from functools import partial
import time
import os
import sys
# 添加项目根目录到Python路径
DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(DIR_PATH)
from utils.early_stopping import AdaptiveES


def prepare_data(config: dict, flag_fetch: bool, flag_test: bool, flag_id: bool, data_path: str | int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads, subsets, and splits the California Housing dataset."""
    print("--- 1. Data Preparation ---")
    if flag_fetch:
        # Fetch Ames housing data from OpenML
        DATA = sklearn.datasets.fetch_openml(
            data_id=data_path, as_frame=True, parser="auto"
        )

        # Separate features (X) and target (y)
        X_df = DATA.data
        y_df = DATA.target
    else:
        DATA = pd.read_csv(data_path)
        X_df = DATA.iloc[:, :-1]
        y_df = DATA.iloc[:, -1]
    
    if flag_id:
        X_df = X_df.iloc[:, 1:]

    # Select only numeric features for simplicity
    X_numeric = X_df.select_dtypes(include=np.number)

    X_all, y_all = X_numeric.values, y_df.values

    rng = np.random.default_rng(config["random_seed"])
    num_samples_to_use = min(config["num_samples_to_use"], len(y_all))
    indices = rng.choice(np.arange(len(y_all)), size=num_samples_to_use, replace=False)
    X, y = X_all[indices], y_all[indices]

    if flag_test:
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
    else:
        print(
            f"Loaded and split data: {X.shape[0]} train samples."
        )
        print("---------------------------\n")
        return X, y


def prepare_data_pretrained(config: dict, data_path: str, seq_len: int, batch_num: int = 8) -> tuple[list[np.ndarray], list[np.ndarray]]:
    assert batch_num <= 8, "Max batch_num is 8."
    print("--- 1. Data Preparation ---")
    folders = [os.path.join(data_path, each_folder) for each_folder in os.listdir(data_path)]
    X_pretrained, y_pretrained = [], []
    for i in range(config["finetuning"]["epochs"]):
        data = [os.path.join(folders[i], each_data) for each_data in os.listdir(folders[i])]
        X_all, y_all = [], []
        for j in range(batch_num):
            DATA = pd.read_csv(data[j])
            X_numeric = DATA.iloc[:seq_len, :-1].select_dtypes(include=np.number).values
            y_numeric = DATA.iloc[:seq_len, -1].values
            X_all.append(X_numeric)
            y_all.append(y_numeric)
        X_pretrained.append(X_all)
        y_pretrained.append(y_all)

    print(
        f"Loaded and split data: {X_all[0].shape[0]} train samples."
    )
    print("---------------------------\n")
    return X_pretrained, y_pretrained


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
) -> tuple[float, float, float, float, float]:
    """Evaluates the regressor's performance on the test set."""
    eval_regressor = clone_model_for_evaluation(regressor, eval_config, TabPFNRegressor)
    eval_regressor.fit(X_train, y_train)

    try:
        predictions = eval_regressor.predict(X_test)
        errors = y_test - predictions
        
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        max_ae = np.max(np.abs(errors))  # Maximum Absolute Error
        std_error = np.std(errors)       # Standard Deviation of Errors
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        mse, mae, r2, max_ae, std_error = np.nan, np.nan, np.nan, np.nan, np.nan

    return mse, mae, r2, max_ae, std_error


def plot_results(plot_dict: dict):
    # Create dual y-axis plot
    _, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    plot_dict['step'] = range(len(plot_dict['train_loss']))
    
    # Plot training loss (left y-axis)
    color1 = 'tab:blue'
    ax1.plot(plot_dict['step'], plot_dict['train_loss'], color=color1, linewidth=3, label='Training Loss')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Training Loss', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Plot validation loss (right y-axis)
    color2 = 'tab:orange'
    ax2.plot(plot_dict['step'], plot_dict['validation_loss'], color=color2, linewidth=3, label='Validation Loss')
    ax2.set_ylabel('Validation Loss', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add best step marker
    best_step = np.argmax(plot_dict['validation_loss'])
    ax1.axvline(x=best_step, color="red", linestyle="--", linewidth=2, label="Best Step")
    ax2.axhline(y=plot_dict['initial_validation_loss'], color="green", linestyle="--", linewidth=2, label="Initial Validation Loss")
    ax1.text(best_step + 0.5, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.1, 
             f"Best Step R2: {plot_dict['validation_loss'][best_step]:.4f}", color="red", ha="left", va="bottom")
    ax2.text(best_step + 0.5, ax2.get_ylim()[0] + (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.2, 
             f"Initial R2: {plot_dict['initial_validation_loss']:.4f}", color="green", ha="left", va="bottom")
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()

    plt.savefig(os.path.join(DIR_PATH, f"figs/fine_tuning_loss_plot.png"))    


def save_log(config: dict, plot_dict: dict, total_time: list):
    with open(os.path.join(DIR_PATH, f"logs/log.txt"), "a") as f:
        f.write(f"--------------------------------\n")
        f.write(f"\tnum_samples_to_use: {config['num_samples_to_use']}\n")
        f.write(f"\tn_inference_context_samples: {config['n_inference_context_samples']}\n")
        f.write(f"\tlearning_rate: {config['finetuning']['learning_rate']}\n")
        f.write(f"\t----------------------------\n")
        f.write(f"\tinitial_validation_loss: {plot_dict['initial_validation_loss']}\n")
        f.write(f"\tbest_validation_loss: {np.max(plot_dict['validation_loss'])}[step: {np.argmax(plot_dict['validation_loss'])}]\n")
        f.write(f"\tepochs: {plot_dict['epochs']}[{int(len(plot_dict['validation_loss']) / plot_dict['epochs'])}steps/epoch]\n")
        f.write(f"\t----------------------------\n")
        f.write(f"\tfinetuning_time: {(total_time[-1] - total_time[0]) / (len(total_time) - 1):.2f}s/step\n")
        f.write(f"\tinference_time: {total_time[0]:.2f}s\n")
        f.write(f"\n\n")


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
        "num_samples_to_use": 10_000,
        # A seed for random number generators to ensure that data shuffling, splitting,
        # and model initializations are reproducible.
        "random_seed": 42,
        # The proportion of the dataset to allocate to the valid set for final evaluation.
        "valid_set_ratio": 0.3,
        # During evaluation, this is the number of samples from the training set given to the
        # model as context before it makes predictions on the test set.
        "n_inference_context_samples": 1_000,
    }
    config["finetuning"] = {
        # The total number of passes through the entire fine-tuning dataset.
        "epochs": 50,
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
        "min_patience": 10,
        "max_patience": 100,
    }
    adaptive_es = AdaptiveES(**config["adptive_es"])
    config["dataset"] = {
        "finetune": "sample",
    }

    # --- Setup Data, Model, and Dataloader ---
    X_train_sample, y_train_sample = prepare_data(config, flag_fetch=False, flag_test=False, flag_id=True, 
                                                  data_path="/home/fit/zhangcs/WORK/chenkq/project/dataset/phm2016/PHM2016_4A_train.csv")
    X_test_sample, y_test_sample = prepare_data(config, flag_fetch=False, flag_test=False, flag_id=True, 
                                                data_path="/home/fit/zhangcs/WORK/chenkq/project/dataset/phm2016/PHM2016_4A_test.csv")
    X_train_origin, X_test_origin, y_train_origin, y_test_origin = prepare_data(config, flag_fetch=True, flag_test=True, flag_id=False, 
                                                                                data_path=44981)
    X_val_train_origin, X_val_test_origin, y_val_train_origin, y_val_test_origin = train_test_split(X_test_origin, y_test_origin, 
                                                                                                   test_size=config["valid_set_ratio"], random_state=config["random_seed"])
    X_train = [X_train_origin]
    y_train = [y_train_origin]

    X_train_pretrained, y_train_pretrained = prepare_data_pretrained(config, seq_len=X_train_origin.shape[0], batch_num=4,
                                                                     data_path="/home/fit/zhangcs/WORK/chenkq/project/dataset/pretrained")

    regressor, regressor_config = setup_regressor(config)

    splitter = partial(train_test_split, test_size=config["valid_set_ratio"])
    # Note: `max_data_size` corresponds to the finetuning `batch_size` in the config
    training_datasets_pretrained = [
        [regressor.get_preprocessed_datasets(
        X_train_pretrained[i][j], y_train_pretrained[i][j], splitter, max_data_size=config["finetuning"]["batch_size"]
            ) for j in range(len(X_train_pretrained[i]))
        ] for i in range(config["finetuning"]["epochs"])
    ]
    finetuning_dataloader_pretrained = [
        [DataLoader(
        training_datasets_pretrained[i][j],
        batch_size=config["finetuning"]["meta_batch_size"],
        collate_fn=meta_dataset_collator,
            ) for j in range(len(training_datasets_pretrained[i]))
        ] for i in range(config["finetuning"]["epochs"])
    ]

    training_datasets = [regressor.get_preprocessed_datasets(
        X, y, splitter, max_data_size=config["finetuning"]["batch_size"]
    ) for X, y in zip(X_train, y_train)]
    finetuning_dataloader = [DataLoader(
        each_training_datasets,
        batch_size=config["finetuning"]["meta_batch_size"],
        collate_fn=meta_dataset_collator,
    ) for each_training_datasets in training_datasets]

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
        "epochs": 0,
    }
    best_validation_loss = -1
    start_time = time.time()
    total_time = []
    for epoch in range(config["finetuning"]["epochs"] + 1):
        if epoch > 0:
            breakpoint()
            plot_dict["epochs"] += 1
            # Create a tqdm progress bar to iterate over the dataloader
            progress_bar = tqdm(zip(*finetuning_dataloader, *finetuning_dataloader_pretrained[epoch-1]), desc=f"Finetuning Epoch {epoch}", total=len(finetuning_dataloader[0]),)
            for finetuning_dataloader_tuple in progress_bar:
                total_loss = None
                optimizer.zero_grad()
                for data_batch in finetuning_dataloader_tuple:              
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

                    loss = loss_fn(logits, y_target.to(config["device"]))
                    total_loss = loss if total_loss is None else torch.concatenate((total_loss, loss))
                    # loss.backward()
                    # optimizer.step()
                progress_bar.set_postfix({"CUDA Memory": f"{torch.cuda.memory_allocated()/1024**2:.2f} MB"})
                total_loss = total_loss.mean()
                total_loss.backward()
                optimizer.step()
                
                # # Set the postfix of the progress bar to show the current loss
                # progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                plot_dict["train_loss"].append(total_loss.item())
                total_time.append(time.time() - start_time)

        # Evaluation Step (runs before finetuning and after each epoch)
        mse, mae, r2, max_ae, std_error = evaluate_regressor(regressor, eval_config, X_train_sample, y_train_sample, X_test_sample, y_test_sample)
        is_best = r2 > best_validation_loss
        if is_best:
            best_validation_loss = r2
        if epoch == 0:
            plot_dict["initial_validation_loss"] = r2
        else:
            plot_dict["validation_loss"].append(r2)
        total_time.append(time.time() - start_time)

        early_stop_no_imp = adaptive_es.update(
            cur_round=epoch, is_best=is_best,
        )

        status = "Initial" if epoch == 0 else f"Epoch {epoch}"
        patience_left = adaptive_es.remaining_patience(cur_round=epoch)
        print(
            f"📊 {status} Evaluation | Test MSE: {mse:.4f}, Test MAE: {mae:.4f}, Test R2: {r2:.4f}, Test max_AE: {max_ae:.4f}, Test std_ERR: {std_error:.4f}\
            | Patience: {patience_left}\n"
        )
        if early_stop_no_imp:
            break

    print("--- ✅ Finetuning Finished ---")
    plot_results(plot_dict)
    save_log(config, plot_dict, total_time)


if __name__ == "__main__":
    main()
