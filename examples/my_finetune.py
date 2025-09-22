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
from tabpfn.model_loading import load_model

import yaml
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


def prepare_data(config: dict, flag_test: bool, flag_id: bool, data_source: str | int, test_size: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads, subsets, and splits the California Housing dataset."""
    print("--- 1. Data Preparation ---")
    if isinstance(data_source, int):
        # Fetch Ames housing data from OpenML
        DATA = sklearn.datasets.fetch_openml(
            data_id=data_source, as_frame=True, parser="auto"
        )

        # Separate features (X) and target (y)
        X_df = DATA.data
        y_df = DATA.target
    else:
        DATA = pd.read_csv(data_source)
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
        if test_size==0:
            test_size = config["valid_set_ratio"]
        splitter = partial(
            train_test_split,
            test_size=test_size,
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
        f"Loaded and split data: {X_all[0].shape[0] if X_all else 0} train samples."
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
        model_path=config["model_path"]
    )

    print(f"Using device: {config['device']}")
    print("----------------------\n")
    return regressor, regressor_config


def evaluate_regressor(regressor: TabPFNRegressor, eval_config: dict, X_train: list[np.ndarray], y_train: list[np.ndarray], X_test: list[np.ndarray], y_test: list[np.ndarray],
    ) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    """Evaluates the regressor's performance on the test set."""
    eval_regressor = clone_model_for_evaluation(regressor, eval_config, TabPFNRegressor)
    mse, mae, r2, max_ae, std_error = [], [], [], [], []
    for each_data in zip(X_train, y_train, X_test, y_test):
        eval_regressor.fit(each_data[0], each_data[1])

        try:
            predictions = eval_regressor.predict(each_data[2])
            errors = each_data[3] - predictions
            
            mse.append(mean_squared_error(each_data[3], predictions))
            mae.append(mean_absolute_error(each_data[3], predictions))
            r2.append(r2_score(each_data[3], predictions))
            max_ae.append(np.max(np.abs(errors)))
            std_error.append(np.std(errors))
        except Exception as e:
            print(f"An error occurred during evaluation: {e}")
            mse.append(np.nan)
            mae.append(np.nan)
            r2.append(np.nan)
            max_ae.append(np.nan)
            std_error.append(np.nan)

    return mse, mae, r2, max_ae, std_error


def save_model_checkpoint(regressor: TabPFNRegressor, id: int, epoch: int):
    """Saves the model checkpoint."""
    ckpt_dir = os.path.join(DIR_PATH,"logs", f"ID_{id}", "ckpt")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    checkpoint_path = os.path.join(ckpt_dir, f"tabpfn_finetuned-ID_{id}-epoch_{epoch}.ckpt")
    checkpoint = {}

    def make_serializable(config_sample):
        if isinstance(config_sample, dict):
            config_sample = {k: make_serializable(config_sample[k]) for k in config_sample}
        if isinstance(config_sample, list):
            config_sample = [make_serializable(v) for v in config_sample]
        if callable(config_sample):
            config_sample = str(config_sample)
        return config_sample
    
    checkpoint["state_dict"] = regressor.model_.state_dict()
    checkpoint["state_dict"].update({"criterion.borders": regressor.bardist_.borders, 
                                     "criterion.losses_per_bucket": regressor.bardist_.losses_per_bucket})
    checkpoint["config"] = make_serializable(regressor.config_)
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Model checkpoint saved to {checkpoint_path}")


def setup_logging(config: dict) -> tuple[object, str]:
    """Sets up logging file and returns file object and log directory."""
    log_dir = os.path.join(DIR_PATH,"logs", f"ID_{config['ID']}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f"fintuning_log-ID_{config['ID']}.txt")

    def write_config(config_, subs=0):
        for k, v in config_.items():
            if isinstance(v, dict):
                f.write(subs * "  ")
                f.write(f"{k}:\n")
                write_config(v, subs+1)
            else:
                f.write(subs * "  ")
                f.write(f"{k}: {v}\n")
    
    # Write config to log file
    with open(log_file, "w") as f:
        f.write("=== Training Configuration ===\n")
        write_config(config)
        f.write("=============================\n\n")
    
    return log_file


def log_epoch(config: dict, log_file: str, epoch: int, mse: list[float], mae: list[float], r2: list[float], 
              max_ae: list[float], std_error: list[float], patience_left: int, is_best: list[bool]):
    """Logs epoch information to file and console."""
    status = "Initial" if epoch == 0 else f"Epoch {epoch}"
    log_entry = f"📊 {status} Evaluation | Patience: {patience_left}\n"

    for idx, each_dataset in enumerate(config["dataset"]["evaluate"].keys()):
        best_marker = "🌟 BEST" if is_best[idx] else ""
        log_entry += (
            f"  {each_dataset:<10} | Test MSE: {mse[idx]:>7.4f}, Test MAE: {mae[idx]:>7.4f}, "
            f"Test R2: {r2[idx]:>7.4f}, Test max_AE: {max_ae[idx]:>7.4f}, Test std_ERR: {std_error[idx]:>7.4f} | {best_marker}\n"
        )
    
    # Write to log file
    with open(log_file, "a") as f:
        f.write(log_entry)
    
    # Print to console
    print(log_entry)


def plot_results(plot_dict: dict, id: int):
    # Create dual y-axis plot
    _, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    epochs = list(range(1, plot_dict['epochs']+1))
    
    # Plot training loss (left y-axis)
    color1 = 'tab:blue'
    ax1.plot(epochs, plot_dict['train_loss'], color=color1, linewidth=3, label='Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Plot validation loss (right y-axis)
    color2 = 'tab:orange'
    ax2.plot(epochs, plot_dict['validation_loss'], color=color2, linewidth=3, label='Validation Loss')
    ax2.set_ylabel('Validation Loss(R2)', color=color2)
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

    plt.savefig(os.path.join(DIR_PATH, "logs", f"ID_{id}", f"finetuning_loss-ID_{id}.png"))    


def save_summary(log_file: str, plot_dict: dict, total_time: list):
    with open(log_file, "a") as f:
        f.write(f"----------------------------\n")
        f.write(f"initial_validation_loss: {plot_dict['initial_validation_loss']}\n")
        f.write(f"best_validation_loss: {np.max(plot_dict['validation_loss'])}[epoch: {np.argmax(plot_dict['validation_loss'])+1}]\n")
        f.write(f"epochs: {plot_dict['epochs']}\n")
        f.write(f"----------------------------\n")
        f.write(f"finetuning_time: {(total_time[-1] - total_time[0]) / (len(total_time) - 1):.2f}s/epoch\n")
        f.write(f"inference_time: {total_time[0]:.2f}s\n")


def main():
    """Main function to configure and run the finetuning workflow."""
    # --- Master Configuration ---
    with open(os.path.join(DIR_PATH, "examples/model_configs.yaml"), "r") as file:
        config = yaml.safe_load(file)
    # Sets the computation device ('cuda' for GPU if available, otherwise 'cpu').
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    # The number of samples within each training data split. It's capped by
    # n_inference_context_samples to align with the evaluation setup.
    config["finetuning"]["batch_size"] = int(
        min(
            config["n_inference_context_samples"],
            config["num_samples_to_use"] * (1 - config["valid_set_ratio"]),
        )
    )
    adaptive_es = AdaptiveES(**config["adptive_es"])

    # --- Setup logging ---
    log_file = setup_logging(config)

    # --- Setup Data, Model, and Dataloader ---
    dataset_evaluate = config["dataset"]["evaluate"]
    X_evaluate_train, y_evaluate_train, X_evaluate_test, y_evaluate_test = [], [], [], []
    for each_dataset in dataset_evaluate.keys():
        if isinstance(dataset_evaluate[each_dataset]["test"], str):
            X_eval_train, y_eval_train = prepare_data(config, flag_test=False, flag_id=dataset_evaluate[each_dataset]["flag_id"], 
                                                      data_source=dataset_evaluate[each_dataset]["train"])
            
            X_eval_test, y_eval_test = prepare_data(config, flag_test=False, flag_id=dataset_evaluate[each_dataset]["flag_id"], 
                                                    data_source=dataset_evaluate[each_dataset]["test"])
        else:
            X_eval_train, X_eval_test, y_eval_train, y_eval_test = prepare_data(config, flag_test=True, test_size=dataset_evaluate[each_dataset]["test"], 
                                                                                flag_id=dataset_evaluate[each_dataset]["flag_id"],
                                                                                data_source=dataset_evaluate[each_dataset]["train"])
        X_evaluate_train.append(X_eval_train)
        y_evaluate_train.append(y_eval_train)
        X_evaluate_test.append(X_eval_test)
        y_evaluate_test.append(y_eval_test)

    dataset_openml = config["dataset"]["openml"]
    X_train, y_train = [], []
    for each_source, each_flag_id in zip(dataset_openml["source"], dataset_openml["flag_id"]):
        X_openml_train, y_openml_train = prepare_data(config, flag_test=False, flag_id=each_flag_id, 
                                                              data_source=each_source)
        batch_len_num = X_openml_train.shape[0] // config["finetuning"]["batch_size"]
        X_openml_train = X_openml_train[:batch_len_num * config["finetuning"]["batch_size"],...]
        y_openml_train = y_openml_train[:batch_len_num * config["finetuning"]["batch_size"],...]
        X_train.append(X_openml_train)
        y_train.append(y_openml_train)

    dataset_pretrained = config["dataset"]["pretrained"]
    X_train_pretrained, y_train_pretrained = prepare_data_pretrained(config, seq_len=config["num_samples_to_use"], batch_num=dataset_pretrained["batch_num"],
                                                                     data_path=dataset_pretrained["data_path"])

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
    regressor.config_ = torch.load(config["model_path"], map_location="cpu", weights_only=None)["config"]
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
    best_validation_loss = [-1 for _ in range(len(dataset_evaluate))]
    start_time = time.time()
    total_time = []
    for epoch in range(config["finetuning"]["epochs"] + 1):
        if epoch > 0:
            plot_dict["epochs"] += 1
            plot_dict["train_loss"].append([])
            # Create a tqdm progress bar to iterate over the dataloader
            assert finetuning_dataloader or finetuning_dataloader_pretrained, "Dataloader!!!"
            len_pb = len(finetuning_dataloader[0]) if finetuning_dataloader else len(finetuning_dataloader_pretrained[0][0])
            progress_bar = tqdm(zip(*finetuning_dataloader, *finetuning_dataloader_pretrained[epoch-1]), 
                                desc=f"Finetuning Epoch {epoch}", total=len_pb,)
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
                plot_dict["train_loss"][-1].append(total_loss.item())
            plot_dict["train_loss"][-1] = sum(plot_dict["train_loss"][-1]) / len(plot_dict["train_loss"][-1])

        # Evaluation Step (runs before finetuning and after each epoch)
        mse, mae, r2, max_ae, std_error = evaluate_regressor(regressor, eval_config, X_evaluate_train, y_evaluate_train, X_evaluate_test, y_evaluate_test)
        is_best = []
        for idx, each_r2 in enumerate(r2):
            if each_r2 > best_validation_loss[idx]:
                is_best.append(True)
                best_validation_loss[idx] = each_r2
            else:
                is_best.append(False)
        any_best = np.array(is_best).any()
        if any_best:
            save_model_checkpoint(regressor, config["ID"], epoch)
        if epoch == 0:
            plot_dict["initial_validation_loss"] = r2[0]
        else:
            plot_dict["validation_loss"].append(r2[0])
        total_time.append(time.time() - start_time)

        early_stop_no_imp = adaptive_es.update(
            cur_round=epoch, is_best=any_best,
        )

        patience_left = adaptive_es.remaining_patience(cur_round=epoch)
        log_epoch(config, log_file, epoch, mse, mae, r2, max_ae, std_error, patience_left, is_best)
        
        if early_stop_no_imp:
            with open(log_file, "a") as f:
                f.write("\n⚠️ Early stopping triggered!\n")
            break

    print("--- ✅ Finetuning Finished ---")
    plot_results(plot_dict, config["ID"])
    save_summary(log_file, plot_dict, total_time)


if __name__ == "__main__":
    main()
