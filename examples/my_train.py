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
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabpfn import TabPFNRegressor
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.model_loading import save_tabpfn_model
from tabpfn.utils import meta_dataset_collator

from train_utils.utils import init_dist, get_uniform_single_eval_pos_sampler, get_cosine_schedule_with_warmup, get_schedule_with_warmup, torch_nanmean
from train_utils.model_configs import get_prior_config_causal, sample_hypers, get_mlp_prior_hyperparameters, get_gp_prior_hyperparameters
import train_utils.priors as priors

import yaml
import pandas as pd
from functools import partial
import itertools
from typing import Any, Callable, Literal
from contextlib import nullcontext
import time
import sys
import csv
import argparse
from pathlib import Path
# 添加项目根目录到Python路径
DIR_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(DIR_PATH))


def create_layered_optimizer(
    model: TabPFNRegressor, layer_config: dict[str, dict], weight_decay: float = 1e-2
) -> tuple[torch.optim.AdamW, torch.optim.AdamW | None]:
    import re

    original_params_groups: list[dict] = []
    new_params_groups: list[dict] = []
    def choose_param_groups_append(layer_type: Literal["original", "new"]) -> Callable:
        if layer_type == "original":
            return original_params_groups.append
        elif layer_type == "new":
            return new_params_groups.append
    
    def create_idx2type_dict(layer_config: dict[str, dict]) -> dict[int, dict[str, str]]:
        idx2type_dict = {}
        for layer_type, config in layer_config.items():
            for layer_idx in config["indices"]:
                idx2type_dict[layer_idx] = {"layer_type": layer_type}
        return idx2type_dict
    
    target_layers = "transformer_encoder"
    other_layers = "other_layers"
    target_layers_idx2type_dict = create_idx2type_dict(layer_config[target_layers])
    
    def create_type2idx_dict(idx2type_dict: dict[int, dict[str, str]]) -> dict[str, list[torch.nn.Parameter]]:
        layers_type2idx_dict: dict[str, dict[str, list[torch.nn.Parameter]]] = {target_layers: {}, other_layers: {}}
        layers_pattern = re.compile(f"{target_layers}\.layers\.(\d+)\.")
        for name, param in model.named_parameters():
            match = layers_pattern.search(name)
            if match:
                layer_idx = int(match.group(1))
                layers_type2idx_dict[target_layers].setdefault(idx2type_dict[layer_idx]["layer_type"], []).append(param)
            else:
                layers_type2idx_dict[other_layers].setdefault("original", []).append(param)
        return layers_type2idx_dict
    
    layers_type2idx_dict = create_type2idx_dict(target_layers_idx2type_dict)
    for each_layers, each_layers_idx2type_dict in layers_type2idx_dict.items():
        for layer_type, params in each_layers_idx2type_dict.items():
            try:
                lr = layer_config[each_layers][layer_type]["lr"]
            except KeyError:
                lr = layer_config.get("default_lr")
            temp_params_dict = {
                "params": params,
                "lr": lr,
                "initial_lr": lr,
                "name": f"{each_layers}_{layer_type}",
            }
            choose_param_groups_append(layer_type)(temp_params_dict)
    
    original_optimizer = torch.optim.AdamW(original_params_groups, weight_decay=weight_decay)
    new_optimizer = torch.optim.AdamW(new_params_groups, weight_decay=weight_decay) if new_params_groups else None
    
    def print_optimizer_info(*param_groups: list[dict[str, Any]]) -> None:
        print("--- 分层学习率优化器配置 ---")
        for i, group in enumerate(itertools.chain(*param_groups)):
            print(f"参数组 {i+1} ({group['name']}): {len(group['params'])} 个参数, 学习率: {group['lr']:.2e}")
        print("----------------------------\n")
    
    print_optimizer_info(original_params_groups, new_params_groups)

    return original_optimizer, new_optimizer


def prepare_data(config: dict, flag_test: bool, flag_id: bool, data_source: str | int, test_size: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads, subsets, and splits the California Housing dataset."""
    print("--- 2. Data Preparation ---")
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


def setup_regressor(config: dict) -> tuple[TabPFNRegressor, dict]:
    """Initializes the TabPFN regressor and its configuration."""
    print("--- 1. Model Setup ---")
    str2dtype_map = {
        "float64": torch.float64,
        "float32": torch.float32,
        "float16": torch.float16,
    }
    regressor_config = {
        "ignore_pretraining_limits": True,
        "device": config["device"],
        "random_state": config["random_seed"],
        "n_estimators": config["n_estimators"],
        "inference_precision": str2dtype_map.get(config["inference_precision"], config["inference_precision"]),
    }
    regressor = TabPFNRegressor(
        **regressor_config,
        model_path=config.get("model_path", config.get("default_model_path", None)),
        fit_mode="batched",
        differentiable_input=False,
    )
    regressor._initialize_model_variables()

    print(f"Using device: {config['device']}")
    print(f"Using a Transformer with {sum(p.numel() for p in regressor.model_.parameters())/1000/1000:.{2}f} M parameters")
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
    ckpt_dir = DIR_PATH / "logs" / f"ID_{id}" / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / f"tabpfn_finetuned-ID_{id}-epoch_{epoch}.ckpt"
    save_tabpfn_model(regressor, checkpoint_path)
    print(f"💾 Model checkpoint saved to {checkpoint_path}")


def setup_logging(config: dict) -> tuple[Path, Path, Path]:
    """Sets up logging file and returns log directory."""
    log_dir = DIR_PATH / "logs" / f"ID_{config['ID']}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    def write_config(config_, subs=0):
        for k, v in config_.items():
            if isinstance(v, dict):
                f.write(subs * "  ")
                f.write(f"{k}:\n")
                write_config(v, subs+1)
            else:
                f.write(subs * "  ")
                f.write(f"{k}: {v}\n")
    
    # log文件
    log_file_path = log_dir / f"fintuning_log-ID_{config['ID']}.txt"
    if not log_file_path.exists():
        with open(log_file_path, "w") as f:
            f.write("=== Training Configuration ===\n")
            write_config(config)
            f.write("=============================\n\n")

    # step loss CSV文件
    step_loss_csv_path = log_dir / f"step_losses-ID_{config['ID']}.csv"
    if not step_loss_csv_path.exists():
        with open(step_loss_csv_path, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch", "step", "loss"])  # header
    
    # evaluate CSV文件
    eval_csv_path = log_dir / f"eval_metrics-ID_{config['ID']}.csv"
    if not eval_csv_path.exists():
        with open(eval_csv_path, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch", "dataset", "mse", "mae", "r2", "max_ae", "std_error", "is_best", "scheduler_lr"])  # header
    
    return log_file_path, step_loss_csv_path, eval_csv_path


def log_epoch(
    config: dict, log_file_path: str, eval_csv_path: str, 
    epoch: int, mse: list[float], mae: list[float], r2: list[float], max_ae: list[float], std_error: list[float], 
    is_best: list[bool], new_scheduler_lr: float, 
    write_to_file: bool = True
) -> None:
    """Logs epoch information to file and console."""
    status = "Initial" if epoch == 0 else f"Epoch {epoch}"
    log_entry = f"📊 {status} Evaluation"
    log_entry += f" | New Scheduler LR: {new_scheduler_lr:.2e}\n" if new_scheduler_lr is not None else "\n"

    for idx, each_dataset in enumerate(config["evaluation"].keys()):
        best_marker = "🌟 BEST" if is_best[idx] else ""
        log_entry += (
            f"  {each_dataset:<10} | Test MSE: {mse[idx]:>7.4f}, Test MAE: {mae[idx]:>7.4f}, "
            f"Test R2: {r2[idx]:>7.4f}, Test max_AE: {max_ae[idx]:>7.4f}, Test std_ERR: {std_error[idx]:>7.4f} | {best_marker}\n"
        )
    
    print(log_entry)
    
    if write_to_file:
        # write to log file
        with open(log_file_path, "a") as f:
            f.write(log_entry)
    
        # write to evaluate CSV
        with open(eval_csv_path, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for idx, each_dataset in enumerate(config["evaluation"].keys()):
                writer.writerow([
                    epoch, 
                    each_dataset, 
                    mse[idx], 
                    mae[idx], 
                    r2[idx], 
                    max_ae[idx], 
                    std_error[idx], 
                    is_best[idx], 
                    new_scheduler_lr
                ])


def init_config() -> tuple[dict, dict, dict]:
    parser = argparse.ArgumentParser(description="Fine-tune TabPFN model with specified config file")
    parser.add_argument(
        "--config", 
        type=Path, 
        default=DIR_PATH / "examples" / "configs" / "train_configs.yaml",
        help="Path to the YAML configuration file (default: examples/configs/train_configs.yaml)"
    )
    args = parser.parse_args()
    with open(args.config, "r") as file:
       manual_config = yaml.safe_load(file)

    prior_config = {
        **get_prior_config_causal(),
        **manual_config["SynsData"]["prior"], 
        **manual_config["SynsData"]["basic"],
    }
    prior_config = sample_hypers(prior_config)
    prior_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    prior_config["verbose"] = False
    if prior_config["bptt_extra_samples"] is not None:
        prior_config["eval_positions"] = [int(prior_config["bptt"])]
    
    model_config = manual_config["Model"]
    model_config["device"] = prior_config["device"]
    assert (model_config.get("lr") is not None) ^ (model_config.get("layers") is not None), \
        "Either 'lr' or 'layers' must be provided (but not both) in the model config"
    if "layers" in model_config:
        if model_config["epoch_start"] > 0:
            assert model_config.get("model_path", None) is not None,\
                "epoch_start > 0 but model_path is not in the model config"
            assert model_config["layers"]["transformer_encoder"]["new"].get("lr", None) is not None, \
                "epoch_start > 0 but lr is not in the model config"
        else:
            model_config["layers"]["transformer_encoder"]["new"]["lr"] = prior_config["lr"]

    dataset_config = manual_config["Dataset"]

    return prior_config, model_config, dataset_config


def make_get_batch(model_proto, **extra_kwargs):
    def new_get_batch(batch_size, seq_len, num_features, hyperparameters
            , device, model_proto=model_proto
            , **kwargs):
        kwargs = {**extra_kwargs, **kwargs} # new args overwrite pre-specified args
        return model_proto.get_batch(
            batch_size=batch_size
            , seq_len=seq_len
            , device=device
            , hyperparameters=hyperparameters
            , num_features=num_features, **kwargs)
    return new_get_batch


def init_prior_hyperparameters(prior_config: dict) -> dict:
    get_batch_gp = make_get_batch(priors.fast_gp)
    get_batch_mlp = make_get_batch(priors.mlp)
    if prior_config.get("flexible"):
        get_batch_gp = make_get_batch(priors.flexible_categorical, **{"get_batch": get_batch_gp})
        get_batch_mlp = make_get_batch(priors.flexible_categorical, **{"get_batch": get_batch_mlp})
    prior_bag_hyperparameters = {
        "prior_bag_get_batch": (get_batch_gp, get_batch_mlp), 
        "prior_bag_exp_weights_1": 2.0,
    }
    prior_hyperparameters = {
        **get_mlp_prior_hyperparameters(prior_config), 
        **get_gp_prior_hyperparameters(prior_config), 
        **prior_bag_hyperparameters,
    }
    if prior_config.get("flexible"):
        prior_hyperparameters["normalize_labels"] = False
        prior_hyperparameters["check_is_compatible"] = False
    prior_hyperparameters["prior_mlp_scale_weights_sqrt"] = prior_config.get("prior_mlp_scale_weights_sqrt", None)
    prior_hyperparameters["rotate_normalized_labels"] = prior_config.get("rotate_normalized_labels", True)

    return prior_hyperparameters


def init_model_proto(prior_config: dict) -> tuple[type, dict]:
    model_proto = priors.prior_bag
    extra_kwargs = {}
    if prior_config.get("differentiable"):
        get_batch_base = make_get_batch(model_proto)
        extra_kwargs = {
            "get_batch": get_batch_base, 
            "differentiable_hyperparameters": prior_config["differentiable_hyperparameters"],
        }
        model_proto = priors.differentiable_prior

    return model_proto, extra_kwargs


def init_dataset(dataset_config: dict, datasets: dict[str, dict]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    X_train, y_train, X_test, y_test = [], [], [], []
    for _, each_dataset in datasets.items():
        if "test" in each_dataset:
            if each_dataset["test"]:
                each_X_train, each_y_train = prepare_data(
                    dataset_config, flag_test=False, flag_id=each_dataset["flag_id"], 
                    data_source=each_dataset["train"],
                )
                each_X_test, each_y_test = prepare_data(
                    dataset_config, flag_test=False, flag_id=each_dataset["flag_id"], 
                    data_source=each_dataset["test"],
                )
            else:
                each_X_train, each_X_test, each_y_train, each_y_test = prepare_data(
                    dataset_config, flag_test=True, flag_id=each_dataset["flag_id"], 
                    data_source=each_dataset["train"]
                )
        else:
            each_X_train, each_y_train = prepare_data(
                dataset_config, flag_test=False, flag_id=each_dataset["flag_id"], 
                data_source=each_dataset["train"],
            )
            each_X_test, each_y_test = None, None
        X_train.append(each_X_train)
        y_train.append(each_y_train)
        X_test.append(each_X_test)
        y_test.append(each_y_test)

    return X_train, y_train, X_test, y_test


def init_dataloader(model: TabPFNRegressor, dataset_config: dict, datasets: dict[str, dict]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    X_train, y_train, _, _ = init_dataset(dataset_config, datasets)
    for each_X_train, each_y_train in zip(X_train, y_train):
        batch_len_num = each_X_train.shape[0] // dataset_config["batch_size"]
        each_X_train = each_X_train[:batch_len_num * dataset_config["batch_size"],...]
        each_y_train = each_y_train[:batch_len_num * dataset_config["batch_size"],...]

    splitter = partial(train_test_split, test_size=dataset_config["valid_set_ratio"])
    dataset = [model.get_preprocessed_datasets(
        X, y, splitter, max_data_size=dataset_config["batch_size"]
    ) for X, y in zip(X_train, y_train)]
    dataloader = [DataLoader(
        each_dataset,
        batch_size=dataset_config["meta_batch_size"],
        collate_fn=meta_dataset_collator,
    ) for each_dataset in dataset]

    return dataloader


def train(
    model: TabPFNRegressor, 
    prior_dataloader_class: type,
    real_dataloader: list[DataLoader] | None, 
    device: str, bptt: int, bptt_extra_samples: int | None,
    steps_per_epoch: int, batch_size: int,
    single_eval_pos_gen: Callable[[], int],
    lr: dict | float, epoch_start: int, epochs: int, train_mixed_precision: bool,
    aggregate_k_gradients: int,
    evaluate_function: Callable,
    ID: int,
    log_function: Callable,
    step_loss_csv_path: str,
    extra_prior_kwargs_dict: dict,
) -> None:
    using_dist, rank, device = init_dist(device)
    if using_dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], output_device=rank, broadcast_buffers=False
        )
        print(f"Using distributed training with rank {rank}")


    single_eval_pos_gen = single_eval_pos_gen if callable(single_eval_pos_gen) else lambda: single_eval_pos_gen
    def eval_pos_seq_len_sampler():
        single_eval_pos = single_eval_pos_gen()
        if bptt_extra_samples is not None:
            return single_eval_pos, single_eval_pos + bptt_extra_samples
        else:
            return single_eval_pos, bptt
    
    dl = prior_dataloader_class(
        num_steps=steps_per_epoch, batch_size=batch_size,
        eval_pos_seq_len_sampler=eval_pos_seq_len_sampler,
        seq_len_maximum=bptt+(bptt_extra_samples if bptt_extra_samples else 0),
        device=device,
        collate_fn=meta_dataset_collator,
        **extra_prior_kwargs_dict,
    )
    dl.model = model
    assert len(dl) % aggregate_k_gradients == 0, \
    f"Please set the number of `steps_per_epoch`: {len(dl)} s.t. \
    `aggregate_k_gradients`:{aggregate_k_gradients} divides it."


    def init_optimizer_scheduler(
        lr: dict | float, epoch_start: int
    ) -> tuple[torch.optim.AdamW, torch.optim.AdamW | None, torch.optim.lr_scheduler.LambdaLR, torch.optim.lr_scheduler.LambdaLR | None]:
        if isinstance(lr, dict):
            if lr.get("warmup_steps"):
                optimizer, new_optimizer = create_layered_optimizer(model.model_, lr, weight_decay=0.0)
                scheduler = get_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=lr["warmup_steps"], 
                    last_epoch=epoch_start,
                )
                new_scheduler = get_cosine_schedule_with_warmup(
                    new_optimizer, 
                    num_warmup_steps=lr["warmup_steps"], 
                    num_training_steps=epochs,
                    last_epoch=epoch_start,
                )
            else:
                optimizer = AdamW(model.model_.parameters(), lr=lr["default_lr"], weight_decay=0.0)
                new_optimizer = None
                scheduler, new_scheduler = None, None
        else:
            optimizer = AdamW(model.model_.parameters(), lr=lr, weight_decay=0.0)
            new_optimizer = None
            scheduler, new_scheduler = None, None

        return optimizer, new_optimizer, scheduler, new_scheduler

    optimizer, new_optimizer, scheduler, new_scheduler = init_optimizer_scheduler(lr, epoch_start)
    scaler = GradScaler('cuda') if train_mixed_precision else None


    def train_epoch(epoch: int) -> float:
        total_loss = 0.0
        step_rows = []
        progress_bar = tqdm(dl, desc=f"Training Epoch {epoch}", total=len(dl))
        for step, data_batch in enumerate(progress_bar):
            flag = True if step % 20 == 0 else False
            if using_dist and not (step % aggregate_k_gradients == aggregate_k_gradients - 1):
                cm = model.no_sync()
            else:
                cm = nullcontext()
            with cm:
                if flag: 
                    before_forward = time.time()
                
                with autocast('cuda', enabled=scaler is not None):
                    (
                        X_trains_preprocessed,
                        X_tests_preprocessed,
                        y_trains_znorm,
                        y_test_znorm,
                        cat_ixs,
                        confs,
                        raw_space_bardist_,
                        znorm_space_bardist_,
                        _,
                        _y_test_raw,
                    ) = data_batch
                    breakpoint()

                    model.raw_space_bardist_ = raw_space_bardist_[0]
                    model.znorm_space_bardist_ = znorm_space_bardist_[0]
                    model.fit_from_preprocessed(
                        X_trains_preprocessed,
                        y_trains_znorm,
                        cat_ixs,
                        confs,
                    )
                    logits, _, _ = model.forward(X_tests_preprocessed)

                    if flag: 
                        forward_time = time.time() - before_forward

                    loss_fn = znorm_space_bardist_[0]
                    losses = loss_fn(logits, y_test_znorm.to(logits.device))
                    losses = losses.view((-1, batch_size))
                    loss = torch_nanmean(losses.mean(dim=0))
                    loss = loss / aggregate_k_gradients
                
                if scaler: 
                    loss = scaler.scale(loss)
                loss.backward()

                if step % aggregate_k_gradients == aggregate_k_gradients - 1:                   
                    if scaler: 
                        scaler.unscale_(optimizer)
                        if new_optimizer:
                            scaler.unscale_(new_optimizer)
                    
                    for group in optimizer.param_groups:
                        torch.nn.utils.clip_grad_norm_(group['params'], 1.)
                    if new_optimizer:
                        for group in new_optimizer.param_groups:
                            torch.nn.utils.clip_grad_norm_(group['params'], 1.)
                    
                    if scaler:
                        scaler.step(optimizer)
                        if new_optimizer:
                            scaler.step(new_optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                        if new_optimizer:
                            new_optimizer.step()
                    
                    optimizer.zero_grad()
                    if new_optimizer:
                        new_optimizer.zero_grad()
                
                if flag: 
                    step_time = time.time() - before_forward

                if not torch.isnan(loss):
                    losses = losses.mean()
                    total_loss += losses.cpu().detach().item()
                    progress_bar.set_postfix(loss=f"{losses.item():.4f}")
                    step_rows.append([epoch, step, float(losses.item())])

                if flag:
                    print(
                        f'| epoch {epoch:3d} | step {step} | mean loss {total_loss / (step+1) :5.2f} | step time {step_time:5.2f} | forward time {forward_time:5.2f} |',
                        flush=True
                    )
        if step_rows:
            with open(step_loss_csv_path, mode="a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(step_rows)
        
        return total_loss / steps_per_epoch

    def finetune_epoch(epoch:int) -> float:
        total_loss = 0.0
        step_rows = []
        progress_bar = tqdm(
            zip(dl, *real_dataloader), 
            desc=f"Finetuning Epoch {epoch}", 
            total=len(real_dataloader[0])
        )
        for step, dataloader_tuple in enumerate(progress_bar):
            loss_step = 0.0
            for data_batch in dataloader_tuple:
                (
                    X_trains_preprocessed,
                    X_tests_preprocessed,
                    y_trains_znorm,
                    y_test_znorm,
                    cat_ixs,
                    confs,
                    raw_space_bardist_,
                    znorm_space_bardist_,
                    _,
                    _y_test_raw,
                ) = data_batch
                model.raw_space_bardist_ = raw_space_bardist_[0]
                model.znorm_space_bardist_ = znorm_space_bardist_[0]
                model.fit_from_preprocessed(
                    X_trains_preprocessed,
                    y_trains_znorm,
                    cat_ixs,
                    confs,
                )
                logits, _, _ = model.forward(X_tests_preprocessed)

                loss_fn = znorm_space_bardist_[0]
                losses = loss_fn(logits, y_test_znorm.to(logits.device))
                # losses = losses.view((-1, batch_size))
                loss = torch_nanmean(losses.mean(dim=0))
                loss_step += loss
                loss.backward()
            
            progress_bar.set_postfix({"CUDA Memory": f"{torch.cuda.memory_allocated()/1024**2:.2f} MB"})
            optimizer.step()
            if new_optimizer:
                new_optimizer.step()
                
            optimizer.zero_grad()
            if new_optimizer:
                new_optimizer.zero_grad()

            if not torch.isnan(loss):
                total_loss += loss_step.cpu().detach().item()
                step_rows.append([epoch, step, float(loss_step.item())])

        if step_rows:
            with open(step_loss_csv_path, mode="a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(step_rows)
        
        return total_loss / steps_per_epoch

    def update_best_metrics(
        metric_name: str, best_metrics: list[float], metrics: list[float], 
        eval_csv_path: str, epoch_start: int
    ) -> tuple[list[float], list[bool]]:
        if not best_metrics:
            if epoch_start == 0:
                best_metrics = metrics
                is_best = [True] * len(metrics)
                return best_metrics, is_best
            else:
                eval_df = pd.read_csv(eval_csv_path)
                eval_df = eval_df[eval_df["epoch"] < epoch_start]
                dataset_order = eval_df["dataset"].drop_duplicates().tolist()
                best_metrics = [float(eval_df[eval_df["dataset"] == ds][metric_name].max()) for ds in dataset_order]
        
        is_best = []
        for idx, each_metric in enumerate(metrics):
            if each_metric > best_metrics[idx]:
                is_best.append(True)
                best_metrics[idx] = each_metric
            else:
                is_best.append(False)
        return best_metrics, is_best

    eval_csv_path = DIR_PATH / "logs" / f"ID_{ID}" / f"eval_metrics-ID_{ID}.csv"
    best_r2 = []
    total_loss = float('-inf')
    for epoch in range(epoch_start, epochs + 1):
        # breakpoint()
        if epoch > epoch_start:
            epoch_start_time = time.time()

            total_loss = train_epoch(epoch)

            print('-' * 100)
            print(f'| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s | mean loss {total_loss:5.2f}')
            print('-' * 100)

            if scheduler: 
                scheduler.step()
            if new_scheduler: 
                new_scheduler.step()
        
        mse, mae, r2, max_ae, std_error = evaluate_function(regressor=model)
        best_r2, is_best = update_best_metrics(
            metric_name="r2", best_metrics=best_r2, metrics=r2, 
            eval_csv_path=eval_csv_path, epoch_start=epoch_start,
        )
        any_best = np.array(is_best).any()
        # if any_best:
        if epoch > epoch_start or epoch_start == 0:
            save_model_checkpoint(regressor=model, id=ID, epoch=epoch)
        new_scheduler_lr = new_scheduler.get_last_lr()[0] if new_scheduler else None
        log_function(
            epoch=epoch, mse=mse, mae=mae, r2=r2, max_ae=max_ae, std_error=std_error, 
            is_best=is_best, new_scheduler_lr=new_scheduler_lr,
            write_to_file=epoch > epoch_start,
        )


def main():
    """Main function to configure and run the training workflow."""
    prior_config, model_config, dataset_config = init_config()
    prior_hyperparameters = init_prior_hyperparameters(prior_config)
    model_proto, extra_kwargs = init_model_proto(prior_config)

    regressor, regressor_config = setup_regressor(model_config)

    log_file_path, step_loss_csv_path, eval_csv_path = setup_logging(model_config)
    log_epoch_fn = partial(
        log_epoch,
        config=dataset_config,
        log_file_path=log_file_path,
        eval_csv_path=eval_csv_path,
    )

    X_train_evaluation, y_train_evaluation, X_test_evaluation, y_test_evaluation = init_dataset(dataset_config, dataset_config["evaluation"])

    eval_config = {
        **regressor_config,
        "inference_config": {
            "SUBSAMPLE_SAMPLES": dataset_config["n_inference_context_samples"]
        },
    }
    evaluate_regressor_fn = partial(
        evaluate_regressor,
        eval_config=eval_config,
        X_train=X_train_evaluation,
        y_train=y_train_evaluation,
        X_test=X_test_evaluation,
        y_test=y_test_evaluation,
    )

    # 训练
    print("--- 3. Starting Training & Evaluation ---")
    train(
        model=regressor,
        prior_dataloader_class=model_proto.DataLoader,
        real_dataloader=None,
        device=prior_config['device'],
        bptt=prior_config['bptt'],
        bptt_extra_samples=prior_config['bptt_extra_samples'],
        steps_per_epoch=model_config['num_steps'],
        batch_size=model_config['batch_size'],
        single_eval_pos_gen=get_uniform_single_eval_pos_sampler(
            prior_config.get('max_eval_pos', 
            prior_config['bptt']), 
            min_len=prior_config.get('min_eval_pos', 1)
        ),
        lr = model_config.get('layers', model_config['lr']),
        epoch_start = model_config['epoch_start'],
        epochs = model_config['epochs'],
        train_mixed_precision = prior_config['train_mixed_precision'],
        aggregate_k_gradients = prior_config['aggregate_k_gradients'],
        evaluate_function=evaluate_regressor_fn,
        ID=model_config['ID'],
        log_function=log_epoch_fn,
        step_loss_csv_path=step_loss_csv_path,
        extra_prior_kwargs_dict={
            'num_features': prior_config['num_features'], 
            'hyperparameters': prior_hyperparameters, 
            'batch_size_per_gp_sample': prior_config.get('batch_size_per_gp_sample', None), 
            **extra_kwargs
        },
    )


if __name__ == "__main__":
    main()
