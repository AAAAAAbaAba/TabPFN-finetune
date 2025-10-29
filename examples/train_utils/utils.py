import os
import torch
import datetime
import random
from typing import Callable, Any, Literal
import math
from torch.optim.lr_scheduler import LambdaLR
from tabpfn.utils import infer_random_state
from tabpfn.preprocessing import (
    BaseDatasetConfig,
    ClassifierDatasetConfig,
    DatasetCollectionWithPreprocessing,
    RegressorDatasetConfig,
)
from sklearn.model_selection import train_test_split
from functools import partial

def print_on_master_only(is_master: bool):
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_dist(device: str) -> tuple[bool, int, str]:
    #print('init dist')
    if 'LOCAL_RANK' in os.environ:
        # launched with torch.distributed.launch
        rank = int(os.environ["LOCAL_RANK"])
        print('torch.distributed.launch and my rank is', rank)
        torch.cuda.set_device(rank)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=20),
                                             world_size=torch.cuda.device_count(), rank=rank)
        torch.distributed.barrier()
        print_on_master_only(rank == 0)
        print(f"Distributed training on {torch.cuda.device_count()} GPUs, this is rank {rank}, "
              "only I can print, but when using print(..., force=True) it will print on all ranks.")
        return True, rank, f'cuda:{rank}'
    elif 'SLURM_PROCID' in os.environ and torch.cuda.device_count() > 1:
        # this is for multi gpu when starting with submitit
        assert device != 'cpu:0'
        rank = int(os.environ['SLURM_PROCID'])
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.cuda.set_device(rank)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
        print('distributed submitit launch and my rank is', rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=20),
                                             world_size=torch.cuda.device_count(), rank=rank)
        torch.distributed.barrier()
        print_on_master_only(rank == 0)
        print(f"Distributed training on {torch.cuda.device_count()} GPUs, this is rank {rank}, "
              "only I can print, but when using print(..., force=True) it will print on all ranks.")

        return True, rank, f'cuda:{rank}'
    else:
        # print('Not using distributed')
        # will not change any of the behavior of print, but allows putting the force=True in the print calls
        print_on_master_only(True)
        return False, 0, device


def get_weighted_single_eval_pos_sampler(max_len: int) -> Callable[[], int]:
    """
    This gives a sampler that can be used for `single_eval_pos` which yields good performance for all positions p,
    where p <= `max_len`. At most `max_len` - 1 examples are shown to the Transformer.
    :return: Sampler that can be fed to `train()` as `single_eval_pos_gen`.
    """
    return lambda: random.choices(range(max_len), [1 / (max_len - i) for i in range(max_len)])[0]


def get_uniform_single_eval_pos_sampler(max_len: int, min_len: int = 0) -> Callable[[], int]:
    """
    Just sample any evaluation position with the same weight
    :return: Sampler that can be fed to `train()` as `single_eval_pos_gen`.
    """
    return lambda: random.choices(range(min_len, max_len))[0]


def eval_pos_seq_len_sampler(single_eval_pos_gen: Callable[[], int], bptt_extra_samples: int = None, bptt: int = 10) -> tuple[int, int]:
    single_eval_pos = single_eval_pos_gen()
    if bptt_extra_samples is not None:
        seq_len = single_eval_pos + bptt_extra_samples
    else:
        seq_len = bptt
    return single_eval_pos, seq_len


def set_locals_in_self(locals: dict):
    """
    Call this function like `set_locals_in_self(locals())` to set all local variables as object variables.
    Especially useful right at the beginning of `__init__`.
    :param locals: `locals()`
    """
    self = locals['self']
    for var_name, val in locals.items():
        if var_name != 'self': setattr(self, var_name, val)


def normalize_by_used_features_f(x: torch.Tensor, num_features_used: int, num_features: int, normalize_with_sqrt: bool = False) -> torch.Tensor:
    if normalize_with_sqrt:
        return x / (num_features_used / num_features)**(1 / 2)
    return x / (num_features_used / num_features)


def normalize_data(data: torch.Tensor, normalize_positions: int = -1) -> torch.Tensor:
    if normalize_positions > 0:
        mean = torch_nanmean(data[:normalize_positions], dim=0)
        std = torch_nanstd(data[:normalize_positions], dim=0) + .000001
    else:
        mean = torch_nanmean(data, dim=0)
        std = torch_nanstd(data, dim=0) + .000001
    data = (data - mean) / std
    data = torch.clip(data, min=-100, max=100)

    return data


def torch_nanmean(x: torch.Tensor, dim: int = 0, return_nanshare: bool = False) -> torch.Tensor:
    return torch_masked_mean(x, ~torch.isnan(x), dim=dim, return_share_of_ignored_values=return_nanshare)


def torch_nanstd(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch_masked_std(x, ~torch.isnan(x), dim=dim)


def torch_masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 0, return_share_of_ignored_values: bool = False) -> torch.Tensor:
    """
    Returns the mean of a torch tensor and only considers the elements, where the mask is true.
    If return_share_of_ignored_values is true it returns a second tensor with the percentage of ignored values
    because of the mask.
    """
    num = torch.where(mask, torch.full_like(x, 1), torch.full_like(x, 0)).sum(dim=dim)
    value = torch.where(mask, x, torch.full_like(x, 0)).sum(dim=dim)
    if return_share_of_ignored_values:
        return value / num, 1.-num/x.shape[dim]
    return value / num
    

def torch_masked_std(x: torch.Tensor, mask: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Returns the std of a torch tensor and only considers the elements, where the mask is true.
    If get_mean is true it returns as a first Tensor the mean and as a second tensor the std.
    """
    num = torch.where(mask, torch.full_like(x, 1), torch.full_like(x, 0)).sum(dim=dim)
    value = torch.where(mask, x, torch.full_like(x, 0)).sum(dim=dim)
    mean = value / num
    mean_broadcast = torch.repeat_interleave(mean.unsqueeze(dim), x.shape[dim], dim=dim)
    quadratic_difference_from_mean = torch.square(torch.where(mask, mean_broadcast - x, torch.full_like(x, 0)))
    return torch.sqrt(torch.sum(quadratic_difference_from_mean, dim=dim) / (num - 1))


def get_nan_value(v: float, set_value_to_nan: float = 0.0) -> float:
    if random.random() < set_value_to_nan:
        return v
    else:
        return random.choice([-999, 0, 1, 999])


def nan_handling_missing_for_unknown_reason_value(set_value_to_nan: float = 0.0) -> Callable[[torch.Tensor], torch.Tensor]:
    return get_nan_value(float('nan'), set_value_to_nan)


def nan_handling_missing_for_no_reason_value(set_value_to_nan: float = 0.0) -> Callable[[torch.Tensor], torch.Tensor]:
    return get_nan_value(float('-inf'), set_value_to_nan)


def nan_handling_missing_for_a_reason_value(set_value_to_nan: float = 0.0) -> Callable[[torch.Tensor], torch.Tensor]:
    return get_nan_value(float('inf'), set_value_to_nan)


def to_ranking_low_mem(data: torch.Tensor) -> torch.Tensor:
    x = torch.zeros_like(data)
    for col in range(data.shape[-1]):
        x_ = (data[:, :, col] >= data[:, :, col].unsqueeze(-2))
        x_ = x_.sum(0)
        x[:, :, col] = x_
    return x


def remove_outliers(X: torch.Tensor, n_sigma: int = 4, normalize_positions: int = -1) -> torch.Tensor:
    # Expects T, B, H
    assert len(X.shape) == 3, "X must be T,B,H"

    data = X if normalize_positions == -1 else X[:normalize_positions]

    data_mean, data_std = torch_nanmean(data, dim=0), torch_nanstd(data, dim=0)
    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    mask = (data <= upper) & (data >= lower) & ~torch.isnan(data)
    data_mean, data_std = torch_masked_mean(data, mask), torch_masked_std(data, mask)

    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    X = torch.maximum(-torch.log(1+torch.abs(X)) + lower, X)
    X = torch.minimum(torch.log(1+torch.abs(X)) + upper, X)
            # print(ds[1][data < lower, col], ds[1][data > upper, col], ds[1][~np.isnan(data), col].shape, data_mean, data_std)
    return X


# copied from huggingface
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step + 1) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step + 1) / float(max(1, num_warmup_steps))
        else:
            return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'


def my_get_preprocessed_datasets(
    calling_instance: Any,  # Union[TabPFNClassifier, TabPFNRegressor],
    X: torch.Tensor, 
    y: torch.Tensor,
    test_size: float,
    model_type: Literal["classifier", "regressor"]
) -> tuple[torch.Tensor, torch.Tensor]:
    X = X.view(-1, X.shape[-1]).cpu()
    y = y.view(-1).cpu()
    split_fn = partial(train_test_split, test_size=test_size)

    if not hasattr(calling_instance, "model_") or calling_instance.model_ is None:
        _, rng = calling_instance._initialize_model_variables()
    else:
        _, rng = infer_random_state(calling_instance.random_state)

    dataset_config = list[BaseDatasetConfig]()
    if model_type == "classifier":
        ensemble_configs, X_mod, y_mod = (
            calling_instance._initialize_dataset_preprocessing(X, y, rng)
        )
        current_cat_ix = calling_instance.inferred_categorical_indices_
        dataset_config.append(ClassifierDatasetConfig(
            config=ensemble_configs,
            X_raw=X_mod,
            y_raw=y_mod,
            cat_ix=current_cat_ix,
        ))
    elif model_type == "regressor":
        ensemble_configs, X_mod, y_mod, bardist_ = (
            calling_instance._initialize_dataset_preprocessing(X, y, rng)
        )
        current_cat_ix = calling_instance.inferred_categorical_indices_
        dataset_config.append(RegressorDatasetConfig(
            config=ensemble_configs,
            X_raw=X_mod,
            y_raw=y_mod,
            cat_ix=current_cat_ix,
            znorm_space_bardist_=bardist_,
        ))
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    return DatasetCollectionWithPreprocessing(split_fn, rng, dataset_config)
