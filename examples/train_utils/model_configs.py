from ConfigSpace import hyperparameters as CSH 
import ConfigSpace as CS
from train_utils.priors.utils import uniform_int_sampler_f, gamma_sampler_f
import torch
from copy import deepcopy

def get_general_config(max_features: int, bptt: int) -> dict:
    """"
    Returns the general PFN training hyperparameters.
    """
    config_general = {
        "lr": CSH.UniformFloatHyperparameter('lr', lower=0.0001, upper=0.00015, log=True),
        "dropout": CSH.CategoricalHyperparameter('dropout', [0.0]),
        "emsize": 2 ** 9, # 512
        # "emsize": CSH.CategoricalHyperparameter('emsize', [2 ** i for i in range(8, 9)]), ## upper bound is -1
        "batch_size": 8,
        # "batch_size": CSH.CategoricalHyperparameter('batch_size', [2 ** i for i in range(6, 8)]),
        "nlayers": CSH.CategoricalHyperparameter('nlayers', [12]),
        "num_features": max_features,
        "nhead": 4,
        # "nhead": CSH.CategoricalHyperparameter('nhead', [4]),
        "nhid_factor": 2,
        "bptt": bptt,
        "eval_positions": [int(bptt * 0.95)],
        # "eval_positions": None,
        "seq_len_used": bptt,
        "sampling": 'normal',
        "epochs": 400,
        # "epochs": 12000,
        "num_steps": 1024,
        # "num_steps": 100,
        "verbose": False,
        "mix_activations": False,
        "pre_sample_causes": True,
        "multiclass_type": 'rank'
    }

    return config_general


def get_flexible_categorical_config(max_features: int) -> dict:
    """"
    Returns the configuration parameters for the tabular multiclass wrapper.
    """
    config_flexible_categorical = {
        "nan_prob_unknown_reason_reason_prior": CSH.CategoricalHyperparameter('nan_prob_unknown_reason_reason_prior', [0.5]),
        "categorical_feature_p": 0.2,
        # "categorical_feature_p": CSH.CategoricalHyperparameter('categorical_feature_p', [0.0, 0.1, 0.2]),
        "nan_prob_no_reason": 0.0,
        # "nan_prob_no_reason": CSH.CategoricalHyperparameter('nan_prob_no_reason', [0.0, 0.1]),
        "nan_prob_unknown_reason": 0.0,
        # "nan_prob_unknown_reason": CSH.CategoricalHyperparameter('nan_prob_unknown_reason', [0.0]),
        "nan_prob_a_reason": CSH.CategoricalHyperparameter('nan_prob_a_reason', [0.0]),
        "max_num_classes": 10,
        "num_classes": 2,
        "noise_type": CSH.CategoricalHyperparameter('noise_type', ["Gaussian"]),
        "balanced": False,
        # "balanced": True,
        "normalize_to_ranking": False,
        # "normalize_to_ranking": CSH.CategoricalHyperparameter('normalize_to_ranking', [False]),
        "set_value_to_nan": 0.1,
        # "set_value_to_nan": CSH.CategoricalHyperparameter('set_value_to_nan', [0.5, 0.2, 0.0]),
        "normalize_by_used_features": True,
        "num_features_used":
            {'uniform_int_sampler_f(3,max_features)': uniform_int_sampler_f(1, max_features)}
    }
    return config_flexible_categorical


def get_diff_flex() -> dict:
    """"
    Returns the configuration parameters for a differentiable wrapper around the tabular multiclass wrapper.
    """
    diff_flex = {
        "output_multiclass_ordered_p": {'distribution': 'uniform', 'min': 0.0, 'max': 0.5},
        "multiclass_type": {'distribution': 'meta_choice', 'choice_values': ['value', 'rank']},
    }

    return diff_flex


def get_diff_gp() -> dict:
    """"
    Returns the configuration parameters for a differentiable wrapper around GP.
    """
    diff_gp = {
        'outputscale': {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 10., 'min_mean': 0.00001, 'round': False,
                        'lower_bound': 0},
        'lengthscale': {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 10., 'min_mean': 0.00001, 'round': False,
                        'lower_bound': 0},
        'noise': {'distribution': 'meta_choice', 'choice_values': [0.00001, 0.0001, 0.01]}
    }

    return diff_gp


def get_diff_causal() -> dict:
    """"
    Returns the configuration parameters for a differentiable wrapper around MLP / Causal mixture.
    """
    diff_causal = {
        "num_layers": {'distribution': 'meta_gamma', 'max_alpha': 2, 'max_scale': 3, 'round': True,
                       'lower_bound': 2},
        "prior_mlp_hidden_dim": {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 100, 'round': True, 
                                 'lower_bound': 4},
        "prior_mlp_dropout_prob": {'distribution': 'meta_beta', 'scale': 0.6, 'min': 0.1, 'max': 5.0},

        "noise_std": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': .3, 'min_mean': 0.0001, 'round': False,
                      'lower_bound': 0.0},
        "init_std": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 10.0, 'min_mean': 0.01, 'round': False,
                     'lower_bound': 0.0},
        "num_causes": {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 7, 'round': True,
                       'lower_bound': 2},
        "is_causal": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "pre_sample_weights": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "y_is_effect": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "sampling": {'distribution': 'meta_choice', 'choice_values': ['normal', 'mixed']},
        "prior_mlp_activations": {'distribution': 'meta_choice_mixed', 'choice_values': [
            torch.nn.Tanh
            , torch.nn.Identity
            , torch.nn.ReLU
        ]},
        "block_wise_dropout": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "sort_features": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "in_clique": {'distribution': 'meta_choice', 'choice_values': [True, False]},
    }

    return diff_causal


def get_diff_prior_bag() -> dict:
    """"
    Returns the configuration parameters for a GP and MLP / Causal mixture.
    """
    diff_prior_bag = {
        'prior_bag_exp_weights_1': {'distribution': 'uniform', 'min': 2.0, 'max': 10.0},
    }

    return diff_prior_bag


def get_diff_config() -> dict:
    """"
    Returns the configuration parameters for a differentiable wrapper around GP and MLP / Causal mixture priors.
    """
    diff_prior_bag = get_diff_prior_bag()
    diff_causal = get_diff_causal()
    diff_gp = get_diff_gp()
    diff_flex = get_diff_flex()

    config_diff = {'differentiable_hyperparameters': {**diff_prior_bag, **diff_causal, **diff_gp, **diff_flex}}

    return config_diff


def get_prior_config_causal(max_features: int = 100) -> dict:
    config_general = get_general_config(max_features, 1024 + 128) # 1024 for bptt, 128 for extra samples

    config_flexible_categorical = get_flexible_categorical_config(max_features)
    config_flexible_categorical['num_categorical_features_sampler_a'] = -1.0  # Categorical features disabled by default
    config_flexible_categorical['num_classes'] = uniform_int_sampler_f(2, config_flexible_categorical['max_num_classes'])

    config_diff = get_diff_config()
    del config_diff['differentiable_hyperparameters']['output_multiclass_ordered_p']
    del config_diff['differentiable_hyperparameters']['multiclass_type']
    del config_diff['differentiable_hyperparameters']['sampling']

    config = {**config_general, **config_flexible_categorical, **config_diff}
    config['output_multiclass_ordered_p'] = 0.

    return config


def fill_in_configsample(config: dict, configsample: CS.Configuration) -> dict:
    hierarchical_configsample = deepcopy(config)
    for k, v in config.items():
        if isinstance(v, CSH.Hyperparameter):
            hierarchical_configsample[k] = configsample[v.name]
        elif isinstance(v, dict):
            hierarchical_configsample[k] = fill_in_configsample(v, configsample)
    return hierarchical_configsample


def list_all_hps_in_nested(config: dict) -> list:
    """"
    Returns a list of hyperparameters from a neszed dict of hyperparameters.
    """

    if isinstance(config, CSH.Hyperparameter):
        return [config]
    elif isinstance(config, dict):
        result = []
        for k, v in config.items():
            result += list_all_hps_in_nested(v)
        return result
    else:
        return []


def sample_hypers(config: dict) -> dict:
    cs = CS.ConfigurationSpace()
    for hp in list_all_hps_in_nested(config):
        cs.add(hp)
    cs_sample = cs.sample_configuration()
    return fill_in_configsample(config, cs_sample)


def get_mlp_prior_hyperparameters(config):
    config = {hp: (list(config[hp].values())[0]) if type(config[hp]) is dict else config[hp] for hp in config}

    if 'random_feature_rotation' not in config:
        config['random_feature_rotation'] = True

    if "prior_sigma_gamma_k" in config:
        sigma_sampler = gamma_sampler_f(config["prior_sigma_gamma_k"], config["prior_sigma_gamma_theta"])
        config['init_std'] = sigma_sampler
    if "prior_noise_std_gamma_k" in config:
        noise_std_sampler = gamma_sampler_f(config["prior_noise_std_gamma_k"], config["prior_noise_std_gamma_theta"])
        config['noise_std'] = noise_std_sampler

    return config


def get_gp_prior_hyperparameters(config):
    return {hp: (list(config[hp].values())[0]) if type(config[hp]) is dict else config[hp] for hp in config}
