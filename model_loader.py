import os
from functools import partial

import gin

import cifar10.model_loader
from eval.eval_config import all_experiments
from eval.eval_util import DummyEnvData, get_checkpoint_and_config
from model.pl_util import LearnerLTC


def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)
    else:
        from experiments.bulletEnv import Dataset
        directory = all_experiments[dataset]["root_dir"]
        model_file, gin_file = get_checkpoint_and_config(directory, 16, model_name)
        gin.clear_config()
        gin.parse_config_files_and_bindings([gin_file] if gin_file else None,
                                            None,  # args.gin_bindings_common,
                                            skip_unknown=True)
        data_cls = partial(Dataset, dataset)
        net = LearnerLTC.load_from_checkpoint(model_file, strict=False, data_cls=data_cls, seed=0)
    return net
