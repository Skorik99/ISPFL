import os
import hydra
import random
import signal
from functools import partial
from omegaconf import DictConfig
from hydra.utils import instantiate

from utils.data_utils import prepare_df_for_federated_training
from utils.utils import handle_main_process_sigterm
from utils.logging_utils import redirect_stdout_to_log

# Make print with flush=True by default
print = partial(print, flush=True)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    redirect_stdout_to_log()
    df, cfg = prepare_df_for_federated_training(cfg, "train_directories")
    # Init federated_method and begin train
    trainer = instantiate(cfg.federated_method, _recursive_=False)
    temp_trainer = trainer._init_federated(cfg, df)
    if temp_trainer is not None:
        trainer = temp_trainer # If trainer changed in init_federated
        
    # Termination handling in multiprocess setup
    signal.signal(
        signal.SIGTERM,
        lambda signum, frame: handle_main_process_sigterm(signum, frame, trainer),
    )
    trainer.begin_train()


if __name__ == "__main__":
    train()
