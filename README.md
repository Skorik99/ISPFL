# Progress-Guided Client Participation in Federated Learning

This repository contains the code used for the ISP paper and related experiments.

All commands below are intended to be launched from the repository root:

```bash
cd /path/to/repo/ISPFL
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

Download the datasets you need:

```bash
python src/utils/cifar_download.py --target_dir=cifar10
python src/utils/image_net_download.py --target_dir=image_net
python src/utils/shakespeare_download.py --target_dir=shakespeare
```

If needed, adjust dataset paths in `src/configs/observed_data_params/`.

## General Notes

- The standard CIFAR-10 setup uses `dataset.alpha=0.1` and `federated_params.amount_of_clients=100`.
- ISP and AdaFL runs require `+federated_method.args.num_clients_subset=20`.
- FedProx uses `federated_method.mu=0.01` by default in this repository.
- `DELTA` uses `optimizer=sgd`.
- `FedCor` inside ISP uses `federated_method.num_samples=1`.

## CIFAR-10 Baselines

Use the following base command and change only `federated_method=...`:

```bash
nohup python src/train.py dataset.alpha=0.1 federated_params.amount_of_clients=100 training_params.batch_size=64 training_params.device_ids=[0] federated_params.print_client_metrics=False random_state=41 manager.batch_size=25 federated_params.communication_rounds=2000 federated_method=fedavg > outputs/ISP_experiments/cifar10_std/run.txt
```

Available values:

- `fedavg`
- `fedprox`
- `fedcbs`
- `pow`
- `fedcor`
- `delta` together with `optimizer=sgd`

Method-specific changes:

- `fedprox`: default `federated_method.mu=0.01`
- `fedcbs`: default `federated_method.lambda_=10`
- `pow`: default `federated_method.candidate_set_size=40`
- `fedcor`: default `federated_method.warmup=12`
- `delta`: add `optimizer=sgd` or keep defaults `federated_method.alpha_1=0.8`, `federated_method.alpha_2=0.2`

## ISP Experiments

Start from:

```bash
nohup python src/train.py dataset.alpha=0.1 federated_params.amount_of_clients=100 training_params.batch_size=64 training_params.device_ids=[0] federated_params.print_client_metrics=False random_state=41 manager.batch_size=25 federated_params.communication_rounds=2000 federated_method=ISP federated_method.base_method=fedavg +federated_method.args.num_clients_subset=20 > outputs/ISP_experiments/cifar10_std/isp_run.txt
```

Then modify only the method-specific part:

- `FedAvg`: keep `federated_method.base_method=fedavg`
- `FedProx`: set `federated_method.base_method=fedprox` and add `+federated_method.args.mu=0.01`
- `FedCBS`: set `federated_method.base_method=fedcbs` and add `+federated_method.args.lambda_=10`
- `DELTA`: set `federated_method.base_method=delta`, add `optimizer=sgd`, `+federated_method.args.alpha_1=0.8`, `+federated_method.args.alpha_2=0.2`
- `POW`: set `federated_method.base_method=pow` and add `+federated_method.args.candidate_set_size=40`
- `FedCor`: set `federated_method.base_method=fedcor`, add `federated_method.num_samples=1` and `+federated_method.args.warmup=12`

## AdaFL Experiments

Start from:

```bash
nohup python src/train.py dataset.alpha=0.1 federated_params.amount_of_clients=100 training_params.batch_size=64 training_params.device_ids=[0] federated_params.print_client_metrics=False random_state=41 federated_params.communication_rounds=2000 federated_method=adafl federated_method.base_method=fedavg +federated_method.args.num_clients_subset=20 > outputs/ISP_experiments/cifar10_std/adafl_run.txt
```

Then use the same method-specific substitutions as for ISP:

- `fedavg`
- `fedcbs` with `+federated_method.args.lambda_=10`
- `delta` with `optimizer=sgd`, `+federated_method.args.alpha_1=0.8`, `+federated_method.args.alpha_2=0.2`
- `pow` with `+federated_method.args.candidate_set_size=40`
- `fedcor` with `+federated_method.args.warmup=12`

## ISP Ablations

Use the ISP command above and change only the relevant parameter:

- Monte Carlo samples: `federated_method.num_samples=10`
- Search period: `federated_method.step_find_optimal=20`
- Momentum: `federated_method.num_clients_momentum=0.5`
- Disable EMA: `federated_method.ema_use=False`
- Corrupted trust set: `federated_method.trust_available=True federated_method.corrupted_trust=True`
- Fixed full-communication budget: `federated_method.full_comm_client_amount=60`
- Multiplier-based full communication: `federated_method.full_comm_client_amount=null federated_method.full_comm_amount_cl_multiplayer=2`

## Selection-Statistics Collection

To collect the client-selection statistics used in the paper, add:

```bash
sampling_statistics.enabled=True
```

The statistics are collected automatically inside ISP at every `find_optimal()` call.

## Gradient Compression

Compression baseline:

```bash
nohup python src/train.py dataset.alpha=1000 federated_params.amount_of_clients=100 training_params.batch_size=64 training_params.device_ids=[0] federated_params.print_client_metrics=False federated_params.communication_rounds=1500 random_state=41 federated_method=compression federated_method.compression_type=topk federated_method.compression_k_percent=5 > outputs/ISP_experiments/grad_compression/compression_topk_cifar10.txt
```

For the ISP version, switch to:

- `federated_method=ISP`
- `federated_method.base_method=compression`
- `+federated_method.args.num_clients_subset=20`
- `+federated_method.args.compression_type=topk`
- `+federated_method.args.compression_k_percent=5`
- `federated_method.warmup_rounds=5`
- `federated_method.local_epoch_on_wp=1`

To run `randk`, replace `topk` with `randk` and adjust `compression_k_percent`.

## ImageNet

FedCor on ImageNet:

```bash
nohup python src/train.py models@models_dict.model1=swin_tiny_patch4_window7_224 observed_data_params@dataset=image_net dataset.alpha=0.5 observed_data_params@server_test=image_net observed_data_params@trust_df=image_net_trust federated_params.amount_of_clients=100 training_params.batch_size=128 training_params.device_ids=[0] federated_params.print_client_metrics=False federated_params.communication_rounds=200 random_state=41 optimizer.lr=0.001 optimizer.weight_decay=0.05 manager.batch_size=5 federated_params.round_epochs=1 optimizer=adamw federated_method=fedcor federated_method.warmup=10 > outputs/ISP_experiments/image_net/fedcor_imagenet.txt
```

For ISP on top of this setup, switch to:

- `federated_method=ISP`
- `federated_method.base_method=fedcor`
- `+federated_method.args.num_clients_subset=20`
- `federated_method.num_samples=1`