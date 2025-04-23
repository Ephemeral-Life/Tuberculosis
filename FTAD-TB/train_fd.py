import os
import json
import random
import time
import torch
import math
import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Optional, Dict
from mmcv import Config
from mmcv.utils import Registry, build_from_cfg
from mmdet.datasets import CocoDataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, set_random_seed
from pycocotools.coco import COCO
import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context, Parameters, Scalar
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

import warnings
warnings.filterwarnings("ignore")  # Suppress warnings
import logging
logging.basicConfig(level=logging.WARNING)

# Load configuration file
cfg = Config.fromfile('FTAD-TB/configs/symformer/symformer_retinanet_p2t_cls_fpn_1x_TBX11K.py')

# Set training epochs for quick validation
cfg.runner = dict(type='EpochBasedRunner', max_epochs=cfg.max_epochs)

# Set GPU ID (consistent with original project)
cfg.gpu_ids = [0]

# Reduce batch size to save memory
cfg.data.samples_per_gpu = 2  # Reduced from default 8 to 4

# Configure optimizer, retaining only grad_clip
cfg.optimizer_config = dict(grad_clip=None)

# Register DATASETS and PIPELINES
DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
DATASETS.register_module(CocoDataset)

# Build the full training dataset
train_dataset = build_from_cfg(cfg.data.train, DATASETS, default_args=None)

# Get all image IDs
image_ids = train_dataset.coco.getImgIds()

# Shuffle image IDs for randomness
random.shuffle(image_ids)

# Split image IDs into subsets for clients
num_clients = cfg.num_clients
random.shuffle(image_ids)
partition_size = len(image_ids) // num_clients
partitions = [image_ids[i * partition_size:(i + 1) * partition_size] for i in range(num_clients)]

# Create temporary directory for client annotation files
temp_dir = 'temp_client_ann'
os.makedirs(temp_dir, exist_ok=True)

# Function to generate temporary annotation file for each client
def create_client_ann_file(partition_id: int, client_image_ids: List[int]) -> str:
    coco = COCO(cfg.data.train.ann_file)
    client_imgs = [img for img in coco.imgs.values() if img['id'] in client_image_ids]
    client_ann_ids = coco.getAnnIds(imgIds=client_image_ids)
    client_anns = [coco.anns[ann_id] for ann_id in client_ann_ids]
    client_coco = {
        'images': client_imgs,
        'annotations': client_anns,
        'categories': coco.dataset['categories'],
        'info': coco.dataset.get('info', {}),
        'licenses': coco.dataset.get('licenses', [])
    }
    temp_ann_file = os.path.join(temp_dir, f'client_{partition_id}_ann.json')
    with open(temp_ann_file, 'w') as f:
        json.dump(client_coco, f)
    return temp_ann_file

# Define Flower client class
class FlowerClient(NumPyClient):
    def __init__(self, partition_id: int):
        self.partition_id = partition_id
        self.net = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )
        self.net.init_weights()
        self.trainloader = self.load_data()
        print(f"Client {self.partition_id} initialized with dataset size: {len(self.trainloader)}")

    def load_data(self):
        client_image_ids = partitions[self.partition_id]
        client_ann_file = create_client_ann_file(self.partition_id, client_image_ids)
        client_cfg = dict(
            type='CocoDataset',
            ann_file=client_ann_file,
            img_prefix=cfg.data.train.img_prefix,
            pipeline=cfg.data.train.pipeline,
            filter_empty_gt=cfg.data.train.filter_empty_gt,
            classes=cfg.data.train.classes
        )
        try:
            client_dataset = build_from_cfg(client_cfg, DATASETS, default_args=None)
            print(f"Client {self.partition_id} loaded dataset with {len(client_dataset)} images")
            return client_dataset
        except Exception as e:
            print(f"Client {self.partition_id} failed to load dataset: {e}")
            raise

    def get_parameters(self, config) -> List[np.ndarray]:
        params = [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        print(f"Client {self.partition_id} returning {len(params)} parameters")
        return params

    def set_parameters(self, parameters: List[np.ndarray]):
        state_dict = self.net.state_dict()
        params_dict = zip(state_dict.keys(), parameters)
        new_state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        try:
            self.net.load_state_dict(new_state_dict, strict=True)
            print(f"Client {self.partition_id} parameters set successfully")
        except Exception as e:
            print(f"Client {self.partition_id} failed to set parameters: {e}")
            raise

    def fit(self, parameters: List[np.ndarray], config) -> Tuple[List[np.ndarray], int, dict]:
        server_round = config.get("server_round", 1)  # Get current round from config, default to 1
        if server_round == 1:
            print(f"Client {self.partition_id} using load_from for round 1")
            # Load model from cfg.load_from for the first round
            if cfg.load_from:
                try:
                    self.net.load_state_dict(torch.load(cfg.load_from), strict=True)
                    print(f"Client {self.partition_id} loaded model from {cfg.load_from}")
                except Exception as e:
                    print(f"Client {self.partition_id} failed to load model from {cfg.load_from}: {e}")
                    raise
        else:
            print(f"Client {self.partition_id} using server parameters for round {server_round}")
            self.set_parameters(parameters)  # Use server-provided global model parameters

        seed = cfg.get('seed', None)
        if seed is not None:
            set_random_seed(seed)
        try:
            print(f"Client {self.partition_id} starting training for round {server_round}...")
            train_detector(
                self.net,
                [self.trainloader],
                cfg,
                distributed=False,
                validate=False,
                timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
                meta={'seed': seed}
            )
            print(f"Client {self.partition_id} training completed for round {server_round}")
        except Exception as e:
            print(f"Client {self.partition_id} training failed for round {server_round}: {e}")
            raise
        params = self.get_parameters(config)
        return params, len(self.trainloader), {}

    def evaluate(self, parameters: List[np.ndarray], config) -> Tuple[float, int, dict]:
        self.set_parameters(parameters)
        return 0.0, len(self.trainloader), {"accuracy": 0.0}

# Define client_fn
def client_fn(context: Context) -> Client:
    partition_id = context.node_config["partition-id"]
    return FlowerClient(partition_id).to_client()

# Create ClientApp
client = ClientApp(client_fn=client_fn)

# Define weighted_average function for metrics aggregation
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples) if sum(examples) > 0 else 0.0}

# Custom FedAvg strategy to save aggregated models and support continued training
class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )
        self.start_round = 1  # Default start from round 1

        # Check for existing aggregated model files
        try:
            model_files = [f for f in os.listdir(cfg.work_dir) if f.startswith("aggregated_model_round_") and f.endswith(".pth")]
        except FileNotFoundError:
            model_files = None
        if model_files:
            rounds = [int(f.split('_')[-1].split('.')[0]) for f in model_files]
            latest_round = max(rounds)
            latest_model_path = os.path.join(cfg.work_dir, f"aggregated_model_round_{latest_round}.pth")
            print(f"Loading latest aggregated model from {latest_model_path}")
            state_dict = torch.load(latest_model_path)
            self.model.load_state_dict(state_dict, strict=True)
            self.start_round = latest_round + 1  # Resume from next round
        else:
            print("No aggregated model found, starting from scratch.")
            self.model.init_weights()

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes]],
            failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print(f"Server round {server_round}: aggregating {len(results)} results, {len(failures)} failures")
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            try:
                aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
                params_dict = zip(self.model.state_dict().keys(),
                                  [torch.from_numpy(np.copy(p)) for p in aggregated_ndarrays])
                state_dict = OrderedDict({k: v for k, v in params_dict})
                self.model.load_state_dict(state_dict, strict=True)
                # Calculate actual global round
                actual_round = self.start_round + server_round - 1
                model_path = os.path.join(cfg.work_dir, f"aggregated_model_round_{actual_round}.pth")
                torch.save(self.model.state_dict(), model_path)
                print(f"Saved aggregated model to {model_path}")
            except Exception as e:
                print(f"Error saving aggregated model: {e}")
        else:
            print(f"No aggregated parameters for round {server_round}, results: {len(results)}, failures: {len(failures)}")
        return aggregated_parameters, metrics

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        # Pass server_round to clients via config
        config = {"server_round": server_round}
        if server_round == 1 and self.start_round == 1:
            # First round with no prior aggregated model; clients use load_from
            pass
        else:
            # Subsequent rounds or resumed training; send aggregated model parameters
            state_dict = self.model.state_dict()
            ndarrays = [val.cpu().numpy() for val in state_dict.values()]
            parameters = ndarrays_to_parameters(ndarrays)

        fit_config = super().configure_fit(server_round, parameters, client_manager)
        for client, fit in fit_config:
            fit.config.update(config)
        return fit_config

# Define server_fn
def server_fn(context: Context) -> ServerAppComponents:
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=math.ceil(num_clients / 2),
        min_evaluate_clients=math.ceil(num_clients / 2),
        min_available_clients=math.ceil(num_clients / 2),
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    # Adjust num_rounds based on completed rounds
    start_round = strategy.start_round if hasattr(strategy, 'start_round') else 1
    num_rounds = cfg.num_rounds
    if start_round > 1:
        num_rounds -= (start_round - 1)
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

# Specify client resources
backend_config = {"client_resources": {"num_cpus": 1.0, "num_gpus": 0.0}}
if torch.cuda.is_available():
    backend_config["client_resources"] = {"num_cpus": 1.0, "num_gpus": 1.0}

# Run simulation
if __name__ == "__main__":
    print("Starting simulation...")
    run_simulation(
        server_app=ServerApp(server_fn=server_fn),
        client_app=client,
        num_supernodes=num_clients,
        backend_config=backend_config,
    )
