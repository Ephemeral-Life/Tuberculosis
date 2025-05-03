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
from mmdet.datasets import CocoDataset, build_dataset, build_dataloader
from mmdet.models import build_detector
from mmdet.apis import train_detector, set_random_seed, single_gpu_test
from pycocotools.coco import COCO
import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context, Parameters, Scalar
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from mmcv.parallel import MMDataParallel
from coco_classification import CocoClassificationDataset
import gc

import warnings

warnings.filterwarnings("ignore")
import logging

logging.basicConfig(level=logging.WARNING)

# Load configuration
cfg = Config.fromfile('FTAD-TB/configs/symformer/symformer_retinanet_p2t_cls_fpn_1x_TBX11K.py')
cfg.gpu_ids = [0]
num_clients = cfg.num_clients

# Register datasets
DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
DATASETS.register_module(CocoDataset)
DATASETS.register_module(CocoClassificationDataset)

# Pre-generated client annotation file paths
client_ann_files = [os.path.join('client_ann', f'client_{partition_id}_ann.json')
                    for partition_id in range(num_clients)]


# Flower client class
class FlowerClient(NumPyClient):
    def __init__(self, partition_id: int):
        self.partition_id = partition_id
        self.net = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )
        self.ann_file = client_ann_files[partition_id]
        self.trainloader = None
        self.tb_ratio = None
        self.image_size = None
        print(f"Client {self.partition_id} initialized, annotation file: {self.ann_file}")

    def freeze_parameters(self):
        for name, param in self.net.named_parameters():
            if "backbone" in name or "neck" in name or "bbox_head" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def load_data(self):
        client_cfg = dict(
            type='CocoClassificationDataset',
            ann_file=self.ann_file,
            img_prefix=cfg.data.train.img_prefix,
            pipeline=cfg.data.train.pipeline,
            classes=cfg.data.train.classes
        )
        try:
            client_dataset = build_from_cfg(client_cfg, DATASETS)
            print(f"Client {self.partition_id} loaded dataset, image count: {len(client_dataset)}")
            return client_dataset
        except Exception as e:
            print(f"Client {self.partition_id} failed to load dataset: {e}")
            raise

    def get_statistics(self):
        # Load COCO annotation
        coco = COCO(self.ann_file)
        images = coco.dataset['images']
        total_images = len(images)

        # Calculate TB image proportion
        tb_images = [img for img in images if 'tb/' in img['file_name']]
        tb_count = len(tb_images)
        self.tb_ratio = tb_count / total_images if total_images > 0 else 0

        # Get image size (assuming all images have the same size)
        if total_images > 0:
            self.image_size = (images[0]['width'], images[0]['height'])
        else:
            self.image_size = (0, 0)

        print(
            f"Client {self.partition_id} statistics: TB image ratio {self.tb_ratio:.4f}, image size {self.image_size}")
        del coco
        gc.collect()

    def get_parameters(self, config) -> List[np.ndarray]:
        with torch.no_grad():
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
        self.set_parameters(parameters)
        seed = cfg.get('seed', None)
        if seed is not None:
            set_random_seed(seed)

        self.trainloader = self.load_data()
        server_round = config.get("server_round", 1)
        print(f"Client {self.partition_id} training (round {server_round})...")

        # Calculate statistics before training in the first round
        if server_round == 1:
            self.get_statistics()

        self.freeze_parameters()
        original_load_from = cfg.get('load_from', None)
        cfg.load_from = None

        try:
            train_detector(
                self.net,
                [self.trainloader],
                cfg,
                distributed=False,
                validate=False,
                timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
                meta={'seed': seed}
            )
            print(f"Client {self.partition_id} training completed")
        except Exception as e:
            print(f"Client {self.partition_id} training failed: {e}")
            raise
        finally:
            cfg.load_from = original_load_from

        params = self.get_parameters(config)
        num_examples = len(self.trainloader)
        del self.trainloader
        self.trainloader = None
        gc.collect()

        # Return statistics in the first round
        metrics = {
            "tb_ratio": self.tb_ratio,
            "image_width": self.image_size[0],
            "image_height": self.image_size[1]
        } if server_round == 1 else {}

        return params, num_examples, metrics


# Client factory function
def client_fn(context: Context) -> Client:
    partition_id = context.node_config["partition-id"]
    return FlowerClient(partition_id).to_client()


# Create ClientApp
client = ClientApp(client_fn=client_fn)


# Weighted average for metrics
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples) if sum(examples) > 0 else 0.0}


# Custom FedAvg strategy
class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        self.model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )
        self.start_round = 1
        self.client_statistics = {}

        if cfg.get('load_from', None):
            try:
                checkpoint = torch.load(cfg.load_from)
                state_dict = checkpoint.get('state_dict', checkpoint)
                self.model.load_state_dict(state_dict, strict=False)
                print(f"Server loaded initial model: {cfg.load_from}")
            except Exception as e:
                print(f"Server failed to load initial model: {e}")
                raise
        else:
            print("No cfg.load_from specified, using randomly initialized model")
            self.model.init_weights()

        try:
            model_files = [f for f in os.listdir(cfg.work_dir) if
                           f.startswith("aggregated_model_round_") and f.endswith(".pth")]
        except FileNotFoundError:
            model_files = None
        if model_files:
            rounds = [int(f.split('_')[-1].split('.')[0]) for f in model_files]
            latest_round = max(rounds)
            latest_model_path = os.path.join(cfg.work_dir, f"aggregated_model_round_{latest_round}.pth")
            print(f"Loading latest aggregated model: {latest_model_path}")
            checkpoint = torch.load(latest_model_path)
            state_dict = checkpoint.get('state_dict', checkpoint)
            self.model.load_state_dict(state_dict, strict=True)
            self.start_round = latest_round + 1

        initial_parameters = ndarrays_to_parameters([val.cpu().numpy() for val in self.model.state_dict().values()])
        super().__init__(*args, initial_parameters=initial_parameters, **kwargs)

        val_cfg = cfg.data.val
        val_cfg['type'] = 'CocoClassificationDataset'
        try:
            self.val_dataset = build_dataset(val_cfg)
            self.val_dataloader = build_dataloader(
                self.val_dataset,
                samples_per_gpu=cfg.data.samples_per_gpu,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=False,
                shuffle=False
            )
            print("Server loaded evaluation dataset successfully")
        except Exception as e:
            print(f"Server failed to load evaluation dataset: {e}")
            raise

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        for instruction in client_instructions:
            instruction[1].config["server_round"] = server_round
        return client_instructions

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes]],
            failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print(f"Server round {server_round}: Aggregating {len(results)} results, {len(failures)} failures")
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            try:
                aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
                params_dict = zip(self.model.state_dict().keys(),
                                  [torch.from_numpy(np.copy(p)) for p in aggregated_ndarrays])
                state_dict = OrderedDict({k: v for k, v in params_dict})
                self.model.load_state_dict(state_dict, strict=True)

                actual_round = self.start_round + server_round - 1
                model_path = os.path.join(cfg.work_dir, f"aggregated_model_round_{actual_round}.pth")
                torch.save(self.model.state_dict(), model_path)
                print(f"Saved aggregated model to {model_path}")

                # Collect and print client statistics in the first round
                if server_round == 1:
                    for client, fit_res in results:
                        client_id = client.cid
                        if 'tb_ratio' in fit_res.metrics and 'image_width' in fit_res.metrics and 'image_height' in fit_res.metrics:
                            self.client_statistics[client_id] = {
                                'tb_ratio': fit_res.metrics['tb_ratio'],
                                'image_size': (fit_res.metrics['image_width'], fit_res.metrics['image_height'])
                            }
                            print(f"Client {client_id}: TB image ratio {fit_res.metrics['tb_ratio']:.4f}, "
                                  f"image size {fit_res.metrics['image_width']}x{fit_res.metrics['image_height']}")

                # self.evaluate_aggregated_model()
            except Exception as e:
                print(f"Aggregation or evaluation failed: {e}")
        else:
            print(
                f"Round {server_round} has no aggregated parameters, results: {len(results)}, failures: {len(failures)}")

        gc.collect()
        return aggregated_parameters, metrics

    def evaluate_aggregated_model(self):
        print("Starting evaluation of aggregated model...")
        self.model.eval()
        try:
            model = MMDataParallel(self.model, device_ids=[0])
            outputs = single_gpu_test(model, self.val_dataloader)

            preds = []
            gts = []
            for i, output in enumerate(outputs):
                pred = torch.argmax(torch.tensor(output)).item()
                gt = self.val_dataset[i]['ann_info']['labels'][0]
                preds.append(pred)
                gts.append(gt)

            accuracy = sum(1 for p, g in zip(preds, gts) if p == g) / len(gts)
            print(f"Aggregated model evaluation completed, accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"Evaluation of aggregated model failed: {e}")
            raise


# Server factory function
def server_fn(context: Context) -> ServerAppComponents:
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=math.ceil(num_clients / 2),
        min_evaluate_clients=math.ceil(num_clients / 2),
        min_available_clients=math.ceil(num_clients / 2),
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    start_round = strategy.start_round if hasattr(strategy, 'start_round') else 1
    num_rounds = cfg.num_rounds
    if start_round > 1:
        num_rounds -= (start_round - 1)
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Client resources
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