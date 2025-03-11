import os
import argparse
import numpy as np
import torch
from collections import OrderedDict
import logging

# Import Flower
import flwr as fl
from flwr.common import NDArrays, Scalar

# Import your existing training code functionality
from mmcv import Config
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from mmdet.apis import train_detector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Client")


class MMDetectionClient(fl.client.NumPyClient):
    def __init__(self, config_path, client_id, server_address="localhost:8080"):
        self.client_id = client_id
        self.config_path = config_path
        self.server_address = server_address

        # Load config
        self.cfg = Config.fromfile(config_path)

        # Modify work_dir to be client-specific
        client_work_dir = os.path.join('./work_dirs', f'client_{client_id}')
        self.cfg.work_dir = client_work_dir
        os.makedirs(client_work_dir, exist_ok=True)

        # Build model
        self.model = build_detector(
            self.cfg.model,
            train_cfg=self.cfg.get('train_cfg'),
            test_cfg=self.cfg.get('test_cfg'))
        self.model.init_weights()

        # Build dataset - we can partition data here if needed for different clients
        # For this example, all clients use the same dataset
        self.datasets = [build_dataset(self.cfg.data.train)]
        if len(self.cfg.workflow) == 2:
            val_dataset = self.cfg.data.val.copy()
            val_dataset.pipeline = self.cfg.data.train.pipeline
            self.datasets.append(build_dataset(val_dataset))

        # Set model classes
        self.model.CLASSES = self.datasets[0].CLASSES

        logger.info(f"Client {client_id} initialized with config: {config_path}")
        logger.info(f"Dataset size: {len(self.datasets[0])}")

    def get_parameters(self, config):
        """Return current model parameters as a list of NumPy arrays."""
        state_dict = self.model.state_dict()
        return [val.cpu().numpy() for _, val in state_dict.items()]

    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays."""
        state_dict = OrderedDict()
        params_dict = zip(self.model.state_dict().keys(), parameters)
        for k, v in params_dict:
            state_dict[k] = torch.Tensor(v)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train the model on the local dataset."""
        logger.info(f"Client {self.client_id} starting training for round {config.get('server_round', 0)}")

        # Check if we should load from server model
        if "server_model_path" in config and os.path.exists(config["server_model_path"]):
            logger.info(f"Loading model from server checkpoint: {config['server_model_path']}")
            checkpoint = torch.load(config["server_model_path"], map_location="cpu")
            self.model.load_state_dict(checkpoint)
        else:
            # Set model parameters from server
            self.set_parameters(parameters)

        # Adjust training params for this round
        self.cfg.runner.max_epochs = config.get("local_epochs", 1)

        # Train model
        timestamp = f"client_{self.client_id}_round_{config.get('server_round', 0)}"
        meta = {
            "client_id": self.client_id,
            "round": config.get("server_round", 0)
        }

        # Distributed is set to False for simplicity in the federated simulation
        train_detector(
            self.model,
            self.datasets,
            self.cfg,
            distributed=False,
            validate=False,
            timestamp=timestamp,
            meta=meta
        )

        # Return updated model parameters and metrics
        updated_parameters = self.get_parameters(config)
        metrics = {
            "client_id": self.client_id,
            "train_size": len(self.datasets[0])
        }

        logger.info(f"Client {self.client_id} completed training for round {config.get('server_round', 0)}")
        return updated_parameters, len(self.datasets[0]), metrics

    def evaluate(self, parameters, config):
        """Evaluate the model on the local test dataset."""
        logger.info(f"Client {self.client_id} evaluating model for round {config.get('server_round', 0)}")

        # Set model parameters
        self.set_parameters(parameters)

        # For simplicity, we're returning a fixed accuracy
        # In a real implementation, you should use the MMDetection evaluation tools
        accuracy = 0.75  # This is a placeholder

        return accuracy, len(self.datasets[0]), {"client_id": self.client_id}


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Flower client with MMDetection')
    parser.add_argument('--config', required=True, help='MMDetection config file path')
    parser.add_argument('--client-id', type=int, required=True, help='Client ID')
    parser.add_argument('--server-address', type=str, default="localhost:8080", help='Server address (host:port)')
    args = parser.parse_args()

    # Create client
    client = MMDetectionClient(
        config_path=args.config,
        client_id=args.client_id,
        server_address=args.server_address
    )

    # Start client
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()
