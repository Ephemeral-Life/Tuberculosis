import os
import argparse
import numpy as np
import torch
from collections import OrderedDict
import time
from typing import Dict, List, Tuple, Optional, Union
import logging

# Import Flower
import flwr as fl
from flwr.common import Parameters, FitRes, EvaluateRes, NDArrays, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

# Import your existing training and evaluation code
from mmcv import Config
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from mmdet.apis import single_gpu_test
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.core import eval_map

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Server")


class MMDetectionServer(FedAvg):
    """Flower server strategy that extends FedAvg with MMDetection validation."""

    def __init__(
            self,
            config_path: str,
            server_model_save_path: str = './server_models',
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn=None,
            *args,
            **kwargs
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            *args,
            **kwargs
        )
        self.config_path = config_path
        self.server_model_save_path = server_model_save_path
        os.makedirs(server_model_save_path, exist_ok=True)

        # Load config
        self.cfg = Config.fromfile(config_path)

        # Build model
        self.model = build_detector(
            self.cfg.model,
            train_cfg=self.cfg.get('train_cfg'),
            test_cfg=self.cfg.get('test_cfg'))
        self.model.init_weights()

        # Build validation dataset
        self.val_dataset = build_dataset(self.cfg.data.val)
        self.model.CLASSES = self.val_dataset.CLASSES

        # Initialize parameters
        self.parameters = self.get_model_parameters()

        logger.info(f"Server initialized with config: {config_path}")
        logger.info(f"Model type: {self.cfg.model.type}")
        logger.info(f"Validation dataset size: {len(self.val_dataset)}")

    def get_model_parameters(self) -> List[np.ndarray]:
        """Get model parameters as a list of NumPy arrays."""
        state_dict = self.model.state_dict()
        return [val.cpu().numpy() for _, val in state_dict.items()]

    def set_model_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from a list of NumPy arrays."""
        state_dict = OrderedDict()
        params_dict = zip(self.model.state_dict().keys(), parameters)
        for k, v in params_dict:
            state_dict[k] = torch.Tensor(v)
        self.model.load_state_dict(state_dict, strict=True)

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure the next round of training."""
        # Save current global model
        self.parameters = parameters
        self.save_model(server_round)

        # Get clients for this round
        config = {}
        if server_round > 1:
            # Tell clients to load from the saved global model
            config["server_model_path"] = os.path.join(
                self.server_model_save_path, f"round-{server_round - 1}-model.pth"
            )

        client_instructions = super().configure_fit(server_round, parameters, client_manager)

        # Add the same config to all clients
        for _, fit_ins in client_instructions:
            fit_ins.config.update(config)

        return client_instructions

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights and store checkpoint."""
        # Call aggregate_fit from parent class (FedAvg)
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Update server model with aggregated parameters
            self.parameters = aggregated_parameters
            self.set_model_parameters(aggregated_parameters)

            # Save the model
            self.save_model(server_round)

            # Evaluate the model on the validation set
            eval_results = self.evaluate_model()
            logger.info(f"Server-side evaluation after round {server_round}: {eval_results}")

            # Add evaluation metrics to the metrics dictionary
            metrics.update(eval_results)

        return aggregated_parameters, metrics

    def save_model(self, server_round: int) -> None:
        """Save the current model."""
        save_path = os.path.join(self.server_model_save_path, f"round-{server_round}-model.pth")
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Saved model to {save_path}")

    def load_model(self, path: str) -> None:
        """Load model from a checkpoint."""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location="cpu")
            self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded model from {path}")
            self.parameters = self.get_model_parameters()
        else:
            logger.warning(f"Model checkpoint not found at {path}")

    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate the current model on the validation dataset."""
        # Create dataloader for validation dataset
        from torch.utils.data import DataLoader
        from mmdet.datasets.pipelines import Compose

        # Preparing the data loader
        data_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            sampler=None,
            num_workers=1,
            shuffle=False
        )

        # Prepare model for testing
        self.model = MMDataParallel(self.model, device_ids=[0])
        self.model.eval()

        # Run evaluation
        results = []
        logger.info("Starting server-side evaluation on validation dataset")

        with torch.no_grad():
            for data in data_loader:
                result = self.model(return_loss=False, rescale=True, **data)
                results.append(result)

        # Calculate mAP
        try:
            # Get annotations from dataset
            annotations = [self.val_dataset.get_ann_info(i) for i in range(len(self.val_dataset))]

            # Calculate mAP
            eval_results = eval_map(
                results,
                annotations,
                scale_ranges=None,
                iou_thr=0.5,
                dataset=self.val_dataset.CLASSES,
                logger='silent'
            )

            # Extract and format results
            mean_ap = eval_results['AP'].mean().item()
            class_aps = {
                f"AP_{class_name}": ap.item()
                for class_name, ap in zip(self.val_dataset.CLASSES, eval_results['AP'])
            }

            metrics = {
                "mean_ap": mean_ap,
                **class_aps
            }

            logger.info(f"Evaluation metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {"evaluation_error": 1.0}


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Flower Server with MMDetection')
    parser.add_argument('--config', required=True, help='MMDetection config file path')
    parser.add_argument('--server-address', type=str, default="0.0.0.0:8080", help='Server address (IP:PORT)')
    parser.add_argument('--rounds', type=int, default=3, help='Number of rounds')
    parser.add_argument('--min-clients', type=int, default=2, help='Minimum number of clients')
    parser.add_argument('--save-dir', type=str, default='./server_models', help='Directory to save models')
    parser.add_argument('--resume-from', type=str, default=None, help='Path to model checkpoint to resume from')
    args = parser.parse_args()

    # Start Flower server
    strategy = MMDetectionServer(
        config_path=args.config,
        server_model_save_path=args.save_dir,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
    )

    # If resume from checkpoint is specified
    if args.resume_from:
        strategy.load_model(args.resume_from)
        logger.info(f"Resumed from {args.resume_from}")

    # Start server
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
