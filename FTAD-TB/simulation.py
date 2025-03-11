import argparse
import os
import sys
import subprocess
import threading
import time


def run_server(config_path, num_rounds, min_clients, save_dir, resume_from=None):
    """Run the Flower server."""
    cmd = [
        "python", "server.py",
        "--config", config_path,
        "--rounds", str(num_rounds),
        "--min-clients", str(min_clients),
        "--save-dir", save_dir
    ]

    if resume_from:
        cmd.extend(["--resume-from", resume_from])

    print(f"Starting server with command: {' '.join(cmd)}")
    server_process = subprocess.Popen(cmd)
    return server_process


def run_client(client_id, config_path, server_address):
    """Run a Flower client."""
    # Modify the config to have client-specific data if needed
    client_config = config_path  # In a real scenario, you might want to create client-specific configs

    cmd = [
        "python", "client.py",
        "--config", client_config,
        "--client-id", str(client_id),
        "--server-address", server_address
    ]

    print(f"Starting client {client_id} with command: {' '.join(cmd)}")
    client_process = subprocess.Popen(cmd)
    return client_process


def main():
    parser = argparse.ArgumentParser(description="Run Flower Federated Learning Simulation")
    parser.add_argument("--config", required=True, help="Path to MMDetection config file")
    parser.add_argument("--num-clients", type=int, default=2, help="Number of clients to simulate")
    parser.add_argument("--num-rounds", type=int, default=3, help="Number of federated learning rounds")
    parser.add_argument("--save-dir", type=str, default="./server_models", help="Directory to save server models")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from a server model checkpoint")
    parser.add_argument("--server-address", type=str, default="localhost:8080", help="Server address (host:port)")
    args = parser.parse_args()

    # Start the server
    server_process = run_server(
        config_path=args.config,
        num_rounds=args.num_rounds,
        min_clients=args.num_clients,
        save_dir=args.save_dir,
        resume_from=args.resume_from
    )

    # Give the server some time to start
    time.sleep(3)

    # Start the clients
    client_processes = []
    for client_id in range(args.num_clients):
        client_process = run_client(
            client_id=client_id,
            config_path=args.config,
            server_address=args.server_address
        )
        client_processes.append(client_process)

    # Wait for the server to finish
    server_process.wait()

    # Terminate any remaining client processes
    for client_process in client_processes:
        if client_process.poll() is None:  # If process is still running
            client_process.terminate()

    print("Federated learning simulation completed.")


if __name__ == "__main__":
    main()
