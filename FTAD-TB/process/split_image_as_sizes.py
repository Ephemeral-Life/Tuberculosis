import os
import random
import shutil
from PIL import Image


def resize_image(image_path, output_path, size):
    """
    Resize an image and save it to the specified path.

    Args:
        image_path (str): Path to the original image
        output_path (str): Path to save the resized image
        size (tuple): Target size (width, height)
    """
    with Image.open(image_path) as img:
        resized_img = img.resize(size, Image.Resampling.LANCZOS)
        resized_img.save(output_path)


def split_dataset(root_dir, output_dir, num_clients=3):
    """
    Split the dataset into multiple clients with random distribution and resize images.

    Args:
        root_dir (str): Root directory of the original dataset
        output_dir (str): Directory to save client datasets
        num_clients (int): Number of clients to simulate
    """
    # Define categories and client-specific resize sizes
    categories = ['health', 'sick', 'tb']
    resize_sizes = [(512, 512), (256, 256), (128, 128)]

    # Define tb category distribution ratios (sum to 1)
    tb_ratios = [0.3, 0.5, 0.2]  # Uneven distribution for tb

    for category in categories:
        category_path = os.path.join(root_dir, category)
        if not os.path.exists(category_path):
            print(f"Warning: Directory {category_path} does not exist")
            continue

        # Get list of images
        images = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        random.shuffle(images)  # Randomize image order

        if category == 'tb':
            # Distribute tb images unevenly based on ratios
            total_images = len(images)
            client_images = []
            start_idx = 0
            for ratio in tb_ratios:
                num_images = int(total_images * ratio)
                client_images.append(images[start_idx:start_idx + num_images])
                start_idx += num_images
            # Assign remaining images to the last client
            if start_idx < total_images:
                client_images[-1].extend(images[start_idx:])
        else:
            # Evenly distribute health and sick images
            client_images = [images[i::num_clients] for i in range(num_clients)]

        # Create directories and resize images for each client
        for client_id in range(num_clients):
            client_category_path = os.path.join(output_dir, str(client_id), category)
            os.makedirs(client_category_path, exist_ok=True)
            for image in client_images[client_id]:
                src_path = os.path.join(category_path, image)
                dst_path = os.path.join(client_category_path, image)
                resize_image(src_path, dst_path, resize_sizes[client_id])
                # shutil.copyfile(src_path, dst_path)


def main():
    # Set paths
    root_dir = '../../data/TBX11K/imgs'  # Adjust this path as needed
    output_dir = '../../data/fed/clients'

    # Clear output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Split and process the dataset
    split_dataset(root_dir, output_dir)


if __name__ == "__main__":
    main()
