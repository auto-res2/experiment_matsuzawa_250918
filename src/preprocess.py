import os
import torch
import requests
import hashlib
from tqdm import tqdm
import logging
import numpy as np
from torch_geometric.datasets import Reddit, OGBNProducts
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops, remove_multi_edges

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_with_progress(url, filepath, md5_hash=None):
    """Downloads a file with a progress bar and optional MD5 check."""
    if os.path.exists(filepath):
        logging.info(f"File {filepath} already exists. Skipping download.")
        if md5_hash:
            logging.info("Verifying MD5 checksum...")
            if verify_md5(filepath, md5_hash):
                logging.info("MD5 checksum verified.")
                return
            else:
                logging.warning("MD5 checksum mismatch. Re-downloading.")
                os.remove(filepath)
        else:
            return

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(filepath, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            raise RuntimeError("ERROR, something went wrong during download")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download {url}. Error: {e}")
        raise RuntimeError(f"Download failed for {url}")

    if md5_hash:
        if not verify_md5(filepath, md5_hash):
            raise RuntimeError(f"MD5 checksum mismatch for {filepath}. Aborting.")
        else:
            logging.info("MD5 checksum verified.")

def verify_md5(filepath, expected_md5):
    """Verifies the MD5 checksum of a file."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest() == expected_md5

def process_graph(data):
    """Applies common preprocessing: make undirected, remove multi-edges, add self-loops."""
    logging.info("Processing graph: converting to undirected, adding self-loops...")
    # Ensure undirected
    edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    # Remove duplicates and add self-loops
    edge_index, _ = remove_multi_edges(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)
    data.edge_index = edge_index

    # Feature normalization
    if data.x is not None:
        logging.info("Normalizing node features (L2 row-wise)...")
        data.x = F.normalize(data.x, p=2, dim=1)
    
    return data

import torch.nn.functional as F

def run_preprocessing(config):
    """Main function to download and preprocess all datasets."""
    raw_dir = os.path.join(config['paths']['data_dir'], 'raw')
    processed_dir = os.path.join(config['paths']['data_dir'], 'processed')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    dataset_name = config['dataset']['name'].lower()
    processed_path = os.path.join(processed_dir, f"{dataset_name}.pt")

    if os.path.exists(processed_path):
        logging.info(f"Processed dataset found at {processed_path}. Skipping preprocessing.")
        return processed_path

    logging.info(f"Processing dataset: {dataset_name}")
    if dataset_name == 'reddit':
        dataset = Reddit(root=raw_dir)
        data = process_graph(dataset[0])
    elif dataset_name == 'ogbn-products':
        dataset = OGBNProducts(root=raw_dir)
        data = process_graph(dataset[0])
        # OGB has its own splits, convert to masks
        split_idx = dataset.get_idx_split()
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[split_idx['train']] = True
        val_mask[split_idx['valid']] = True
        test_mask[split_idx['test']] = True
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data.y = data.y.squeeze()

    elif dataset_name == 'friendster':
        d_config = config['dataset']
        url = d_config['url']
        md5 = d_config['md5']
        raw_path = os.path.join(raw_dir, 'friendster.txt.gz')
        download_with_progress(url, raw_path, md5)
        
        logging.info("Loading Friendster edge list... This may take a while.")
        import pandas as pd
        edge_list = pd.read_csv(raw_path, sep='\t', comment='#', header=None, compression='gzip').values
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        num_nodes = edge_index.max().item() + 1

        logging.warning("Friendster dataset has no node features or labels. Generating synthetic ones as per experimental design.")
        # Generate random features and labels
        num_features = 128
        num_classes = 10
        x = torch.randn((num_nodes, num_features), dtype=torch.float32)
        y = torch.randint(0, num_classes, (num_nodes,))

        data = Data(x=x, edge_index=edge_index, y=y)
        data.num_nodes = num_nodes
        data = process_graph(data)

        # Create random splits
        torch.manual_seed(42)
        perm = torch.randperm(num_nodes)
        train_end = int(0.6 * num_nodes)
        val_end = int(0.8 * num_nodes)
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[perm[:train_end]] = True
        data.val_mask[perm[train_end:val_end]] = True
        data.test_mask[perm[val_end:]] = True

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Add num_classes to data object if not present
    if not hasattr(data, 'num_classes'):
        if data.y.max() is not None:
            data.num_classes = data.y.max().item() + 1
        else:
             raise ValueError("Could not determine number of classes.")

    logging.info(f"Saving processed data to {processed_path}")
    torch.save(data, processed_path)
    return processed_path
