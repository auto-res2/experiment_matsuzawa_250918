import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, IterableDataset, ChainDataset, Dataset
from datasets import load_dataset
import time
import threading
import cupy
import logging
import timm
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BackgroundLoadSimulator(threading.Thread):
    def __init__(self, load_level, matrix_size=8192):
        super().__init__()
        if not (0 <= load_level <= 1):
            raise ValueError("Load level must be between 0 and 1.")
        self.load_level = load_level
        self.matrix_size = matrix_size
        self._running = threading.Event()
        self._running.set()
        self.daemon = True # Allows main program to exit even if this thread is running

    def run(self):
        logging.info(f"Background GPU load thread started with load level {self.load_level:.2f}")
        try:
            a = cupy.random.random((self.matrix_size, self.matrix_size), dtype=cupy.float32)
            b = cupy.random.random((self.matrix_size, self.matrix_size), dtype=cupy.float32)
            
            # Calibrate work duration
            start_time = time.perf_counter()
            cupy.dot(a, b)
            cupy.cuda.runtime.deviceSynchronize()
            work_duration = time.perf_counter() - start_time
            
            if self.load_level > 0:
                sleep_duration = work_duration * (1 - self.load_level) / self.load_level
            else:
                sleep_duration = float('inf')

            while self._running.is_set():
                if self.load_level > 0:
                    cupy.dot(a, b)
                    cupy.cuda.runtime.deviceSynchronize()
                    if sleep_duration > 0:
                        time.sleep(sleep_duration)
                else:
                    time.sleep(1) # Sleep if no load to avoid busy-waiting
        except Exception as e:
            logging.error(f"Error in BackgroundLoadSimulator: {e}. Cupy might not be installed correctly.")
        logging.info("Background GPU load thread stopped.")

    def stop(self):
        self._running.clear()

# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------
class RealTimeStreamer(IterableDataset):
    def __init__(self, dataset, transform, target_fps):
        self.dataset = dataset
        self.transform = transform
        if target_fps <= 0:
            raise ValueError("Target FPS must be positive.")
        self.interval = 1.0 / target_fps

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Handle multiprocessing correctly
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            iter_start = worker_id
            iter_end = len(self.dataset)
            dataset_iter = iter(range(iter_start, iter_end, num_workers))
        else:
            dataset_iter = iter(range(len(self.dataset)))
        
        for index in dataset_iter:
            start_time = time.perf_counter()
            
            item = self.dataset[index]
            image = item['image']
            label = item.get('label', -1) # Use -1 for datasets without labels
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            transformed_image = self.transform(image)
            yield transformed_image, torch.tensor(label, dtype=torch.long)
            
            elapsed_time = time.perf_counter() - start_time
            sleep_time = self.interval - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def get_transform(backbone_name):
    model = timm.create_model(backbone_name, pretrained=True)
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    return transform

class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, num_classes=1000, img_size=(224, 224)):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(3, *self.img_size)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return {'image': image, 'label': label}

# -----------------------------------------------------------------------------
# Unified public API
# -----------------------------------------------------------------------------

def _load_imagenet_c(subset_size):
    """Load the ImageNet-C parquet conversion from ang9867/ImageNet-C."""
    try:
        ds = load_dataset('ang9867/ImageNet-C', split=f"train[:{subset_size}]")
        return ds
    except Exception as e:
        raise RuntimeError("Failed to load ImageNet-C from ang9867/ImageNet-C.") from e


def _load_domainnet(subset_size):
    try:
        return load_dataset('wltjr1007/DomainNet', split=f"train[:{subset_size}]")
    except Exception as e:
        raise RuntimeError("Failed to load DomainNet dataset from wltjr1007/DomainNet.") from e


def get_data_stream(config, backbone_name):
    """Return a PyTorch DataLoader that yields (image, label) batches in real-time."""
    name = config['name']
    batch_size = config['batch_size']
    eta = config.get('eta', 1.0)
    base_fps = 30.0
    target_fps = base_fps * eta
    num_workers = config.get('num_workers', 4)
    subset_size = config.get('subset_size', '100%')

    transform = get_transform(backbone_name)

    # ------------------------------------------------------------------
    # Synthetic – trivial path used primarily for unit tests
    # ------------------------------------------------------------------
    if name.lower() == 'synthetic':
        dataset = SyntheticDataset()
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    # ------------------------------------------------------------------
    # Real datasets
    # ------------------------------------------------------------------
    logging.info(f"Loading dataset: {name}")
    if name == 'ImageNet-C':
        dataset = _load_imagenet_c(subset_size)

    elif name == 'DomainNet':
        dataset = _load_domainnet(subset_size)

    elif name == 'Recurring-TTA':
        # We approximate the 3-cycle stream with available repos.
        inet_s = load_dataset('imagenet_sketch', split=f"train[:{subset_size}]")
        inet_c = _load_imagenet_c(subset_size)
        inet_r = load_dataset('imagenet-r', split=f"train[:{subset_size}]")
        cycle_datasets = [inet_s, inet_c, inet_r] * 20  # 20 cycles as before
        dataset = ChainDataset(cycle_datasets)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # ------------------------------------------------------------------
    # Wrap inside real-time streamer & DataLoader
    # ------------------------------------------------------------------
    streamer_dataset = RealTimeStreamer(dataset, transform, target_fps)

    return DataLoader(streamer_dataset, batch_size=batch_size, num_workers=num_workers)
