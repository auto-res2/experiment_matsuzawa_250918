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
        return image, label

def get_data_stream(config, backbone_name):
    name = config['name']
    batch_size = config['batch_size']
    eta = config.get('eta', 1.0)
    base_fps = 30.0
    target_fps = base_fps * eta
    num_workers = config.get('num_workers', 4)

    transform = get_transform(backbone_name)

    if name == 'synthetic':
        dataset = SyntheticDataset()
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    logging.info(f"Loading dataset: {name}")
    try:
        if name == 'ImageNet-C':
            # ImageNet-C needs special handling as it has many subsets
            corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                           'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 
                           'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
            severities = [1, 2, 3, 4, 5]
            all_datasets = []
            for corruption in corruptions:
                for severity in severities:
                    # Load a small subset for smoke tests if specified
                    split = f"{corruption}_{severity}[:{config.get('subset_size', '100%')}]"
                    ds = load_dataset('hendrycks/imagenet-c', split=split, trust_remote_code=True)
                    all_datasets.append(ds)
            dataset = ChainDataset(all_datasets)
        elif name == 'DomainNet':
            dataset = load_dataset('domainnet', split=f"train[:{config.get('subset_size', '100%')}]")
        elif name == 'Recurring-TTA':
            inet_s = load_dataset('imagenet_sketch', split=f"train[:{config.get('subset_size_per_cycle', '100%')}]")
            inet_c_fog = load_dataset('hendrycks/imagenet-c', name='fog', split=f"5[:{config.get('subset_size_per_cycle', '100%')}]", trust_remote_code=True)
            inet_r = load_dataset('imagenet-r', split=f"train[:{config.get('subset_size_per_cycle', '100%')}]")
            cycle_datasets = [inet_s, inet_c_fog, inet_r] * 20 # 20 cycles
            dataset = ChainDataset(cycle_datasets)
        else:
            raise ValueError(f"Unknown dataset: {name}")
    except Exception as e:
        logging.error(f"Failed to load dataset '{name}'. Please ensure you have the necessary permissions and the dataset exists. Error: {e}")
        raise RuntimeError(f"Dataset loading failed for '{name}'. Aborting experiment.") from e

    # The IterableDataset wrapper handles the real-time simulation
    streamer_dataset = RealTimeStreamer(dataset, transform, target_fps)

    return DataLoader(streamer_dataset, batch_size=batch_size, num_workers=num_workers)
