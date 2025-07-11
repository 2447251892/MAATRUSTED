# 文件名: MAA_base.py

from abc import ABC, abstractmethod
import random, torch, numpy as np
from utils.util import setup_device
import os

class MAABase(ABC):
    def __init__(self, N_pairs, batch_size, num_epochs,
                 generator_names, discriminators_names,
                 ckpt_dir, output_dir,
                 initial_learning_rate = 2e-4,
                 precise = torch.float32,
                 do_distill_epochs: int = 1,
                 cross_finetune_epochs: int = 5,
                 device = None,
                 seed=None,
                 ckpt_path="auto",):
        super().__init__()

        self.N = N_pairs
        self.initial_learning_rate = initial_learning_rate
        self.generator_names = generator_names
        self.discriminators_names = discriminators_names
        self.ckpt_dir = ckpt_dir
        self.ckpt_path = ckpt_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.seed = seed
        self.do_distill_epochs = do_distill_epochs
        self.cross_finetune_epochs = cross_finetune_epochs
        self.device = device
        self.precise = precise

        self.set_seed(self.seed)
        self.device = setup_device(device)
        print("运行设备:", self.device)

        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"输出目录已创建: {self.output_dir}")

        if self.ckpt_dir and not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
            print(f"检查点目录已创建: {self.ckpt_dir}")

    def set_seed(self, seed):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    @abstractmethod
    def process_data(self):
        pass

    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def init_dataloader(self):
        pass

    @abstractmethod
    def init_hyperparameters(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def save_models(self):
        pass

    @abstractmethod
    def distill(self):
        pass

    @abstractmethod
    def visualize_and_evaluate(self):
        pass

    @abstractmethod
    def init_history(self):
        pass