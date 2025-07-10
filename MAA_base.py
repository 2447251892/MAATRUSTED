# 文件名: MAA_base.py

from abc import ABC, abstractmethod
import random, torch, numpy as np
from utils.util import setup_device
import os

class MAABase(ABC):
    """
    MAA (多智能体对抗) 框架的抽象基类,
    定义了核心方法的接口。
    所有子类都必须实现以下方法。
    """

    def __init__(self, N_pairs, batch_size, num_epochs,
                 generator_names, discriminators_names,
                 ckpt_dir, output_dir,
                 initial_learning_rate = 2e-4,
                 train_split = 0.8,
                 precise = torch.float32,
                 do_distill_epochs: int = 1,
                 cross_finetune_epochs: int = 5,
                 device = None,
                 seed=None,
                 ckpt_path="auto",):
        """
        初始化必要的超参数。

        :param N_pairs: 生成器或判别器的数量
        :param batch_size: 小批量大小 (mini-batch size)
        :param num_epochs: 计划的训练轮数
        :param initial_learning_rate: 初始学习率
        :param generator_names: 生成器名称列表，建议是包含不同特征生成器的可迭代对象
        :param discriminators_names: 判别器名称列表，建议是可迭代对象，可以是相同的判别器
        :param ckpt_dir: 用于保存每个模型检查点的目录路径
        :param output_dir: 用于保存可视化结果、日志等的输出目录路径
        :param train_split: 训练集划分比例
        :param precise: 计算精度，如 torch.float32
        :param do_distill_epochs: 知识蒸馏的轮数
        :param cross_finetune_epochs: 交叉微调的轮数
        :param device: 计算设备 (cpu, cuda)
        :param seed: 随机种子
        :param ckpt_path: 模型检查点的具体路径
        """

        self.N = N_pairs
        self.initial_learning_rate = initial_learning_rate
        self.generator_names = generator_names
        self.discriminators_names = discriminators_names
        self.ckpt_dir = ckpt_dir
        self.ckpt_path = ckpt_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.train_split = train_split
        self.seed = seed
        self.do_distill_epochs = do_distill_epochs
        self.cross_finetune_epochs = cross_finetune_epochs
        self.device = device
        self.precise = precise

        self.set_seed(self.seed)  # 初始化随机种子
        self.device = setup_device(device)
        print("运行设备:", self.device)

        # ==================== MODIFICATION START ====================
        # 只有在提供了非空路径时才创建目录，增加代码健壮性
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"输出目录已创建: {self.output_dir}")

        if self.ckpt_dir and not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
            print(f"检查点目录已创建: {self.ckpt_dir}")
        # ===================== MODIFICATION END =====================

    def set_seed(self, seed):
        """设置随机种子以保证结果的可复现性。"""
        # 修复了当 seed 为 None 时可能引发的 TypeError
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    @abstractmethod
    def process_data(self):
        """数据预处理，包括读取、清洗、切分等。"""
        pass

    @abstractmethod
    def init_model(self):
        """模型结构初始化。"""
        pass

    @abstractmethod
    def init_dataloader(self):
        """初始化用于训练和评估的数据加载器。"""
        pass

    @abstractmethod
    def init_hyperparameters(self):
        """初始化训练所需的超参数。"""
        pass

    @abstractmethod
    def train(self):
        """执行训练流程。"""
        pass

    @abstractmethod
    def save_models(self):
        """保存模型。"""
        pass

    @abstractmethod
    def distill(self):
        """执行知识蒸馏流程。"""
        pass

    @abstractmethod
    def visualize_and_evaluate(self):
        """评估模型性能并可视化结果。"""
        pass

    @abstractmethod
    def init_history(self):
        """初始化训练过程中的指标记录结构。"""
        pass