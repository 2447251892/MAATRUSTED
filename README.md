# MAATRUSTED 🚀

**基于多智能体对抗网络 (Multi-GAN) 的量化交易预测与边缘端部署系统**

![MAATRUSTED Logo/Banner](docs/images/banner.png)  
*(建议：这里放一张项目的架构图或 Logo)*

## 📖 项目简介

**MAATRUSTED** 是一个集成了数据获取、深度学习建模、策略回测、可视化分析以及嵌入式 NPU 部署的全栈量化交易系统。

本系统利用先进的时序模型（Transformer, BiLSTM, GRU 等）结合 **Multi-GAN（多生成器对抗网络）** 架构，旨在从复杂的股票市场数据中提取有效特征，生成买卖信号，并支持将最优策略模型无缝部署到 **瑞芯微 RK3568** 等边缘计算设备上进行离线推理。

## ✨ 核心特性

*   **📈 全自动化数据流**：集成 Tushare Pro 接口，自动完成日线/指数/基金数据的获取、清洗、技术指标计算（MACD, KDJ, MA等）及归一化处理。
*   **🧠 多智能体对抗训练**：
    *   引入 **RoPE (旋转位置编码)** 增强模型时序捕捉能力。
    *   支持无监督预训练 (CAE/t3VAE) 提取潜在特征。
    *   多生成器 (Generator) 与多判别器 (Discriminator) 博弈，提升预测鲁棒性。
*   **🛡️ 鲁棒的策略回测**：内置涨跌停限制、一字板过滤、止损策略等实战规则，自动评选最佳交易策略。
*   **📊 交互式决策大屏**：基于 Dash 构建的 Web 可视化界面，提供 K 线复盘、信号标记及一键更新功能。
*   **⚡ 端到端 NPU 部署**：
    *   独创的 JSON 参数解耦方案，解决跨平台数据处理痛点。
    *   支持一键批量导出 ONNX 模型。
    *   完美适配 RKNN 工具链，实现 RK3568 NPU 硬件加速推理。

## 🛠️ 环境依赖 (Windows)

本项目在 **Windows 10/11** 环境下开发，推荐使用 Conda 管理环境。

### 必要版本配置
*   **Python**: 3.11
*   **CUDA**: 11.8 (配合 cuDNN 8.7)
*   **PyTorch**: 2.1.2 + cu118

### ⚡ 快速安装指南

1.  **创建并激活虚拟环境**
    ```bash
    conda create -n maatrusted python=3.11
    conda activate maatrusted
    ```

2.  **安装 PyTorch (使用清华镜像源)**
    ```bash
    pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
    # 或者使用您提供的简化命令：
    pip install torch==2.1.2+cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

3.  **安装其他依赖**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn tushare dash plotly joblib onnx onnxruntime tqdm talib
    ```
    *(注：TA-Lib 在 Windows 下可能需要下载 `.whl` 文件安装)*

---

## 🚀 快速开始

### 1. 配置目标股票
编辑根目录下的 `标的参考.txt`，按板块填入您关注的股票名称。

### 2. 获取数据
运行数据获取脚本，自动下载并预处理数据。
```bash
python get_stock_data.py
```
*(注意：请在脚本中替换您的 Tushare Token)*

### 3. 启动训练
启动多智能体对抗训练流程。系统会自动进行预训练和主训练。
```bash
python experiment_runner.py --mode train --num_epochs 100
```
![训练过程截图](docs/images/training_process.png)
*(建议：放一张训练时的 Loss 曲线或控制台输出截图)*

### 4. 策略回测与筛选
训练完成后，运行回测脚本。系统会根据交易规则筛选出表现最好的模型（G1/G2/G3...）。
```bash
python filter_trading_signals.py
```
此步骤会生成 `best_metrics.csv`，记录最佳策略的收益率和胜率。

### 5. 可视化分析
启动 Dash 仪表盘，在浏览器中查看 K 线图和模型信号。
```bash
python dash_kline_visualizer.py
```
访问：`http://127.0.0.1:8050`

![可视化大屏截图](docs/images/dash_ui.png)
*(建议：放一张浏览器中 Dash K线图界面的截图，最好能看到买卖点箭头)*

---

## 💾 边缘端部署 (RK3568)

本项目支持将最佳策略模型导出并部署到 RK3568 开发板。

### 步骤 1：Windows 端一键导出
运行导出脚本，通过交互式菜单选择要导出的股票（支持批量）。
```bash
python deploy_export.py
```
**产出物**：`deploy_output/` 目录，包含 `model_deploy.onnx` (模型) 和 `scaler_params.json` (参数)。

### 步骤 2：模型转换 (Linux/虚拟机)
使用 `rknn-toolkit2` 将 ONNX 转换为 RKNN 模型（FP16精度）。
```bash
# 请参考 deploy_convert_batch.py (需自行配置RKNN环境)
python deploy_convert_batch.py
```

### 步骤 3：板端推理
将模型和 JSON 参数上传至 RK3568，运行推理脚本即可实现 NPU 加速预测。

![部署流程图](docs/images/deployment_flow.png)
*(建议：放一张从 Windows 到 RK3568 的文件流转示意图)*

---

## 📂 目录结构说明

```text
MAATRUSTED/
├── csv_data/                # 存放原始及预处理后的股票数据
├── output/                  # 训练日志、权重检查点 (Checkpoints)
├── output_filtered_signals/ # 回测结果、最佳策略指标
├── models/                  # 模型定义 (GRU, LSTM, Transformer, RoPE等)
├── utils/                   # 工具类 (Logger, Trainer, Visualization)
├── deploy_output/           # 导出的部署文件
├── get_stock_data.py        # 数据获取
├── experiment_runner.py     # 训练入口
├── filter_trading_signals.py# 回测入口
├── dash_kline_visualizer.py # 可视化大屏
├── deploy_export.py         # 部署导出工具
└── README.md                # 项目说明
```

## ⚠️ 免责声明

本项目仅供计算机科学研究、深度学习算法验证及量化交易技术交流使用。**项目中的任何预测结果、信号或策略均不构成投资建议。** 金融市场风险巨大，实盘交易请自行承担风险。

---

## 🤝 贡献与支持

欢迎提交 Issue 或 Pull Request！
如果是 Tushare 数据问题，请查阅 [Tushare 官方文档](https://tushare.pro/)。
