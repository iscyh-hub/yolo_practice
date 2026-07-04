# 基于 YOLOv5 的皮肤（细胞）检测项目

> 本项目以 YOLOv5 为基线，在 BCCD 血细胞数据集上完成可运行验证。数据集可替换为皮肤病数据集，训练与检测流程通用。

## 项目总结（STAR）

- **Situation（背景）**：项目目标是在血细胞/皮肤细胞场景下构建可运行的目标检测方案。原始 YOLOv5 官方代码基于旧版 PyTorch/NumPy 编写，在 RTX 5060 + CUDA 12.8 + PyTorch 2.8 的新环境下存在兼容性报错，无法直接训练。
- **Task（任务）**：需要让旧版 YOLOv5 在新环境下跑通训练与推理，并进一步迁移到 YOLOv8，加入注意力机制进行改进，为后续对比实验提供 baseline 与改进版代码。
- **Action（行动）**：
  1. 使用 **conda + Python 3.10 + PyTorch 2.8.0+cu128** 搭建虚拟环境 `yolov5-bccd`；
  2. 对旧版 YOLOv5 做最小化兼容性补丁，修改 `train.py`、`detect.py`、`hubconf.py`、`models/experimental.py`、`utils/general.py`、`utils/datasets.py`、`models/yolo.py` 共 7 个文件，解决 `torch.load(weights_only=False)`、`np.int` 弃用、CUDA tensor 转 numpy、leaf Variable in-place 修改等问题；
  3. 在 **BCCD 血细胞数据集**（3 类：Platelets/RBC/WBC，train/valid/test 划分）上完成 YOLOv5s 训练与测试；
  4. 新增 **YOLOv8 迁移与注意力机制**：本地 editable 安装 Ultralytics v8.3.213，新增 `attention.py` 模块（SE / ECA / C2f_CBAM / C2f_SE / C2f_ECA），修改模块注册与任务文件，新增 4 套模型配置（baseline / CBAM / SE / ECA）及训练/对比脚本。
- **Result（结果）**：
  - YOLOv5s 训练 **100 epochs** 跑通，测试指标：`P=0.646`、`R=0.947`、`mAP@.5=0.888`、`mAP@.5:.95=0.603`，权重保存于 `yolov5-master/runs/exp6/weights/`；
  - YOLOv8n 4 个 variant（baseline / CBAM / SE / ECA）的 **1-epoch 快速验证全部通过**，RTX 5060 8GB 无 CUDA OOM；
  - 产出可直接复现的训练命令、环境配置（`environment.yml` / `requirements.txt`）、YOLOv8 对比脚本（`val_all.py`），为后续正式训练与论文实验搭建好完整代码基线。

## 1. 项目结构

```
基于YOLO5细胞检测实战/
├── BCCDDataSet/
│   ├── data.yaml          # 数据集配置（类别、路径）
│   ├── train/
│   ├── valid/
│   └── test/
├── yolov5-master/         # 原始 YOLOv5 代码（仅做兼容性修改）
│   ├── train.py
│   ├── detect.py
│   ├── test.py
│   ├── models/
│   ├── utils/
│   └── runs/              # 训练结果
├── environment.yml        # conda 环境配置
├── requirements.txt       # pip 依赖
└── README.md              # 本文件
```

## 2. 环境配置

已验证环境：

- OS：Windows 11
- GPU：NVIDIA GeForce RTX 5060
- CUDA：12.8
- Python：3.10
- PyTorch：2.8.0+cu128

使用 conda 创建环境：

```bash
conda env create -f environment.yml
conda activate yolov5-bccd
```

或直接用 pip：

```bash
pip install -r requirements.txt
```

> Windows 下训练前建议设置环境变量，避免 OpenMP 冲突：
>
> ```powershell
> $env:KMP_DUPLICATE_LIB_OK="TRUE"
> ```

## 3. YAML 改动说明

### BCCDDataSet/data.yaml

```yaml
train: ../BCCDDataSet/train/images
val: ../BCCDDataSet/valid/images

nc: 3
names: ['Platelets', 'RBC', 'WBC']
```

改动点：

- `train` / `val`：指定训练集和验证集图片路径。
- `nc: 3`：检测类别数。
- `names`：三个类别的名称。

### yolov5-master/models/yolov5s.yaml

**没有修改**模型结构文件，保持官方默认结构。
训练时通过 `data.yaml` 里的 `nc=3` 自动覆盖模型 yaml 中的 `nc=2`，日志中可见：

```
Overriding model.yaml nc=2 with nc=3
```

## 4. 代码做了哪些兼容性修改？

为适应 PyTorch 2.x / NumPy 2.x，对原始 YOLOv5 做了最小化兼容性补丁，**没有新增网络模块**，只是让它能在新环境下跑起来：

| 文件                                     | 主要修改                                                                                                                  |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `yolov5-master/train.py`               | `torch.load(weights, weights_only=False)`                                                                               |
| `yolov5-master/detect.py`              | `torch.load(..., weights_only=False)`                                                                                   |
| `yolov5-master/hubconf.py`             | `torch.load(..., weights_only=False)`                                                                                   |
| `yolov5-master/models/experimental.py` | `torch.load(..., weights_only=False)`                                                                                   |
| `yolov5-master/utils/general.py`       | `torch.load(..., weights_only=False)`；`np.int` 改为 `int`；`output_to_target` 中 CUDA tensor 先转 CPU 再转 numpy |
| `yolov5-master/utils/datasets.py`      | `torch.load(..., weights_only=False)`；`np.int` 改为 `int`                                                          |
| `yolov5-master/models/yolo.py`         | `_initialize_biases` 中避免 in-place 修改 leaf Variable                                                                 |

> 如果加载的是已经训练过的权重，`train.py` 约第 140 行会自动把新设置的 epochs 追加到已训练 epoch 数上，实现继续微调。

## 5. 训练命令

```bash
cd yolov5-master
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python train.py --data ../BCCDDataSet/data.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --epochs 100 --batch-size 8 --img 416
```

测试结果保存在 `yolov5-master/runs/exp*/weights/`，包含 `best.pt` 和 `last.pt`。

## 6. 检测/推理

```bash
python detect.py --weights runs/exp6/weights/best.pt --source ../BCCDDataSet/test/images --img 416 --save-txt
```

结果

Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:03<00:00,  1.66it/s]
                 all          73         967       0.646       0.947       0.888       0.603
Optimizer stripped from runs\exp0\weights\last.pt, 14.7MB
Optimizer stripped from runs\exp0\weights\best.pt, 14.7MB
100 epochs completed in 0.136 hours.

## 7. 下一步改进方向

- **升级到 YOLOv8 / YOLO11**：Ultralytics 新版 API 更统一，训练更稳定，支持更多内置增强。
- **加入注意力机制**：在 Backbone 或 Neck 中插入 CBAM、SE、ECA 等注意力模块，提升小目标和难例检出率。
- **数据增强**：使用 Mosaic、MixUp、Copy-Paste、Albumentations 等策略增强泛化能力。
- **超参与 Anchor**：自动锚框计算、调整学习率、将 `imgsz` 提升到 640 或 1280。
- **模型轻量化**：尝试 YOLOv5n / YOLOv8n，便于部署到移动端或边缘设备。
- **迁移学习**：在皮肤病公开数据集（如 ISIC、HAM10000）上预训练，再微调目标数据。

## 8. 许可证与使用声明

- YOLOv5 源码采用 **GPL-3.0** 许可证，请保留原始 LICENSE 文件。
- BCCD 数据集采用 **MIT / 类公共领域** 授权，使用时应按 Roboflow 要求引用。
