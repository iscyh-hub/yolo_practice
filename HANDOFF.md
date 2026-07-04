# 项目交接说明

> 本文件用于在切换会话/人员时快速接手项目。

## 当前状态

- 项目路径：`D:\Project\基于YOLO5细胞检测实战`
- 代码基线：YOLOv5 官方旧版（`yolov5-master/`）
- 运行环境：`yolov5-bccd`（conda）
- 已验证：1 epoch 训练成功，RTX 5060 + CUDA 12.8 + PyTorch 2.8.0+cu128

## 已完成的工作

1. 创建并验证了 `yolov5-bccd` conda 虚拟环境。
2. 修改旧版 YOLOv5 代码，适配 PyTorch 2.x / NumPy 2.x：
   - 所有 `torch.load(...)` 增加 `weights_only=False`。
   - `np.int` 全部改为 `int`。
   - `models/yolo.py` 的 `_initialize_biases` 避免 in-place 修改 leaf Variable。
   - `utils/general.py` 的 `output_to_target` 将 CUDA tensor 先 `.cpu().item()` 再转 numpy。
3. 训练 1 epoch 成功，权重保存在 `yolov5-master/runs/exp6/weights/`。
4. 编写了项目说明文件：
   - `README.md`
   - `environment.yml`
   - `requirements.txt`
5. **新增 YOLOv8 迁移与注意力机制（`yolov8/`）**：
   - 克隆 Ultralytics `v8.3.213` 到 `yolov8/ultralytics/`，使用 editable 安装（通过 `yolov8-ultralytics.pth`）。
   - 新增 `ultralytics/nn/modules/attention.py`：SE、ECA、C2f_CBAM、C2f_SE、C2f_ECA。
   - 修改 `ultralytics/nn/modules/__init__.py` 与 `ultralytics/nn/tasks.py` 注册新模块。
   - 新增模型配置：`yolov8/cfg/yolov8-bccd.yaml`、`yolov8-cbam-bccd.yaml`、`yolov8-se-bccd.yaml`、`yolov8-eca-bccd.yaml`。
   - 新增数据集配置 `yolov8/bccd.yaml`、训练脚本 `yolov8/scripts/train.py`、对比脚本 `yolov8/scripts/val_all.py`。
   - 4 个 variant（baseline / CBAM / SE / ECA）的 YOLOv8n 1-epoch 快速验证全部通过，无 CUDA OOM。
   - 更新 `requirements.txt`，加入 `-e yolov8/ultralytics`。

## YOLOv5 训练命令

```powershell
cd D:\Project\基于YOLO5细胞检测实战\yolov5-master
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python train.py --data ../BCCDDataSet/data.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --epochs 100 --batch-size 8 --img 416
```

## YOLOv8 训练命令

1. 快速验证（1 epoch / YOLOv8n）：

   ```powershell
   cd D:\Project\基于YOLO5细胞检测实战
   $env:KMP_DUPLICATE_LIB_OK="TRUE"
   python yolov8/scripts/train.py --variant baseline --scale n --epochs 1
   python yolov8/scripts/train.py --variant cbam   --scale n --epochs 1
   python yolov8/scripts/train.py --variant se     --scale n --epochs 1
   python yolov8/scripts/train.py --variant eca    --scale n --epochs 1
   ```

2. 标准对比训练（YOLOv8s / 150 epochs）：

   ```powershell
   cd D:\Project\基于YOLO5细胞检测实战
   $env:KMP_DUPLICATE_LIB_OK="TRUE"
   python yolov8/scripts/train.py --variant baseline --scale s --epochs 150
   python yolov8/scripts/train.py --variant cbam   --scale s --epochs 150
   python yolov8/scripts/train.py --variant se     --scale s --epochs 150
   python yolov8/scripts/train.py --variant eca    --scale s --epochs 150
   ```

3. 测试集对比评估：

   ```powershell
   python yolov8/scripts/val_all.py --scale s
   ```

## 已知注意事项

- Windows 训练前必须设置 `$env:KMP_DUPLICATE_LIB_OK="TRUE"`（PowerShell）或 `set KMP_DUPLICATE_LIB_OK=TRUE`（CMD），否则会报 `libiomp5md.dll` 冲突。
- 目前 `yolov5s.yaml` 的 `nc=2` 未改，实际类别数由 `BCCDDataSet/data.yaml` 的 `nc=3` 覆盖。
- 有两个 FutureWarning 不影响运行，如需清理可后续修改 `train.py` 中的 `torch.cuda.amp` 为 `torch.amp`。
- **YOLOv8 注意**：`yolov8/ultralytics` 为本地源码 editable 安装，修改源码后无需重新安装；但删除该目录后需重新 clone 并确保 `.pth` 指向正确。
- **数据集缓存**：YOLOv5 生成的 `labels.cache` 是 `.npz` 格式，与 Ultralytics 的 `.npy` 不兼容。若切换版本训练，请先删除 `BCCDDataSet/*/labels.cache`。
- **显存**：RTX 5060 8GB 跑 YOLOv8s@416 batch=16 约 4-5GB，YOLOv8m 建议 batch=8。

## 下一步建议

1. **正式训练 YOLOv8**：在 `yolov8/` 下使用 YOLOv8s 跑满 100~150 epochs，记录 baseline 与 CBAM/SE/ECA 的 mAP、参数量、推理速度对比。
2. **对比 YOLOv5**：保留 `yolov5-master/` 不动，论文中可对比 YOLOv5 baseline 与 YOLOv8(+attention) 的结果。
3. **数据集**：当前使用 BCCD；如需改为皮肤病数据集，复制一份并按 YOLO 格式调整 `yolov8/bccd.yaml` 的 `path`/`names`/`nc`。
4. **工程优化**：开启 Mosaic/MixUp、自动锚框计算、TensorBoard 可视化、模型导出（ONNX/TensorRT）。
5. **论文撰写**：补充数据集来源、注意力机制插入位置、预训练权重转移说明和 License 声明。

## 待确认/待办

- [ ] 确定最终选题是“血细胞检测”还是“皮肤病检测”。
- [ ] 如需换数据集，准备目标数据集并完成格式转换（YOLO 格式）。
- [ ] 跑满 YOLOv8s 正式训练（baseline / CBAM / SE / ECA），生成对比表格。
- [ ] 同时保留 YOLOv5 baseline 结果，用于论文对比。
- [ ] 撰写论文/毕设时补充数据集来源、YOLOv8 改进点（注意力机制插入位置、参数量变化）、训练超参和 License 声明。

## 联系人/备注

- 无特定联系人。
- 后续修改建议保留此文件并更新状态。
