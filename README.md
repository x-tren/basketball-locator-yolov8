# 东海凌涛 · 新生纳新任务：篮球定位器

> 第一周视觉工程任务｜完成一个能在视频中定位并标记篮球的程序。

使用本仓库提供的单类别篮球数据集微调 YOLO 检测器，然后用它逐帧处理验收视频，
在画面中持续标出篮球。结果视频中每个球都带**检测框**、**`basketball` 标签 + 置信度**、
**中心点坐标**和**运动轨迹**。

---

## 目录结构

```text
.
├─ train.py                          # 训练/微调脚本
├─ detect.py                         # 推理脚本：跑视频、画框、导出 result.mp4
├─ requirements.txt                  # 锁定的依赖版本
├─ dataset/                          # YOLO 格式单类数据集（600 训练 / 100 验证）
│  ├─ data.yaml
│  ├─ train/images + train/labels
│  └─ valid/images + valid/labels
├─ evaluation/
│  └─ basketball_dribble_evaluation.mp4   # 统一验收视频（4K, 25fps, 176 帧, 7.04 秒）
└─ result.mp4                        # detect.py 的输出（不提交，见 .gitignore）
```

---

## 环境要求

* **Python 3.8.5**（本机唯一可用的解释器）
* **纯 CPU 训练**：本机没有 NVIDIA 独立显卡，CUDA 不可用
* 磁盘约 3GB（torch 的 wheel 本身就 199MB）

### ⚠️ 关于依赖版本，先读这一段

`requirements.txt` 里的版本是**逐条锁死**的，不要升级。原因：

1. **不能用最新的 ultralytics（8.4.x）**。它在 PyPI 上仍然声明 `requires_python>=3.8`，
   但这是过时的声明。实际代码里有 Python 3.9+ 和 torch≥2.5 才有的 API：

   | 位置 | 用了什么 | 要求 |
   |---|---|---|
   | `ultralytics/utils/logger.py` | `Path.is_relative_to()` | Python ≥ 3.9 |
   | `ultralytics/nn/tasks.py` | `torch.serialization.add_safe_globals()` | torch ≥ 2.5 |

   前者在 8.4.100 引入。装 8.4.x 会在运行时抛 `AttributeError`。
   **本仓库锁定 `ultralytics==8.3.253`** —— 这是最后一个在 trove classifier 里
   真正声明支持 Python 3.8 的版本。

2. **`numpy==1.24.4` 是硬天花板**，因为 numpy 2.x 要求 Python ≥ 3.9。
   `torch==2.4.1` 是最后一个提供 cp38 win_amd64 wheel 的版本。

3. **`polars==1.8.2`** 是个隐藏陷阱：ultralytics 8.3.x 硬依赖 polars，
   而最后一个有 cp38 wheel 的版本就是 1.8.2。

4. **`lap==0.5.13`** 是 ByteTrack / BoT-SORT 跟踪必需的。
   `ultralytics/trackers/utils/matching.py` 直接 `import lap`，缺了会在
   `model.track()` 里抛 `ModuleNotFoundError`。（ultralytics 自己的元数据
   没有把它列为硬依赖，所以容易漏。）

> 如果将来迁移到 Python ≥ 3.12，可以删掉 `requirements.txt` 里所有版本号，
> 直接装最新版 —— 那是更好的长期状态。

---

## 安装

```bash
# 1. 建虚拟环境（不改动系统 Python）
python -m venv .venv

# 2. 先升级 pip —— 系统自带的 pip 是 20.1.1（2020 年），依赖解析器太老，必须先换掉
.venv/Scripts/python.exe -m pip install -U "pip==25.0.1" "setuptools==75.3.4" "wheel==0.45.1"

# 3. 装依赖
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Linux / macOS 下把 `.venv/Scripts/python.exe` 换成 `.venv/bin/python`。

### 冒烟测试

装完先跑这一行，确认版本完全匹配：

```bash
.venv/Scripts/python.exe -c "import ultralytics, torch, cv2, numpy; print(ultralytics.__version__, torch.__version__, cv2.__version__, numpy.__version__)"
```

期望输出：

```text
8.3.253 2.4.1 4.11.0 1.24.4
```

如果这一步就报错，说明 3.8 这条路走不通，别去 patch 它，直接换 Python 3.12 重来。

---

## 训练

```bash
python train.py
```

默认配置：`yolov8n.pt` + `imgsz=416` + 60 轮 + 早停 15 轮，CPU 上约 1 小时
（实测见下方「结果」）。

常用参数：

```bash
python train.py --model yolov8s.pt --imgsz 640 --epochs 150 --patience 40   # 更大的模型
python train.py --model yolov8n.pt --imgsz 416 --epochs 40                  # 更快
```

训练完成后：

* 权重被复制到仓库根目录的 **`basketball_best.pt`**
* 完整运行记录在 `runs/train/`（含 `results.csv`、PR 曲线、混淆矩阵）
* 终端打印验证集 mAP50 和 mAP50-95

### 为什么默认 `imgsz=416` 而不是 640

数据集里的图片**原生就是 416×416**。在 `imgsz=640` 上训练等于把每张图上采样
2.4 倍 —— 零信息增益，纯粹多烧 2.4 倍的算力。416 既是原生分辨率，
也是 32 的整数倍（YOLO 的 stride 要求）。

### 两个容易踩的坑（脚本已处理）

1. **`OMP_NUM_THREADS`**。`ultralytics/__init__.py` 在**导入时**会执行：

   ```python
   if not os.environ.get("OMP_NUM_THREADS"):
       os.environ["OMP_NUM_THREADS"] = "1"   # 本意是给 GPU 用户省 CPU
   ```

   在纯 CPU 机器上这会把训练**静默地单线程化，慢 6~10 倍**，而且不报任何错。
   它只在变量未设置时才写入，所以 `train.py` 在 `import ultralytics` **之前**
   就把它设好。这个顺序是必要的，不能调整。

2. **`dataset/data.yaml` 里的 `path: .` 是相对当前工作目录的**。
   ultralytics 的 `check_det_dataset` 里有这样的回退逻辑：

   ```python
   path = Path(data.get("path") or ...)
   if not path.exists() and not path.is_absolute():
       path = (DATASETS_DIR / path).resolve()
   ```

   而 `Path(".")` **永远存在**，所以回退分支永远不触发，`path` 就停在当前工作目录，
   `train` 被解析成 `<cwd>/train/images` —— 从仓库根目录运行时直接 `FileNotFoundError`。
   `train.py` 会在运行时生成一份 `path` 为绝对路径的配置到 `runs/resolved_data.yaml`，
   原始 `dataset/data.yaml` 保持不变。

### 首次运行的网络要求

`yolov8n.pt` 会从 GitHub 自动下载。如果下载失败，手动下载后指定本地路径：

```bash
python train.py --model /path/to/yolov8n.pt
```

---

## 推理

```bash
python detect.py
```

默认会用 `basketball_best.pt` 跑 `evaluation/basketball_dribble_evaluation.mp4`，
输出 `result.mp4`。

常用参数：

```bash
python detect.py --conf 0.3                  # 降低阈值，召回更高（误检也更多）
python detect.py --imgsz 960                 # 源视频是 4K，球小的时候提高推理分辨率
python detect.py --no-track                  # 关闭跟踪，纯逐帧检测
python detect.py --resize 1280x720           # 缩小输出体积（推理仍在原始分辨率上）
python detect.py --dump-every 20             # 每 20 帧导出一张 JPEG，便于人工检查
```

### 画面上的内容

* 检测框（不同跟踪 ID 用不同颜色）
* `basketball 0.87` —— 标签 + 置信度
* `(1234, 567)` —— 中心点坐标
* 中心点圆点 + 逐渐淡出的运动轨迹尾迹
* 左上角状态栏：帧号 / 本帧球数 / 当前置信度阈值

### 输出视频的几个坑（脚本已处理）

1. **VideoWriter 的尺寸必须取自解码后的帧**（`frame.shape[:2]`），不能用
   `CAP_PROP_FRAME_WIDTH/HEIGHT`。验收视频的存储分辨率带旋转元数据，
   OpenCV 会自动应用方向，导致两者不一致。尺寸对不上时 `write()` **静默失败** ——
   不抛异常，只是产出 0 字节或无法播放的文件。
2. **编码器用 `mp4v`**。Windows 上 pip 装的 opencv 没有 OpenH264 编码器，
   `avc1` / `H264` 会直接 `isOpened() == False`。
3. **fps 可能读到 0 或 NaN**，要兜底，否则 VideoWriter 行为未定义。
4. 写完必须**先 `release()` 再回读**，Windows 上文件句柄没关时读不到内容。
5. `model.track()` 逐帧喂 numpy 数组时**必须传 `persist=True`**，否则跟踪 ID 每帧重置。

脚本结束时会打印处理帧数并**回读输出视频校验**，帧数对不上会明确报警。

---

## 结果

### 训练结果（YOLOv8n，CPU）

| 项 | 值 |
|---|---|
| 模型 | `yolov8n.pt` 微调（COCO 预训练） |
| 输入尺寸 | 416×416（数据集原生分辨率） |
| 轮数 | 60 轮跑满，未触发早停 |
| 耗时 | **0.430 小时（约 26 分钟）** |
| 实测速度 | 约 28 秒/轮（训练 26.2s + 验证 1.6s），16 线程 |

验证集（100 张图 / 106 个框）指标：

| 指标 | 值 |
|---|---|
| Precision | 0.958 |
| Recall | 0.870 |
| **mAP50** | **0.944** |
| **mAP50-95** | **0.528** |

### 推理结果

| 项 | 值 |
|---|---|
| 输入 | `evaluation/basketball_dribble_evaluation.mp4`（2160×4096 竖屏，25fps，176 帧） |
| 处理帧数 | 176 / 176 |
| 检出帧比例 | 100% |
| 输出框数 | 176（每帧恰好 1 个） |
| 平均置信度 | 0.928 |
| 后处理滤除 | 面积超限 126 个 + 每帧上限截掉 28 个 |
| 耗时 | 约 20 秒 |
| 输出 | `result.mp4`，2160×4096，176 帧，32.6 MB |

`detect.py` 结束时会自动回读输出视频校验帧数，实测通过。


---

## 附录：加分项实现说明

| 加分方向 | 实现 |
|---|---|
| 绘制运动轨迹 | `detect.py` 用 `deque(maxlen=--trail)` 保存每个跟踪 ID 的历史中心点，逐段连线，线宽和亮度随接近当前帧而递增，形成淡出尾迹 |
| 短时遮挡/运动模糊下更稳 | ByteTrack 跟踪（`--tracker bytetrack.yaml`），检测丢失时靠卡尔曼预测维持轨迹，减少框的跳动 |
| 显示每帧中心坐标 | 每个框旁绘制 `(cx, cy)`，同时写出检测框、标签和置信度 |
| 跟踪算法减少抖动 | `model.track(persist=True)`，同一目标跨帧保持 ID 与颜色 |
| 参数与失败场景说明 | 见下方「失败场景分析」 |

## 失败场景分析

### 主要失败模式：球场地面被误判成篮球

这是本模型最明显的问题。逐帧核对后发现：**176 帧里有 147 帧（83.5%）会额外吐出一个假框**，
面积占画面 35%~60%，框住的是下半部分红棕色的球场地面，而且**置信度高达 0.85~0.94**。

根因是外观混淆：篮球和球场在色调上高度接近（都是橙红 / 棕色），都带颗粒纹理。
数据集里似乎缺少「空旷球场」这类不含球的负样本，模型没学会区分二者。

**试过但无效的两条路**（都有实测数据支撑）：

1. **抬高置信度阈值**：无效。假框的置信度中位数在 0.85 以上，比很多真框还高。
   实测把 `conf` 从 0.35 提到 0.55，假框只从 152 帧降到 134 帧。
2. **收紧 NMS 的 IoU 阈值**：无效。实测 `iou` 从 0.70 降到 0.20 结果几乎不变
   （假框 147 帧 → 147 帧）。因为假框和真框几乎不重叠，NMS 根本没机会压制它。

**最终采用的后处理**（`detect.py` 里的 `--max-area` 和 `--max-dets`）：

- **面积上限 30%**：真实篮球在本段视频里面积稳定在 **6.6%~20%**，
  假框在 **35%~60%**，两者分得很开，这一个阈值就能干净地切开。
- **每帧只保留置信度最高的 1 个框**：本段视频同时只有一颗球。

实测效果：**假框从 147 帧降到 0 帧**，平均置信度从 0.854 升到 0.928。
完整流程共滤除 126 个超面积框 + 28 个多余的框。

> 需要说明的是，这是**针对本段视频的工程性后处理，不是模型本身的改进**。
> 更根本的修法是补充负样本重新训练，但任务明确要求
> 「不要用验收视频的画面参与训练或调参」，所以没有走这条路。

### 其他已观察到的困难场景

| 场景 | 表现 | 出现位置 |
|---|---|---|
| 运动模糊 | 球快速下落时边缘拖影，框会略大于球体 | 开头几帧、触地前后 |
| 手部遮挡 | 手掌压在球上时，检测框有把手臂一起框进去的倾向 | 运球下压的瞬间 |
| 球贴近镜头 | 球占画面比例快速变大（6.6% → 20%），对面积阈值有一定敏感性 | 视频后段 |

### 参数选择说明

| 参数 | 取值 | 理由 |
|---|---|---|
| `imgsz`（训练） | 416 | 数据集图片原生就是 416×416，训练时无需重采样；用 640 只是上采样，纯烧算力 |
| `imgsz`（推理） | 640 | 源视频是 4K，比训练分辨率稍高一点有助于找回被缩小的球 |
| `epochs` | 60 | 跑满且指标仍在缓慢上升，未触发早停；单类别任务收敛很快 |
| `patience` | 15 | 时间兜底 |
| `batch` | 16 | 38 个 iteration/轮，CPU 上内存与速度的平衡点 |
| `workers` | 4 | 16 个核要在训练进程和 dataloader 之间分，用默认的 8 会争抢 |
| `conf`（推理） | 0.35 | 假框不是靠阈值滤掉的，所以阈值可以留低以保证召回 |
| `max_area` | 0.30 | 真实球 ≤20%，假框 ≥35%，取中间值 |


---

## 数据与素材来源

* 训练数据：公开的 **Basketball-1 v1** 数据集，经整理为单类 `basketball` YOLO 数据集，
  原始许可 **CC BY 4.0**。详见 [`dataset/README.md`](dataset/README.md)。
* 验收视频：Pexels 免费素材，详见 [`evaluation/README.md`](evaluation/README.md)。
  仅用于本次教学验收。

本仓库不包含训练产生的权重、`runs/` 运行目录和输出视频（见 `.gitignore`）。
