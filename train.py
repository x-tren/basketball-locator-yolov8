#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py — 在单类别（basketball）数据集上微调 YOLO 检测器，纯 CPU 训练。

用法::

    python train.py                                   # 默认 yolov8n @ 416，约 1 小时
    python train.py --model yolov8s.pt --imgsz 640 --epochs 150 --patience 40

训练完成后会把 best.pt 复制到仓库根目录（默认 basketball_best.pt），
并打印验证集 mAP，供 detect.py 直接使用。
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
RUNS_DIR = ROOT / "runs"
# 解析后的数据配置写进 runs/（已被 .gitignore 排除），不污染原始的 dataset/data.yaml
RESOLVED_YAML = RUNS_DIR / "resolved_data.yaml"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="在篮球数据集上微调 YOLO（CPU）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="yolov8n.pt",
                   help="预训练权重名（首次运行会自动从 GitHub 下载）或本地 .pt 路径")
    p.add_argument("--imgsz", type=int, default=416,
                   help="训练分辨率。本数据集图片原生就是 416x416，"
                        "调到 640 只会把图片上采样 2.4 倍，零信息增益，纯烧算力")
    p.add_argument("--epochs", type=int, default=60, help="最大训练轮数")
    p.add_argument("--patience", type=int, default=15,
                   help="早停：连续这么多轮验证指标没有提升就停止")
    p.add_argument("--batch", type=int, default=16, help="批大小")
    p.add_argument("--workers", type=int, default=4,
                   help="数据加载进程数。不要用默认的 8：16 个核要在训练进程和 "
                        "dataloader 之间分，CPU 训练时争抢很明显")
    p.add_argument("--threads", type=int, default=0,
                   help="OMP 线程数，0 表示用全部 CPU 核心")
    p.add_argument("--device", default="cpu", help="训练设备；本机无 NVIDIA 显卡，只能用 cpu")
    p.add_argument("--name", default="train", help="runs/ 下的运行目录名")
    p.add_argument("--out", default="basketball_best.pt",
                   help="训练完把 best.pt 复制到这个文件名（相对仓库根目录）")
    p.add_argument("--no-cache", action="store_true",
                   help="关闭把图片缓存进内存（数据只有约 312MB，默认开启）")
    return p.parse_args(argv)


def configure_threads(n):
    """
    设置 OpenMP 线程数。

    ★ 这个函数的调用时机是 load-bearing 的：必须在 ``import ultralytics`` 之前调用。★

    ultralytics/__init__.py 在模块导入时会执行::

        if not os.environ.get("OMP_NUM_THREADS"):
            os.environ["OMP_NUM_THREADS"] = "1"

    它的本意是给「用 GPU 训练」的大多数人省下 CPU，只在变量未设置时才写入。
    所以在纯 CPU 机器上，如果先 import 了 ultralytics，训练会被静默地
    单线程化 —— 不会报任何错，只是慢 6~10 倍。我们先设置就赢了。
    """
    n = n or os.cpu_count() or 4
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    # 关闭 ultralytics 的自动装包：它默认会自己跑 pip，而且用的是当前进程的
    # 解释器，实测会把包装进系统 Python 而不是虚拟环境。缺什么由 requirements.txt 负责。
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    return os.environ["OMP_NUM_THREADS"]


def write_resolved_yaml():
    """
    生成一份 ``path`` 为绝对路径的数据配置副本，返回其路径。

    原始的 dataset/data.yaml 里写的是 ``path: .``，这是相对当前工作目录的。
    ultralytics 的 check_det_dataset 会做::

        path = Path(data.get("path") or ...)
        if not path.exists() and not path.is_absolute():
            path = (DATASETS_DIR / path).resolve()

    而 ``Path(".")`` 永远存在，所以那个回退分支永远不会触发，``path`` 就停留在
    当前工作目录；随后 ``train`` 被解析成 ``<cwd>/train/images``。于是从仓库根目录
    运行训练时会直接 FileNotFoundError。这里把 path 换成绝对路径来规避。
    """
    import yaml

    src = DATASET_DIR / "data.yaml"
    if not src.exists():
        sys.exit("[train] 找不到数据配置: %s" % src)

    with open(str(src), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["path"] = str(DATASET_DIR)

    # 提前检查目录，给出比 ultralytics 更清楚的报错
    for key in ("train", "val"):
        d = DATASET_DIR / cfg.get(key, "")
        if not d.is_dir():
            sys.exit("[train] 数据配置里的 %s 目录不存在: %s" % (key, d))

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(RESOLVED_YAML), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return RESOLVED_YAML


def main(argv=None):
    args = parse_args(argv)

    threads = configure_threads(args.threads)
    print("[train] OMP_NUM_THREADS = %s" % threads)

    # 故意延迟导入：必须在 configure_threads() 之后，见上面的说明
    from ultralytics import YOLO

    yml = write_resolved_yaml()
    print("[train] 数据配置 : %s" % yml)
    print("[train] 权重     : %s" % args.model)
    print("[train] imgsz=%d epochs=%d patience=%d batch=%d device=%s"
          % (args.imgsz, args.epochs, args.patience, args.batch, args.device))

    try:
        model = YOLO(args.model)
    except Exception as e:
        print("\n[train] 加载权重失败: %s" % e)
        print("[train] 如果是网络问题（无法从 GitHub 下载 %s），请手动下载该文件，" % args.model)
        print("[train] 然后运行: python train.py --model /path/to/%s" % args.model)
        return 1

    results = model.train(
        data=str(yml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        amp=False,               # 混合精度只在 CUDA 上有意义
        cache=not args.no_cache,
        seed=0,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        plots=True,
        val=True,
    )

    best = RUNS_DIR / args.name / "weights" / "best.pt"
    if not best.exists():
        print("[train] 没有找到训练产物: %s" % best)
        return 1

    dest = ROOT / args.out
    shutil.copy2(str(best), str(dest))

    metrics = getattr(results, "results_dict", None) or {}

    def fmt(key):
        v = metrics.get(key)
        try:
            return "%.4f" % float(v)
        except (TypeError, ValueError):
            return "n/a"

    print("")
    print("=" * 62)
    print("训练完成")
    print("  mAP50    : %s" % fmt("metrics/mAP50(B)"))
    print("  mAP50-95 : %s" % fmt("metrics/mAP50-95(B)"))
    print("  权重     : %s" % dest)
    print("  运行目录 : %s" % (RUNS_DIR / args.name))
    print("  下一步   : python detect.py --weights %s" % args.out)
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
