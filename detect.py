#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
detect.py — 用训练好的篮球检测器跑验收视频，输出带标注的结果视频。

在每个检测到的篮球上画：
  * 检测框
  * 标签 `basketball` + 置信度
  * 中心点坐标 (x, y)
  * 运动轨迹（逐渐淡出的尾迹）
可选 ByteTrack 跟踪，用来抑制检测框逐帧抖动。

用法::

    python detect.py                                   # 用 basketball_best.pt 跑默认验收视频
    python detect.py --weights run1.pt --conf 0.3
    python detect.py --no-track --resize 1280x720
"""

import argparse
import math
import os
import sys
from collections import deque
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "evaluation" / "basketball_dribble_evaluation.mp4"
RUNS_DIR = ROOT / "runs"

# 不同跟踪 ID 用不同颜色，方便看出跟踪是否连贯
COLORS = [
    (0, 255, 0), (0, 165, 255), (255, 128, 0), (255, 0, 255),
    (0, 255, 255), (255, 255, 0), (128, 0, 255), (0, 0, 255),
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="在视频中定位篮球并输出标注视频",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights", default=None,
                   help="模型权重路径。不指定则自动找根目录的 .pt，再找 runs 里最新的 best.pt")
    p.add_argument("--source", default=str(DEFAULT_SOURCE), help="输入视频")
    p.add_argument("--output", default=str(ROOT / "result.mp4"), help="输出视频")
    p.add_argument("--conf", type=float, default=0.35, help="置信度阈值")
    p.add_argument("--imgsz", type=int, default=640,
                   help="推理分辨率。源视频是 4K，球如果偏小可以提高（如 960）")
    p.add_argument("--device", default="cpu", help="推理设备")
    p.add_argument("--no-track", action="store_true",
                   help="关闭跟踪，改用纯逐帧检测（默认开启 ByteTrack 以防抖动）")
    p.add_argument("--tracker", default="bytetrack.yaml", help="跟踪器配置")
    p.add_argument("--max-dets", type=int, default=1,
                   help="每帧最多保留几个检测框（按置信度从高到低）。默认 1，"
                        "因为本段视频里同时只有一颗球。设为 0 表示不限制")
    p.add_argument("--max-area", type=float, default=0.30,
                   help="检测框面积占画面的比例上限，超过就丢弃。用来滤掉模型把"
                        "红棕色球场地面误判成篮球产生的大框（见 README 失败场景分析）。"
                        "设为 0 表示不限制")
    p.add_argument("--trail", type=int, default=30,
                   help="轨迹保留的历史中心点数量，0 表示不画轨迹")
    p.add_argument("--codec", default="mp4v",
                   help="输出编码。Windows 上 pip 安装的 opencv 没有 H264 编码器，"
                        "avc1/H264 会直接失败，保持 mp4v")
    p.add_argument("--resize", default=None,
                   help="缩放输出尺寸，格式 WxH（如 1280x720）。推理仍在原始分辨率上做。"
                        "默认不缩放（保持 4K）；4K 的 mp4v 文件偏大")
    p.add_argument("--dump-every", type=int, default=0,
                   help="每 N 帧额外导出成 JPEG 以便人工检查，0 表示关闭")
    p.add_argument("--dump-dir", default=None, help="导出帧的目录，默认为输出视频旁的 frames/")
    p.add_argument("--threads", type=int, default=0, help="OMP 线程数，0 表示用全部核心")
    return p.parse_args(argv)


def configure_threads(n):
    """
    必须在 ``import ultralytics`` 之前调用，原因见 train.py 里的同名函数。

    顺带关掉 ultralytics 的自动装包：它默认会在缺依赖时自己跑 pip，而且用的是
    当前进程的解释器 —— 实测会把包装到系统 Python 而不是虚拟环境里，
    让环境变得难以复现。这里显式关闭，缺什么由 requirements.txt 负责。
    """
    n = n or os.cpu_count() or 4
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    return os.environ["OMP_NUM_THREADS"]


def find_weights(explicit):
    """按优先级找一个可用的权重文件。"""
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit("[detect] 权重文件不存在: %s" % p)
        return p

    # 1) 仓库根目录下的 .pt
    root_pts = sorted(ROOT.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if root_pts:
        return root_pts[0]

    # 2) runs/ 下最新的 best.pt
    bests = list(RUNS_DIR.glob("**/weights/best.pt")) if RUNS_DIR.is_dir() else []
    if bests:
        return max(bests, key=lambda x: x.stat().st_mtime)

    sys.exit("[detect] 没有找到任何权重文件。请先运行 train.py，或用 --weights 指定路径。")


def parse_resize(spec):
    """把 '1280x720' 解析成 (1280, 720)。"""
    if not spec:
        return None
    try:
        w, h = spec.lower().split("x")
        w, h = int(w), int(h)
    except ValueError:
        sys.exit("[detect] --resize 格式应为 WxH，例如 1280x720，收到: %s" % spec)
    if w <= 0 or h <= 0:
        sys.exit("[detect] --resize 的宽高必须为正数")
    return (w, h)


def put_text(img, text, org, color, scale=0.6, thickness=2):
    """画带深色底衬的文字，保证在任意背景上都读得清。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x, y - th - baseline - 2), (x + tw + 4, y + baseline + 2),
                  (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_trail(frame, pts, color):
    """把历史中心点连成一条逐渐淡出的折线。"""
    if len(pts) < 2:
        return
    n = len(pts)
    for i in range(1, n):
        frac = i / float(n)          # 越靠近当前帧越亮、越粗
        c = tuple(int(v * (0.25 + 0.75 * frac)) for v in color)
        th = max(1, int(round(1 + 3 * frac)))
        cv2.line(frame, pts[i - 1], pts[i], c, th, cv2.LINE_AA)


def main(argv=None):
    args = parse_args(argv)

    threads = configure_threads(args.threads)
    print("[detect] OMP_NUM_THREADS = %s" % threads)

    from ultralytics import YOLO   # 延迟导入，保证上面的环境变量先生效

    weights = find_weights(args.weights)
    print("[detect] 权重     : %s" % weights)

    source = Path(args.source)
    if not source.exists():
        sys.exit("[detect] 输入视频不存在: %s" % source)
    print("[detect] 输入视频 : %s" % source)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)   # VideoWriter 不会自动建目录

    dump_size = parse_resize(args.resize)

    model = YOLO(str(weights))

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        sys.exit("[detect] 无法打开视频: %s" % source)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or math.isnan(fps) or fps <= 0 or fps > 1000:
        print("[detect] 读到的 fps 不可用 (%r)，回退到 25.0" % fps)
        fps = 25.0

    ok, frame = cap.read()
    if not ok:
        cap.release()
        sys.exit("[detect] 读不出第一帧")

    # ★ 尺寸必须取自解码后的帧，不能用 CAP_PROP_FRAME_WIDTH/HEIGHT ★
    # 验收视频的存储分辨率带旋转元数据，OpenCV 会自动应用方向，于是 CAP_PROP 报的
    # 尺寸和 read() 实际返回的帧尺寸可能不一致。VideoWriter 尺寸对不上时 write()
    # 会静默失败 —— 不抛异常，只是产出一个 0 字节或无法播放的文件。
    src_h, src_w = frame.shape[:2]
    print("[detect] 帧尺寸   : %dx%d @ %.2f fps" % (src_w, src_h, fps))

    writer_size = dump_size if dump_size else (src_w, src_h)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*args.codec),
                             fps, writer_size)
    if not writer.isOpened():
        cap.release()
        sys.exit("[detect] VideoWriter 打开失败 (codec=%s, size=%dx%d)。\n"
                 "         试试 --codec mp4v，或换一个输出文件名。"
                 % (args.codec, writer_size[0], writer_size[1]))

    dump_dir = None
    if args.dump_every > 0:
        dump_dir = Path(args.dump_dir) if args.dump_dir else RUNS_DIR / "frames"
        dump_dir.mkdir(parents=True, exist_ok=True)
        print("[detect] 抽样帧导出到: %s" % dump_dir)

    trails = {}            # track_id -> deque[(x, y)]
    frame_area = float(src_w * src_h)
    n_frames = 0
    n_with_det = 0
    n_dets = 0
    n_rej = 0              # 被面积上限丢弃的检测数
    n_drop_extra = 0       # 被 --max-dets 截掉的检测数
    conf_sum = 0.0

    while ok:
        if args.no_track:
            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                                   device=args.device, verbose=False)[0]
        else:
            result = model.track(frame, persist=True, tracker=args.tracker,
                                 conf=args.conf, imgsz=args.imgsz,
                                 device=args.device, verbose=False)[0]

        boxes = result.boxes
        n_here = 0
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

            # ---- 后处理：面积上限 + 每帧保留前 N 个 ----
            # 模型的已知失败模式是把红棕色的球场地面看成篮球，吐出面积占画面
            # 30%~60% 的假框。真实篮球在本段视频里稳定在 6%~20%，
            # 所以先按面积剔掉明显不可能的，再按置信度取前 N 个。
            cand = []
            for i in range(len(xyxy)):
                area = (xyxy[i][2] - xyxy[i][0]) * (xyxy[i][3] - xyxy[i][1]) / frame_area
                if args.max_area > 0 and area > args.max_area:
                    n_rej += 1
                    continue
                cand.append(i)
            cand.sort(key=lambda j: -confs[j])
            if args.max_dets > 0 and len(cand) > args.max_dets:
                n_drop_extra += len(cand) - args.max_dets
                cand = cand[:args.max_dets]

            for i in cand:
                x1, y1, x2, y2 = [int(round(v)) for v in xyxy[i][:4]]
                conf = float(confs[i])
                tid = int(ids[i]) if ids is not None else 0
                color = COLORS[tid % len(COLORS)]

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 标签 + 置信度，画在框的上方；贴到画面顶边时改画在框内
                label = "basketball %.2f" % conf
                ly = y1 - 6 if y1 - 6 > 14 else y2 + 18
                put_text(frame, label, (x1, ly), color)

                # 中心点坐标
                put_text(frame, "(%d, %d)" % (cx, cy), (x1, ly + 20), color, scale=0.5, thickness=1)

                # 中心点本身
                cv2.circle(frame, (cx, cy), 4, color, -1, cv2.LINE_AA)

                # 轨迹
                if args.trail > 0:
                    if tid not in trails:
                        trails[tid] = deque(maxlen=args.trail)
                    trails[tid].append((cx, cy))
                    draw_trail(frame, list(trails[tid]), color)

                n_here += 1
                conf_sum += conf

        n_frames += 1
        if n_here:
            n_with_det += 1
            n_dets += n_here

        # 左上角状态栏
        put_text(frame, "frame %d   balls %d   conf>=%.2f"
                 % (n_frames, n_here, args.conf), (10, 28), (0, 255, 0),
                 scale=0.7, thickness=2)

        if dump_size:
            frame = cv2.resize(frame, dump_size, interpolation=cv2.INTER_AREA)

        writer.write(frame)

        if dump_dir and (n_frames % args.dump_every == 0):
            cv2.imwrite(str(dump_dir / ("frame_%04d.jpg" % n_frames)), frame)

        ok, frame = cap.read()

    # ★ 先释放再回读：Windows 上文件句柄没关时，后续 VideoCapture 读不到内容
    cap.release()
    writer.release()

    print("")
    print("=" * 62)
    print("推理完成")
    print("  输出视频     : %s" % out_path)
    print("  处理帧数     : %d" % n_frames)
    print("  含检测的帧   : %d (%.1f%%)"
          % (n_with_det, (100.0 * n_with_det / n_frames) if n_frames else 0.0))
    print("  检测框总数   : %d" % n_dets)
    if n_dets:
        print("  平均置信度   : %.3f" % (conf_sum / n_dets))
    print("  已滤除       : 面积超限 %d 个, 超出每帧上限 %d 个" % (n_rej, n_drop_extra))
    print("=" * 62)

    # 回读校验：确认写出的文件真的能打开、帧数对得上
    chk = cv2.VideoCapture(str(out_path))
    if not chk.isOpened():
        print("[detect] 警告：输出视频无法回读，可能编码失败")
        return 1
    out_frames = int(chk.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w = int(chk.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(chk.get(cv2.CAP_PROP_FRAME_HEIGHT))
    chk.release()
    size_mb = out_path.stat().st_size / 1048576.0
    print("[detect] 回读校验 : %dx%d, %d 帧, %.1f MB" % (out_w, out_h, out_frames, size_mb))
    if out_frames != n_frames:
        print("[detect] 警告：回读帧数 (%d) 与写入帧数 (%d) 不一致" % (out_frames, n_frames))
        return 1
    print("[detect] 校验通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
