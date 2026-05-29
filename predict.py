#!/usr/bin/env python3
"""MoGe 单帧/视频预测脚本 — 一键推理 + 去飞边 + 多格式输出"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import click
import cv2
import numpy as np
import torch
import utils3d
from pathlib import Path

from moge.model.v2 import MoGeModel
from moge.utils.io import save_glb, save_ply
from moge.utils.vis import colorize_depth, colorize_normal
from moge.utils.geometry_numpy import depth_occlusion_edge_numpy


def load_model(weights: str, device: torch.device) -> MoGeModel:
    model = MoGeModel.from_pretrained(weights).to(device).eval()
    model.half()
    return model


def extract_frames(src: str, frame_idx: list[int] | None) -> list[tuple[int, np.ndarray]]:
    """从图片或视频提取帧，返回 [(idx, bgr_array), ...]"""
    path = Path(src)
    if path.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv', '.webm'):
        cap = cv2.VideoCapture(src)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = frame_idx if frame_idx else [0]
        frames = []
        for idx in indices:
            if idx < 0:
                idx = total + idx
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append((idx, frame))
        cap.release()
        return frames
    else:
        img = cv2.imread(src)
        assert img is not None, f'无法读取: {src}'
        return [(0, img)]


def infer_single(model: MoGeModel, bgr: np.ndarray, device: torch.device,
                 resolution_level: int = 9) -> dict:
    """单帧推理，返回numpy结果字典"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
    with torch.no_grad():
        out = model.infer(tensor, resolution_level=resolution_level, use_fp16=True)
    return {k: v.cpu().numpy() for k, v in out.items()}


def postprocess(output: dict, image_rgb: np.ndarray, threshold: float = 0.02):
    """去飞边 + 构建mesh顶点数据
    联合 depth_map_edge + occlusion_edge，覆盖大深度跳变和手臂贴体等小深度差场景
    """
    points, depth, mask = output['points'], output['depth'], output['mask']
    normal = output.get('normal')
    h, w = depth.shape

    # 去飞边: 深度梯度阈值 | 遮挡边缘检测(对小深度差敏感)
    mask_bool = mask > 0.5
    edge = utils3d.np.depth_map_edge(depth, rtol=threshold) | \
           depth_occlusion_edge_numpy(depth, mask_bool, thickness=2, tol=0.02)
    mask_clean = mask_bool & ~edge

    # 构建mesh
    uv = utils3d.np.uv_map(h, w)
    colors_f = image_rgb.astype(np.float32) / 255
    if normal is not None:
        faces, verts, vcolors, vuvs, vnormals = utils3d.np.build_mesh_from_map(
            points, colors_f, uv, normal, mask=mask_clean, tri=True)
    else:
        faces, verts, vcolors, vuvs = utils3d.np.build_mesh_from_map(
            points, colors_f, uv, mask=mask_clean, tri=True)
        vnormals = None

    # OpenGL坐标系
    verts *= [1, -1, -1]
    vuvs = vuvs * [1, -1] + [0, 1]
    if vnormals is not None:
        vnormals *= [1, -1, -1]

    return dict(faces=faces, verts=verts, vcolors=vcolors, vuvs=vuvs,
                vnormals=vnormals, mask_clean=mask_clean,
                depth=depth, normal=normal, intrinsics=output['intrinsics'])


def save_outputs(out_dir: Path, stem: str, bgr: np.ndarray, result: dict,
                 fmt: set[str]):
    """按需保存各格式输出"""
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if 'vis' in fmt:
        depth_vis = colorize_depth(result['depth'])
        cv2.imwrite(str(out_dir / f'{stem}_depth.png'),
                    cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))
        if result['normal'] is not None:
            normal_vis = colorize_normal(result['normal'])
            cv2.imwrite(str(out_dir / f'{stem}_normal.png'),
                        cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR))

    if 'ply' in fmt:
        save_ply(out_dir / f'{stem}.ply', result['verts'],
                 np.zeros((0, 3), dtype=np.int32), result['vcolors'], result['vnormals'])

    if 'glb' in fmt:
        save_glb(out_dir / f'{stem}.glb', result['verts'], result['faces'],
                 result['vuvs'], rgb, result['vnormals'])


def segment_human(bgr: np.ndarray, sam3_weights: str, device: torch.device) -> np.ndarray:
    """SAM3 text prompt 分割人体，返回最近(最大面积)人体的bool mask (H,W)"""
    import torch.nn.functional as F
    from ultralytics.models.sam.build_sam3 import build_sam3_image_model

    model = build_sam3_image_model(sam3_weights).to(device).eval().half()
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 预处理
    img = cv2.resize(rgb, (1008, 1008))
    img_t = torch.tensor(img, dtype=torch.float16, device=device).permute(2, 0, 1)[None]
    img_t = (img_t - 127.5) / 127.5

    model.set_classes(text=["human"])
    with torch.no_grad():
        backbone_out = model.backbone.forward_image(img_t)
        outputs = model.forward_grounding(
            backbone_out=backbone_out,
            text_ids=torch.arange(1, device=device, dtype=torch.long))

    # 解析: 取置信度>0.3的mask中面积最大的(距镜头最近)
    scores = (outputs['pred_logits'].sigmoid() *
              outputs['presence_logit_dec'].sigmoid().unsqueeze(1)).squeeze(-1).flatten()
    masks = outputs['pred_masks'].flatten(0, 1)
    keep = scores > 0.3
    if not keep.any():
        print('[SAM3] 未检测到人体')
        return np.zeros((h, w), dtype=bool)

    masks_keep = F.interpolate(masks[keep].unsqueeze(1).float(), (h, w), mode='bilinear')[:, 0] > 0.5
    best = masks_keep.sum(dim=(1, 2)).argmax()
    mask = masks_keep[best].cpu().numpy()
    print(f'[SAM3] 人体分割: score={scores[keep][best]:.3f}, area={mask.sum()} px')

    del model
    torch.cuda.empty_cache()
    return mask


@click.command()
@click.option('-i', '--input', 'src', required=True, help='输入图片或视频路径')
@click.option('-o', '--output', 'out_dir', default='./output', help='输出目录')
@click.option('-w', '--weights', default='/root/paddlejob/workspace/env_run/penghaotian/models/moge-2-vitl-normal/model.pt')
@click.option('-f', '--frames', default='0', help='视频帧索引, 逗号分隔 (如 "0,10,50"), -1表示最后一帧')
@click.option('-t', '--threshold', default=0.02, type=float, help='去飞边阈值, 越小去除越多')
@click.option('-r', '--resolution', default=9, type=int, help='分辨率级别 [0-9]')
@click.option('--resize', type=str, default=None, help='输入缩放, 如 "360x640" (HxW)')
@click.option('--fmt', default='ply,vis', help='输出格式: ply,glb,vis (逗号分隔)')
@click.option('--human', 'sam3_weights', default=None, type=str,
              help='启用SAM3人体分割, 传入sam3.pt路径')
def main(src, out_dir, weights, frames, threshold, resolution, resize, fmt, sam3_weights):
    """MoGe 预测: 单目几何估计 + 去飞边 + 点云/Mesh导出"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    formats = set(fmt.split(','))
    frame_indices = [int(x) for x in frames.split(',')]

    print(f'[加载模型] {weights}')
    model = load_model(weights, device)

    print(f'[读取输入] {src}')
    frame_list = extract_frames(src, frame_indices)

    out_path = Path(out_dir)
    src_stem = Path(src).stem

    for idx, bgr in frame_list:
        # resize
        if resize:
            rh, rw = [int(x) for x in resize.split('x')]
            bgr = cv2.resize(bgr, (rw, rh))

        tag = f'{src_stem}_f{idx}'
        print(f'[推理] frame={idx}, shape={bgr.shape[:2]}')
        output = infer_single(model, bgr, device, resolution)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = postprocess(output, rgb, threshold)
        save_outputs(out_path, tag, bgr, result, formats)

        n_pts = result['verts'].shape[0]
        d_min, d_max = result['depth'][result['mask_clean']].min(), result['depth'][result['mask_clean']].max()
        print(f'  -> 全局: {n_pts} 顶点, 深度 [{d_min:.2f}, {d_max:.2f}]m')

        # SAM3 人体分割模式
        if sam3_weights:
            del model; torch.cuda.empty_cache()
            human_mask = segment_human(bgr, sam3_weights, device)
            # 保存mask
            cv2.imwrite(str(out_path / f'{tag}_human_mask.png'),
                        (human_mask * 255).astype(np.uint8))
            # 人体点云: mask_clean & human_mask
            mask_human = result['mask_clean'] & human_mask
            h, w = output['depth'].shape
            uv = utils3d.np.uv_map(h, w)
            colors_f = rgb.astype(np.float32) / 255
            normal = output.get('normal')
            if normal is not None:
                _, verts_h, vcolors_h, _, vnormals_h = utils3d.np.build_mesh_from_map(
                    output['points'], colors_f, uv, normal, mask=mask_human, tri=True)
            else:
                _, verts_h, vcolors_h, _ = utils3d.np.build_mesh_from_map(
                    output['points'], colors_f, uv, mask=mask_human, tri=True)
                vnormals_h = None
            verts_h *= [1, -1, -1]
            if vnormals_h is not None:
                vnormals_h *= [1, -1, -1]

            save_ply(out_path / f'{tag}_human.ply', verts_h,
                     np.zeros((0, 3), dtype=np.int32), vcolors_h, vnormals_h)
            print(f'  -> 人体: {verts_h.shape[0]} 顶点')
            # 重新加载MoGe用于后续帧
            model = load_model(weights, device)

    print(f'[完成] 输出目录: {out_path}')


if __name__ == '__main__':
    main()
