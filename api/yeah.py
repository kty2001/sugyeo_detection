# combine_run.py  (YOLOv11n sliding-window + person-only + SegFormer overlay)
# - 4K 작은 사람 대응: 기본 conf 낮춤(0.25), overlap 크게(0.5), tile=1280
# - 필요시 타일 업샘플 (--scale 1.5 등)로 초소형 객체 감지 강화
import os, sys, io, contextlib, warnings
from pathlib import Path
import argparse
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SEG_DIR = ROOT / "segformer"
sys.path.insert(0, str(SEG_DIR))

# Windows OpenCV DLL 이슈 방지, torchvision 경고 억제
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(os.path.join(sys.prefix, "Library", "bin"))
os.environ.setdefault("TORCHVISION_DISABLE_IMAGE", "1")
warnings.filterwarnings("ignore", message=r".*torch\.meshgrid.*indexing.*")

# torch.load weights_only 경고 무력화(호환용)
import torch
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# ==== YOLOv11n (Ultralytics) ====
from ultralytics import YOLO

# SegFormer
from transformers import SegformerImageProcessor
from src.model import create_model  # segformer/src/model.py

# ----------------------------- 유틸 -----------------------------
def _silent():
    return contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())

def collect_images(inp: str, recursive: bool=False):
    p = Path(inp)
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
    if p.is_file():
        return [p]
    if p.is_dir():
        it = p.rglob("*") if recursive else p.glob("*")
        return sorted([f for f in it if f.is_file() and f.suffix.lower() in exts], key=lambda x: str(x).casefold())
    if any(ch in inp for ch in ("*","?")):
        return sorted([f for f in Path().glob(inp) if f.is_file() and f.suffix.lower() in exts], key=lambda x: str(x).casefold())
    raise FileNotFoundError(f"input not found: {inp}")

def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1])
    x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    ua=(a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return 0.0 if ua==0 else inter/ua

def nms_global(boxes, iou_thr):
    if not boxes: return []
    boxes = sorted(boxes, key=lambda x: x["conf"], reverse=True)
    keep, used = [], [False]*len(boxes)
    for i in range(len(boxes)):
        if used[i]: continue
        keep.append(boxes[i])
        for j in range(i+1,len(boxes)):
            if not used[j] and iou(boxes[i]["bbox"], boxes[j]["bbox"]) >= iou_thr:
                used[j]=True
    return keep

def _parse_device_str(dev_str: str) -> str:
    """args.device: '0'|'1'|'cpu'|'0,1' 등 -> ultralytics device 문자열"""
    dev_str = str(dev_str).strip().lower()
    if dev_str in ("cpu", "-1"):
        return "cpu"
    return dev_str

# ----------------------------- 추론: 슬라이딩 윈도우 -----------------------------
@torch.no_grad()
def _predict_tile_ultra(tile_img_bgr, model: YOLO, device: str,
                        conf_thr: float, iou_thr_tile: float, max_det: int,
                        scale: float):
    """
    Ultralytics YOLOv11n로 타일 1장 추론.
    - person(클래스 0)만 감지: classes=[0]
    - 작은 사람 대응: 필요 시 scale>1.0로 업샘플 후 감지, box는 원래 타일 좌표로 환산
    """
    th, tw = tile_img_bgr.shape[:2]

    if scale != 1.0:
        # 업샘플: 초소형 객체에 유리 (비용 증가)
        tile_up = cv2.resize(tile_img_bgr, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        up_h, up_w = tile_up.shape[:2]
        imgsz = max(up_h, up_w)  # 한쪽 최대를 imgsz로, letterbox 사용
        src = tile_up
        inv_scale = 1.0/scale
    else:
        imgsz = max(th, tw)
        src = tile_img_bgr
        inv_scale = 1.0

    results = model.predict(
        source=src,
        verbose=False,
        device=device,
        conf=conf_thr,
        iou=iou_thr_tile,
        classes=[0],        # person only
        max_det=max_det,
        imgsz=imgsz
    )
    out = []
    if not results:
        return out
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return out

    xyxy = r.boxes.xyxy.detach().cpu().numpy()
    conf = r.boxes.conf.detach().cpu().numpy()
    cls  = r.boxes.cls.detach().cpu().numpy()

    # 업샘플 했다면 원래 타일 좌표로 복원
    for i in range(xyxy.shape[0]):
        x1, y1, x2, y2 = xyxy[i]
        if inv_scale != 1.0:
            x1 *= inv_scale; y1 *= inv_scale; x2 *= inv_scale; y2 *= inv_scale
        bx = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
        # 음수/초과 클램프
        bx[0] = max(0, min(bx[0], tw))
        bx[1] = max(0, min(bx[1], th))
        bx[2] = max(0, min(bx[2], tw))
        bx[3] = max(0, min(bx[3], th))
        if bx[2] > bx[0] and bx[3] > bx[1]:
            out.append({"bbox": bx, "conf": float(conf[i]), "cls": int(cls[i])})
    return out

@torch.no_grad()
def run_yolov11_sliding(image_path: str, model: YOLO, device: str,
                        tile=1280, overlap=0.5, conf_thr=0.25, iou_thr_tile=0.45,
                        iou_thr_global=0.6, max_det=3000, scale=1.0):
    """
    슬라이딩 윈도우(패딩 없음)로 전체 이미지 탐색 후 글로벌 NMS.
    - tile: 타일 한 변 길이 (정사각 형태로 슬라이드)
    - overlap: 타일 겹침 비율 (작을수록 빠름, 클수록 작은 객체/경계 보강)
    - scale: 타일 업샘플 배율(>1.0이면 초소형 객체 검출 강화, 비용↑)
    """
    im = cv2.imread(image_path)
    if im is None:
        raise FileNotFoundError(f"image load failed: {image_path}")
    H, W = im.shape[:2]

    stride = max(1, int(tile * (1 - overlap)))
    all_boxes = []

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            x2 = min(x + tile, W)
            y2 = min(y + tile, H)
            tile_im = im[y:y2, x:x2]
            th, tw = tile_im.shape[:2]
            if th == 0 or tw == 0:
                continue

            rs = _predict_tile_ultra(tile_im, model, device,
                                     conf_thr=conf_thr, iou_thr_tile=iou_thr_tile,
                                     max_det=max_det, scale=scale)
            for r in rs:
                x1t, y1t, x2t, y2t = r["bbox"]
                # 타일 -> 전체 이미지 좌표로 변환
                gx1 = x1t + x; gy1 = y1t + y
                gx2 = x2t + x; gy2 = y2t + y
                # 클램프
                gx1 = max(0, min(gx1, W)); gy1 = max(0, min(gy1, H))
                gx2 = max(0, min(gx2, W)); gy2 = max(0, min(gy2, H))
                if gx2 > gx1 and gy2 > gy1:
                    all_boxes.append({"bbox": [gx1, gy1, gx2, gy2],
                                      "conf": r["conf"], "cls": r["cls"]})

    if not all_boxes:
        return [], im
    return nms_global(all_boxes, iou_thr_global), im

# ----------------------------- SegFormer -----------------------------
@torch.no_grad()
def run_segformer(image_path: str, model, processor):
    from PIL import Image
    img_p = Path(image_path)
    image = Image.open(str(img_p)).convert("RGB")
    W,H = image.size
    inputs = processor(image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(next(model.parameters()).device)
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits
    up = torch.nn.functional.interpolate(logits, size=(H,W), mode="bilinear", align_corners=False)
    pred = up.argmax(dim=1)[0].detach().cpu().to(torch.uint8).numpy()
    return pred

# ----------------------------- 후처리/저장 -----------------------------
def overlay_and_save(im_bgr, mask, bboxes, out_path, mask_alpha=0.45,
                     mask_color=(0,255,0), box_color=(0,0,255)):
    if mask.shape[:2] != im_bgr.shape[:2]:
        mask = cv2.resize(mask, (im_bgr.shape[1], im_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    color = np.zeros_like(im_bgr)
    color[mask>0] = mask_color
    over = cv2.addWeighted(color, mask_alpha, im_bgr, 1.0-mask_alpha, 0)
    for r in bboxes:
        x1,y1,x2,y2 = r["bbox"]
        cv2.rectangle(over, (x1,y1), (x2,y2), box_color, 2)
        cv2.putText(over, f'{r["conf"]:.2f}', (x1, max(0,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 1, cv2.LINE_AA)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), over)

def check_overlap(mask, bboxes) -> bool:
    for r in bboxes:
        x1,y1,x2,y2 = r["bbox"]
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(mask.shape[1],x2), min(mask.shape[0],y2)
        if x2<=x1 or y2<=y1: continue
        if np.any(mask[y1:y2, x1:x2] > 0):
            return True
    return False

def save_txt(bboxes, txt_path):
    lines=[]
    for r in bboxes:
        x1,y1,x2,y2 = r["bbox"]; conf=r["conf"]; cls=r["cls"]
        lines.append(f"{x1} {y1} {x2} {y2} {conf:.6f} {cls}")
    Path(txt_path).parent.mkdir(parents=True, exist_ok=True)
    Path(txt_path).write_text("\n".join(lines), encoding="utf-8")

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # === 기본값만으로 실행 가능하도록 required 제거 ===
    ap.add_argument("--input", default="./test2.jpg")
    ap.add_argument("--savedir", default="./result_combine")
    ap.add_argument("--recursive", action="store_true", help="폴더 재귀 탐색")

    # Object Detection params (YOLOv11n + sliding)
    ap.add_argument("--od-weights", default="yolo11n.pt", help="Ultralytics YOLOv11 가중치(.pt)")
    ap.add_argument("--device", default="0", help="'cpu' 또는 '0','1' 등 GPU 인덱스/문자열")
    ap.add_argument("--tile", type=int, default=1280)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-tile", type=float, default=0.45)
    ap.add_argument("--iou-global", type=float, default=0.6)
    ap.add_argument("--max-det", type=int, default=3000, help="최대 박스 수(타일 단위)")
    ap.add_argument("--scale", type=float, default=1.0, help="타일 업샘플 배율(>1.0 권장: 초소형 객체)")

    # SegFormer params
    ap.add_argument("--seg-ckpt", default="./segformer/weights/epoch=26-val_epoch_miou=0.9809.ckpt")
    ap.add_argument("--mask-alpha", type=float, default=0.45)

    # 저장 옵션
    ap.add_argument("--save-txt", action="store_true")

    args = ap.parse_args()

    images = collect_images(args.input, recursive=args.recursive)
    if not images:
        raise FileNotFoundError(f"No images found under: {args.input}")
    print(f"[INFO] images: {len(images)}")

    # ==== Load YOLOv11n ====
    yolo_device = _parse_device_str(args.device)
    with _silent()[0], _silent()[1]:
        yolo = YOLO(args.od_weights)
        try:
            yolo.to(yolo_device)
        except Exception:
            pass

    # ==== Load SegFormer ====
    seg_torch_device = torch.device("cuda:0" if (yolo_device != "cpu" and torch.cuda.is_available()) else "cpu")
    seg_model = create_model().to(seg_torch_device).eval()
    ckpt = torch.load(args.seg_ckpt, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    cleaned = { (k[6:] if k.startswith("model.") else k): v for k,v in state.items() }
    seg_model.load_state_dict(cleaned, strict=False)
    processor = SegformerImageProcessor(do_resize=False)

    out_root = Path(args.savedir); out_root.mkdir(parents=True, exist_ok=True)

    for p in images:
        p = Path(p)
        dets, im = run_yolov11_sliding(
            str(p), yolo, yolo_device,
            tile=args.tile, overlap=args.overlap,
            conf_thr=args.conf, iou_thr_tile=args.iou_tile,
            iou_thr_global=args.iou_global, max_det=args.max_det,
            scale=args.scale
        )

        mask = run_segformer(str(p), seg_model, processor)

        if check_overlap(mask, dets):
            print(f"[위험] {p.name} : 마스크와 bbox 겹침")
        else:
            print(f"[OK] {p.name} : 겹침 없음")

        img_out = out_root / f"{p.stem}_od_seg.jpg"
        overlay_and_save(im, mask, dets, img_out, mask_alpha=args.mask_alpha)
        if args.save_txt:
            save_txt(dets, out_root / f"{p.stem}.txt")

        seg_png = out_root / f"{p.stem}.png"
        cv2.imwrite(str(seg_png), (mask*255).astype(np.uint8))

        print(f"[DONE] {p.name}  ->  {img_out}")

if __name__ == "__main__":
    main()
