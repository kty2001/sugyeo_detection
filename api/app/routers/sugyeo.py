import os
import signal
import json
from datetime import datetime

import cv2
import numpy as np
import onnxruntime as ort

from fastapi import APIRouter, Query
from starlette.responses import StreamingResponse
from starlette.requests import Request

from app.utils.model_utils import get_model_path

router = APIRouter()

latest_alert = False
frame_logs = []
streaming_status = {}

# ---------------- YOLO ONNX 초기화 ----------------
yolo_weight_path = get_model_path("yolo11n.onnx")
yolo_sess = ort.InferenceSession(yolo_weight_path, providers=["CPUExecutionProvider"])
# yolo_sess = ort.InferenceSession(yolo_weight_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# ---------------- SegFormer ONNX 초기화 ----------------
seg_weight_path = get_model_path("segformer-onnx/model.onnx")
seg_sess = ort.InferenceSession(seg_weight_path, providers=["CPUExecutionProvider"])
# seg_sess = ort.InferenceSession(seg_weight_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

seg_cfg_path = get_model_path("segformer-onnx/preprocessor_config.json")
with open(seg_cfg_path, "r") as f:
    cfg = json.load(f)

SEG_HEIGHT = cfg["size"]["height"]
SEG_WIDTH  = cfg["size"]["width"]
MEAN = np.array(cfg["image_mean"], dtype=np.float32)
STD  = np.array(cfg["image_std"], dtype=np.float32)
RESCALE = cfg.get("rescale_factor", 1/255.0)

# ---------------- Global IOU + NMS ----------------
def iou(a, b):
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

# ---------------- Tile Padding ----------------
def pad_tile(tile_img, stride=32):
    h, w = tile_img.shape[:2]
    pad_h = (stride - h % stride) % stride
    pad_w = (stride - w % stride) % stride
    if pad_h > 0 or pad_w > 0:
        tile_img = cv2.copyMakeBorder(tile_img, 0, pad_h, 0, pad_w,
                                      cv2.BORDER_CONSTANT, value=114)
    return tile_img, h, w  # 원래 타일 크기 반환

# ---------------- YOLO ONNX 타일 추론 ----------------
def _predict_tile_onnx(tile_img_bgr, session: ort.InferenceSession,
                       conf_thr=0.25, scale=1.0, stride=32):
    th, tw = tile_img_bgr.shape[:2]
    if scale != 1.0:
        tile_up = cv2.resize(tile_img_bgr, dsize=None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_LINEAR)
        inv_scale = 1.0/scale
    else:
        tile_up = tile_img_bgr
        inv_scale = 1.0

    tile_pad, orig_h, orig_w = pad_tile(tile_up, stride)
    img = cv2.cvtColor(tile_pad, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    img = np.transpose(img, (2,0,1))[np.newaxis, ...]

    outputs = session.run(None, {session.get_inputs()[0].name: img})
    print(np.array(outputs).shape)
    pred = outputs[0][0]  # (84, N)
    pred = pred.T         # (N, 84)

    out = []
    for row in pred:
        cx, cy, w, h = row[:4]
        class_confs = row[4:]
        cls = int(np.argmax(class_confs))
        conf = float(class_confs[cls])
        if conf < conf_thr or cls != 0:  # cls=0(person)만
            continue
        x1 = cx - w/2; y1 = cy - h/2
        x2 = cx + w/2; y2 = cy + h/2
        x1 *= inv_scale; x2 *= inv_scale
        y1 *= inv_scale; y2 *= inv_scale
        x1 = max(0, min(x1, orig_w)); x2 = max(0, min(x2, orig_w))
        y1 = max(0, min(y1, orig_h)); y2 = max(0, min(y2, orig_h))
        if x2 > x1 and y2 > y1:
            out.append({"bbox":[int(round(x1)), int(round(y1)),
                                int(round(x2)), int(round(y2))],
                        "conf": conf, "cls": cls})
    return out

# ---------------- Sliding Window ----------------
def run_yolov11_sliding_onnx(image: np.ndarray, session: ort.InferenceSession = yolo_sess,
                             tile=1280, overlap=0.5, conf_thr=0.25,
                             iou_thr_global=0.6, scale=1.0, stride=32):
    H, W = image.shape[:2]
    stride_tile = max(1, int(tile*(1-overlap)))
    all_boxes = []

    for i, y in enumerate(range(0, H, stride_tile)):
        for j, x in enumerate(range(0, W, stride_tile)):
            print(f"Processing tile row {i}, col {j} at (x,y)=({x},{y})")

            x2 = min(x+tile, W)
            y2 = min(y+tile, H)
            tile_im = image[y:y2, x:x2]
            th, tw = tile_im.shape[:2]
            if th==0 or tw==0: continue

            rs = _predict_tile_onnx(tile_im, session, conf_thr, scale, stride)
            for r in rs:
                x1t, y1t, x2t, y2t = r["bbox"]
                gx1 = x1t + x; gy1 = y1t + y
                gx2 = x2t + x; gy2 = y2t + y
                gx1 = max(0, min(gx1, W)); gy1 = max(0, min(gy1, H))
                gx2 = max(0, min(gx2, W)); gy2 = max(0, min(gy2, H))
                if gx2>gx1 and gy2>gy1:
                    all_boxes.append({"bbox":[gx1,gy1,gx2,gy2],
                                      "conf":r["conf"], "cls":r["cls"]})
    if not all_boxes:
        return []
    return nms_global(all_boxes, iou_thr_global)

# ------------------- Segformer -------------------
def run_segformer_onnx(image: np.ndarray):
    H, W = image.shape[:2]
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (SEG_WIDTH, SEG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))[np.newaxis, :, :, :]

    outputs = seg_sess.run(None, {seg_sess.get_inputs()[0].name: img})
    logits = outputs[0]

    _, C, h_pred, w_pred = logits.shape
    
    upsampled = np.zeros((C, SEG_HEIGHT, SEG_WIDTH), dtype=np.float32)
    for c in range(C):
        upsampled[c] = cv2.resize(logits[0, c], (SEG_WIDTH, SEG_HEIGHT), interpolation=cv2.INTER_LINEAR)

    mask = np.argmax(upsampled, axis=0).astype(np.uint8)
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return mask

# ---------------- Overlay -------------------
def overlay_yolo_and_seg(image: np.array, yolo_results: list, seg_mask: np.array):
    overlay = image.copy()
    mask_colored = np.zeros_like(image)
    mask_colored[seg_mask > 0] = [0, 255, 0]
    overlay = cv2.addWeighted(mask_colored, 0.6, overlay, 1.0, 0)

    for det in yolo_results:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["conf"]
        cls = det["cls"]

        if cls == 0:
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(seg_mask.shape[1], x2), min(seg_mask.shape[0], y2)
            overlap = np.any(seg_mask[y1c:y2c, x1c:x2c] > 0)

            color = (0, 0, 255) if overlap else (255, 0, 0)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, f'{conf:.2f}', (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    return overlay

def check_overlap(bboxes, mask) -> bool:
    for r in bboxes:
        x1, y1, x2, y2 = r["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(mask.shape[1], x2), min(mask.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        if np.any(mask[y1:y2, x1:x2] > 0):
            return True
    return False

def log_frame(dets, seg_mask):
    alert = check_overlap(dets, seg_mask)
    total_people = len(dets)
    overlap_count = sum(
        1 for r in dets if np.any(seg_mask[max(0,r["bbox"][1]):min(seg_mask.shape[0],r["bbox"][3]),
                                           max(0,r["bbox"][0]):min(seg_mask.shape[1],r["bbox"][2])] > 0)
    )
    non_overlap_count = total_people - overlap_count

    frame_logs.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "alert": alert,
        "total_people": total_people,
        "overlap_count": overlap_count,
        "non_overlap_count": non_overlap_count
    })
    return alert

def generate_frames(camera_id: int = 0):
    global latest_alert

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    frame_count = 0
    while streaming_status.get(camera_id, False):
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        if frame_count % 5 != 0:
            continue
        try:
            dets = run_yolov11_sliding_onnx(frame)
            seg_mask = run_segformer_onnx(frame)

            latest_alert = log_frame(dets, seg_mask)
            overlay_frame = overlay_yolo_and_seg(frame, dets, seg_mask)
            success, buffer = cv2.imencode('.jpg', overlay_frame)
            if success:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error processing frame: {e}")
            break

    cap.release()


for i in range(3):
    cap = cv2.VideoCapture(i)
    print(i, cap.isOpened())
    cap.release()

# ---------------- FastAPI ----------------
@router.get('/process')
async def video_feed(camera: int = Query(0)):
    if not streaming_status.get(camera, False):
        return {"message": "Streaming not started"}  # 혹은 204 No Content
    return StreamingResponse(generate_frames(camera), media_type='multipart/x-mixed-replace; boundary=frame')

@router.post("/start")
async def start_stream(camera: int):
    streaming_status[camera] = True
    return {"ok": True}

@router.post("/stop")
async def stop_stream(camera: int):
    streaming_status[camera] = False
    return {"ok": True}

@router.get("/check")
async def check_alert():
    global latest_alert
    return {"alert": latest_alert}

@router.get("/logs")
async def get_logs():
    return {"logs": list(reversed(frame_logs))}
