import os, time, hashlib
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

# ----------------- Small container -----------------
class Detection:
    def __init__(self, xyxy, score, class_id, class_name=''):
        x1, y1, x2, y2 = xyxy
        self.cx = float((x1 + x2) / 2.0)
        self.cy = float((y1 + y2) / 2.0)
        self.w_px = float(x2 - x1)
        self.h_px = float(y2 - y1)
        self.score = float(score)
        self.class_id = int(class_id)
        self.class_name = class_name

# ----------------- Base backend -----------------
class BaseBackend:
    def __init__(self, input_size=(640, 640), class_name='landing_plate'):
        self.input_size = tuple(input_size)
        self.class_name = class_name
        self.valid = False

    def warmup(self): pass
    def infer(self, img_bgr, stamp): return []

    def info(self):
        return {
            "backend": "base",
            "valid": False,
            "input_size": self.input_size,
            "expected_class": self.class_name
        }

# ----------------- Torch / Ultralytics backend -----------------
class TorchBackend(BaseBackend):
    """
    Fast YOLO backend with explicit perf knobs.
    - device: 'auto' | 'cpu' | 'cuda:0' (etc)
    - use_half: bool (only used on CUDA)
    - imgsz: inference size (int), e.g., 640
    - max_det: max detections per image
    - score_th: confidence threshold
    - iou_th: NMS IoU threshold
    """
    def __init__(self,
                 model_path,
                 class_name='landing_plate',
                 input_size=(640, 640),
                 device='cuda:0',       # default: force GPU
                 use_half=False,        # start safe (FP32); can set True later
                 imgsz=640,
                 max_det=3,
                 score_th=0.5,
                 iou_th=0.45):
        super().__init__(input_size=input_size, class_name=class_name)

        import os, torch, hashlib
        from pathlib import Path
        from ultralytics import YOLO

        torch.backends.cudnn.benchmark = True
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

        self.model_path = str(model_path or "")
        self.imgsz     = int(imgsz)
        self.max_det   = int(max_det)
        self.score_th  = float(score_th)
        self.iou_th    = float(iou_th)
        self.class_idx = None

        # ---- Choose device immediately (force GPU) ----
        if isinstance(device, str) and device:
            self.device_pref = device
        else:
            self.device_pref = 'cuda:0'
        if self.device_pref.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        self.device  = self.device_pref
        self.half_ok = bool(use_half) and self.device.startswith('cuda')

        # ---- Load model ----
        p = Path(self.model_path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Model file not found: {p}")
        try:
            self._model_sha12 = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
        except Exception:
            self._model_sha12 = ""

        self.model = YOLO(self.model_path)  # Ultralytics wrapper
        self.valid = True  # model exists/loaded

        # resolve class index
        names = getattr(self.model, "names", None)
        self.class_idx = None
        all_names = []
        if isinstance(names, dict):
            all_names = [names[k] for k in sorted(names)]
        elif isinstance(names, list):
            all_names = names[:]

        # case-insensitive match
        wanted = (self.class_name or "").strip().lower()
        idx = None
        for i, n in enumerate(all_names):
            if str(n).strip().lower() == wanted:
                idx = i; break

        self.class_idx = idx
        if self.class_idx is None:
            print(f"[TorchBackend] WARNING: class_name='{self.class_name}' not found in model names {all_names}. "
                f"Running with NO class filter to avoid dropping valid detections.")

        # Mark ready now (we have a device)
        print(f"[TorchBackend] init: device={self.device}, half={self.half_ok}, imgsz={self.imgsz}, class_idx={self.class_idx}, sha12={self._model_sha12}")

    def info(self):
        return {
            "backend": "torch",
            "valid": bool(self.valid),
            "ready": True,  # we picked a device in __init__
            "model_path": self.model_path,
            "model_sha256_12": getattr(self, "_model_sha12", ""),
            "device": str(self.device),
            "half": bool(self.half_ok),
            "imgsz": self.imgsz,
            "class_idx": self.class_idx,
            "input_size": self.input_size,
            "expected_class": self.class_name,
        }

    def _predict(self, img_bgr, dev, half_flag):
        # Single call wrapper so we can retry cleanly
        return self.model.predict(
            source=img_bgr,
            device=str(dev),
            imgsz=self.imgsz,
            half=bool(half_flag),
            conf=self.score_th,
            iou=self.iou_th,
            classes=[self.class_idx] if self.class_idx is not None else None,
            #max_det=self.max_det,
            agnostic_nms=False,
            verbose=False,
            #persist=True,
            stream=False
        )

    def infer(self, img_bgr, stamp):
        if not getattr(self, 'valid', False) or self.model is None:
            return []

        # Try current precision; if half fails with dtype error, flip to FP32 on CUDA and retry once.
        try:
            res = self._predict(img_bgr, self.device, self.half_ok)
        except Exception as e:
            msg = str(e)
            if self.device.startswith('cuda') and self.half_ok and ("Half" in msg or "dtype" in msg or "mat1 and mat2" in msg):
                # Auto fallback to FP32 on the same GPU
                self.half_ok = False
                print(f"[TorchBackend] FP16 failed on CUDA; retrying FP32. Error: {e}")
                res = self._predict(img_bgr, self.device, False)
            else:
                print(f"[TorchBackend] Inference failed: {e}")
                return []

        results = res if isinstance(res, list) else [res]
        dets = []
        for r in results:
            b = r.boxes
            if b is None or b.shape[0] == 0:
                continue
            xywh = b.xywh.cpu().numpy()
            conf = b.conf.cpu().numpy()
            cls  = b.cls.cpu().numpy().astype(int)
            names = getattr(self.model, "names", {})
            for (cx, cy, w, h), sc, ci in zip(xywh, conf, cls):
                x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
                name = names[int(ci)] if isinstance(names, dict) and int(ci) in names else str(ci)
                dets.append(Detection([x1, y1, x2, y2], float(sc), int(ci), name))
        return dets



# ----------------- factory -----------------
def make_backend(node):
    ns = 'perception.yolo_detector'
    backend    = node.get_parameter(f'{ns}.backend').value
    class_name = node.get_parameter(f'{ns}.class_name').value
    input_size = tuple(node.get_parameter(f'{ns}.input_size').value)
    if backend == 'torch':
        model_path = node.get_parameter(f'{ns}.model_path').value
        device     = node.get_parameter(f'{ns}.torch_device').value     # set to 'cuda:0'
        use_half   = bool(node.get_parameter(f'{ns}.torch_half').value) # start False, can True later
        imgsz      = int(node.get_parameter(f'{ns}.imgsz').value)
        max_det    = int(node.get_parameter(f'{ns}.max_det').value)
        score_th   = float(node.get_parameter(f'{ns}.score_thresh').value)
        iou_th     = float(node.get_parameter(f'{ns}.nms_thresh').value)
        return TorchBackend(model_path=model_path, class_name=class_name, input_size=input_size,
                            device=device, use_half=use_half, imgsz=imgsz, max_det=max_det,
                            score_th=score_th, iou_th=iou_th)
    return BaseBackend(class_name=class_name, input_size=input_size)
