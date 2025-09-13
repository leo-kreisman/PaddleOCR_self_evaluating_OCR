# adaptive_paddleocr.py — v8 (doc-aligned)

from __future__ import annotations
from pathlib import Path
from typing import Any, List, Tuple, Dict
import json, time
import cv2
import numpy as np
from paddleocr import PaddleOCR

# ── helpers ───────────────────────────────────────────────────────────────────

def _geom_to_xyxy(g):
    """Aceita [x1,y1,x2,y2] ou polígono e devolve (x1,y1,x2,y2) int."""
    g = _tolist(g)
    if isinstance(g, (list, tuple)) and len(g) == 4 and all(isinstance(v,(int,float)) for v in g):
        x1,y1,x2,y2 = g
        return int(x1), int(y1), int(x2), int(y2)
    # polígono
    xs = [int(p[0]) for p in g]; ys = [int(p[1]) for p in g]
    return min(xs), min(ys), max(xs), max(ys)

def _safe_crop(img_bgr, xyxy, pad=2):
    h, w = img_bgr.shape[:2]
    x1,y1,x2,y2 = xyxy
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w-1, x2 + pad); y2 = min(h-1, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img_bgr[y1:y2, x1:x2].copy()

def _first_geoms_from_rec(rec):
    """Escolhe a melhor lista de geometrias disponível em rec."""
    if not isinstance(rec, dict):
        return []
    wboxes = _get_first(rec, "rec_word_boxes", "word_boxes", "rec_wordbox", "wordbox")
    if _is_nonempty(wboxes):
        return wboxes
    polys = rec.get("rec_polys")
    if _is_nonempty(polys):
        return polys
    boxes = rec.get("rec_boxes")
    if _is_nonempty(boxes):
        return boxes
    return []

# ── enhancement helpers (OpenCV) ─────────────────────────────────────────────

def _export_line_crops(orig_bgr: np.ndarray, rec: dict, out_dir: Path, ts: str):
    """
    Salva recortes por linha + versões pré-processadas.
    NÃO altera PNG/JSON principais já gerados.
    """
    geoms = _first_geoms_from_rec(rec)
    if not _is_nonempty(geoms):
        return

    crop_root = out_dir / "crops" / ts
    crop_root.mkdir(parents=True, exist_ok=True)

    # quais modos gerar para cada recorte
    crop_modes = ("raw", "hires", "hires_bin", "dpi_norm")

    for i, g in enumerate(geoms, start=1):
        try:
            xyxy = _geom_to_xyxy(g)
            crop = _safe_crop(orig_bgr, xyxy, pad=4)
            if crop is None:
                continue

            # salva RAW
            raw_path = crop_root / f"line_{i:03d}_raw.png"
            cv2.imwrite(str(raw_path), crop)

            # versões melhoradas
            for mode in crop_modes[1:]:
                proc = preprocess(crop, mode)
                outp = crop_root / f"line_{i:03d}_{mode}.png"
                cv2.imwrite(str(outp), proc)
        except Exception as e:
            # falha pontual num recorte não deve abortar todo o processo
            # (silencioso para não poluir a saída)
            _ = e

def _white_balance_lab(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L = cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX)
    A = cv2.addWeighted(A, 0.6, np.full_like(A, 128), 0.4, 0)
    B = cv2.addWeighted(B, 0.6, np.full_like(B, 128), 0.4, 0)
    lab = cv2.merge([L, A, B])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def _clahe_gray(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

def _unsharp(img: np.ndarray, sigma=0.8, amount=0.6) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0,0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(img, 1+amount, blur, -amount, 0)

def _bilateral(img: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)

def _upscale(img: np.ndarray, fx=2.0, fy=2.0) -> np.ndarray:
    return cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

def _deskew_binary(bin_img: np.ndarray):
    edges = cv2.Canny(bin_img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                            minLineLength=bin_img.shape[1]//3, maxLineGap=10)
    angle = 0.0
    if lines is not None and len(lines) > 0:
        angles = []
        for l in lines[:200]:
            x1,y1,x2,y2 = l[0]
            a = np.degrees(np.arctan2(y2-y1, x2-x1))
            if -45 < a < 45:
                angles.append(a)
        if angles:
            angle = float(np.median(angles))
    (h,w) = bin_img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(bin_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), angle

def _deskew_rgb(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if np.mean(th) < 127:
        th = 255 - th
    _, ang = _deskew_binary(th)
    rot = cv2.getRotationMatrix2D((img_bgr.shape[1]/2, img_bgr.shape[0]/2), ang, 1.0)
    return cv2.warpAffine(img_bgr, rot, (img_bgr.shape[1], img_bgr.shape[0]),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def _normalize_for_rec(img_bgr: np.ndarray, target_h=48, scale=1.4) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    new_h = int(target_h * scale)
    new_w = int(round(w * (new_h / max(1, h))))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

def _autocrop_with_polys(img_bgr: np.ndarray, rec_any, pad=20):
    rec_polys, rec_boxes, rec_word_boxes = _pick_boxes(rec_any)
    pts = []
    for coll in (rec_word_boxes, rec_polys, rec_boxes):
        if _is_nonempty(coll):
            for g in coll:
                g = _tolist(g)
                if (isinstance(g, (list, tuple)) and len(g) == 4
                        and all(isinstance(v,(int,float)) for v in g)):
                    x1,y1,x2,y2 = map(int, g)
                    pts.extend([(x1,y1),(x2,y2)])
                else:
                    for p in g:
                        pts.append((int(p[0]), int(p[1])))
            break
    if not pts:
        return img_bgr
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x1,y1,x2,y2 = (max(0, min(xs)-pad), max(0, min(ys)-pad),
                   min(img_bgr.shape[1]-1, max(xs)+pad), min(img_bgr.shape[0]-1, max(ys)+pad))
    return img_bgr[y1:y2, x1:x2].copy()


def flatten_result(res_any) -> List[str]:
    texts: List[str] = []
    if hasattr(res_any, "rec_texts"):
        texts.extend(res_any.rec_texts)
    elif isinstance(res_any, list):
        for item in res_any:
            texts.extend(flatten_result(item))
    return texts

# --- PATCH 1: substitua flatten_result e texts_from_predict por isto:

def _texts_from_any(res) -> list[str]:
    """Extrai rec_texts de objetos PaddleOCR, listas OU dicts."""
    out = []
    if res is None:
        return out

    # 1) dict no formato dos seus JSONs: { rec_texts, rec_scores, rec_polys, ... }
    if isinstance(res, dict):
        if "rec_texts" in res and isinstance(res["rec_texts"], list):
            out.extend([str(t) for t in res["rec_texts"]])
        return out

    # 2) objeto com atributos (PipelineResult)
    if hasattr(res, "rec_texts"):
        try:
            out.extend(list(res.rec_texts))
        except Exception:
            pass

    # 3) lista (de itens que podem ser dict ou objeto)
    if isinstance(res, list):
        for item in res:
            out.extend(_texts_from_any(item))
    return out

def texts_from_predict(result) -> str:
    texts = _texts_from_any(result)
    return "\n".join(texts)

def score(text: str) -> Tuple[int, float, int]:
    L = len(text)
    A = sum(c.isalpha() for c in text) / max(1, L)
    N = text.count("\n") + 1
    return L, A, N

def preprocess(img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "binarize":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    if mode == "invert":
        return 255 - img
    if mode == "upscale":
        return _upscale(img, 2.0, 2.0)

    # focados em linhas finas/diacríticos
    if mode == "hires":
        x = _white_balance_lab(img)
        x = _clahe_gray(x)
        x = _bilateral(x)
        x = _unsharp(x, sigma=0.8, amount=0.6)
        x = _upscale(x, 2.0, 2.0)
        return x

    if mode == "hires_bin":
        x = preprocess(img, "hires")
        gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 7)
        return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

    if mode == "deskew":
        return _deskew_rgb(img)

    if mode == "dpi_norm":
        return _normalize_for_rec(img, target_h=48, scale=1.4)

    return img

# ── build/init ────────────────────────────────────────────────────────────────

def build_ocr(args) -> PaddleOCR:
    """
    Alinhado à doc v3.x:
      - Passe lang/ocr_version sempre.
      - Só passe *_model_name se o usuário explicitou (caso contrário deixe a
        pipeline escolher pelos 'lang/ocr_version').
      - NÃO passe char_dict via kwargs (não suportado no init desta pipeline).
    """
    kwargs: Dict[str, Any] = dict(
        lang=args.lang,
        ocr_version="PP-OCRv5",
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,        # default True na doc; aqui mantemos False p/ ganho de velocidade
        use_textline_orientation=True,  # idem
        return_word_box=True
    )
    if getattr(args, "det_model", None):
        kwargs["text_detection_model_name"] = args.det_model
    if getattr(args, "rec_model", None):
        kwargs["text_recognition_model_name"] = args.rec_model
    return PaddleOCR(**kwargs)

def _draw_overlays(img_bgr, rec_polys=None, rec_boxes=None, rec_word_boxes=None):
    vis = img_bgr.copy()

    # 1) polígonos têm prioridade
    if _is_nonempty(rec_polys):
        for poly in rec_polys:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], True, (0, 0, 255), 2)

    # 2) se NÃO houver polígonos, desenha boxes
    elif _is_nonempty(rec_boxes):
        for box in rec_boxes:
            box = np.array(box).ravel().tolist()
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 3) word boxes sempre
    if _is_nonempty(rec_word_boxes):
        for wb in rec_word_boxes:
            wb = np.array(wb)
            flat = wb.ravel()
            if flat.size == 4:
                x1, y1, x2, y2 = map(int, flat[:4])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
            else:
                pts = wb.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [pts], True, (0, 255, 0), 1)
    return vis



# def _save_outputs(res_any, *, original_img_path: Path | None = None):
#     out_dir = Path("output"); out_dir.mkdir(exist_ok=True)
#     ts = time.strftime("%Y%m%d-%H%M%S")

#     # 1) Se o objeto tem os métodos nativos, use-os
#     objs = res_any if isinstance(res_any, list) else [res_any]
#     saved_native = False
#     for obj in objs:
#         if hasattr(obj, "save_to_img"):
#             obj.save_to_img(str(out_dir))
#             obj.save_to_json(str(out_dir))
#             saved_native = True

#     # 2) Caso contrário, se vier como dict, salve JSON e gere overlay
#     if not saved_native:
#         # Salva JSON bruto
#         json_path = out_dir / f"ocr_res_{ts}.json"
#         try:
#             with open(json_path, "w", encoding="utf-8") as f:
#                 json.dump(res_any, f, ensure_ascii=False, indent=2)
#         except TypeError:
#             # se for lista de itens não serializáveis, faça um best-effort
#             serializable = []
#             for item in objs:
#                 if isinstance(item, dict):
#                     serializable.append(item)
#             with open(json_path, "w", encoding="utf-8") as f:
#                 json.dump(serializable, f, ensure_ascii=False, indent=2)

#         # Gera overlay simples se tivermos imagem original + polys/boxes
#         if isinstance(res_any, dict) and original_img_path and Path(original_img_path).exists():
#             img = cv2.imread(str(original_img_path))
#             if img is not None:
#                 rec_polys = res_any.get("rec_polys")
#                 rec_boxes = res_any.get("rec_boxes")
#                 # ⟵ NOVO: tente ambas as variantes de chave de word-box
#                 rec_word_boxes = (res_any.get("rec_word_boxes")
#                                 or res_any.get("word_boxes")
#                                 or res_any.get("rec_wordbox")
#                                 or res_any.get("wordbox"))
#                 vis = _draw_overlays(img,
#                                     rec_polys=rec_polys,
#                                     rec_boxes=rec_boxes,
#                                     rec_word_boxes=rec_word_boxes)
#                 cv2.imwrite(str(out_dir / f"ocr_res_{ts}.png"), vis)

#     print(f"💾 outputs dir: {out_dir.resolve()}")

def _obj_to_dict(item):
    if isinstance(item, dict):
        return item
    d = {}
    for k in ("rec_texts","rec_scores","rec_polys","rec_boxes","rec_word_boxes","word_boxes"):
        v = getattr(item, k, None)
        if v is not None:
            d[k] = v
    return d or None

def _to_serializable(obj):
    import numpy as np
    # Tipos nativos primeiro
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    # NumPy escalares
    if isinstance(obj, (np.integer,)):  # noqa: F821
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    # Sequências
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    # NumPy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Dict
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}

    # Qualquer outro objeto (ex.: Font, Path, etc.) -> string
    try:
        return str(obj)
    except Exception:
        return repr(obj)

def _is_nonempty(seq):
    if seq is None:
        return False
    try:
        return len(seq) > 0
    except Exception:
        # fallback para objetos que não têm len()
        return True

def _get_first(res: dict, *keys):
    """Retorna o primeiro valor NÃO-None para as chaves, sem usar OR em arrays."""
    for k in keys:
        if k in res and res[k] is not None:
            return res[k]
    return None

def _tolist(x):
    return np.array(x).tolist() if isinstance(x, np.ndarray) else x

def _pick_boxes(rec):
    if not rec:
        return None, None, None
    polys  = rec.get("rec_polys", None)
    boxes  = rec.get("rec_boxes", None)
    wboxes = _get_first(rec, "rec_word_boxes", "word_boxes", "rec_wordbox", "wordbox")
    return polys, boxes, wboxes

def _compact_payload(res_any):
    """
    Converte resultados PaddleOCR em estrutura serializável:
    [
      {"text": str, "score": float|None, "poly": [[x,y]...]|None,
       "box": [x1,y1,x2,y2]|None, "word_box": [[x,y]...]/[x1,y1,x2,y2]|None}
    ]
    """
    items = []

    if res_any is None:
        return items

    # lista: processa cada item
    if isinstance(res_any, list):
        for r in res_any:
            items.extend(_compact_payload(r))
        return items

    # dict
    if isinstance(res_any, dict):
        texts  = res_any.get("rec_texts", [])
        scores = res_any.get("rec_scores", [])
        polys  = res_any.get("rec_polys", [])
        boxes  = res_any.get("rec_boxes", [])

        wboxes = _get_first(res_any, "rec_word_boxes", "word_boxes", "rec_wordbox", "wordbox")
        if wboxes is None:
            wboxes = []

        texts  = list(map(str, texts))
        scores = [float(s) for s in scores] if len(scores) == len(texts) else [None]*len(texts)

        polys  = [_tolist(p) for p in polys]
        boxes  = [_tolist(b) for b in boxes]
        wboxes = [_tolist(w) for w in wboxes]

        L = len(texts)
        for i in range(L):
            items.append({
                "text": texts[i],
                "score": scores[i] if i < len(scores) else None,
                "poly": polys[i] if i < len(polys) else None,
                "box":  boxes[i] if i < len(boxes) else None,
                "word_box": wboxes[i] if i < len(wboxes) else None,
            })
        return items

    # objeto (PipelineResult)
    if hasattr(res_any, "rec_texts"):
        try:
            texts  = list(res_any.rec_texts)
            scores = list(res_any.rec_scores)
        except Exception:
            return items
        polys_attr = getattr(res_any, "rec_polys", [])
        boxes_attr = getattr(res_any, "rec_boxes", [])
        polys  = [_tolist(p) for p in polys_attr]
        boxes  = [_tolist(b) for b in boxes_attr]
        L = len(texts)
        for i in range(L):
            items.append({
                "text": str(texts[i]),
                "score": float(scores[i]) if i < len(scores) else None,
                "poly": polys[i] if i < len(polys) else None,
                "box":  boxes[i] if i < len(boxes) else None,
                "word_box": None,
            })
        return items

    return items

# ── reconstrução de espaços a partir de word boxes ───────────────────────────

def _to_rect(g):
    g = _tolist(g)
    if isinstance(g, (list, tuple)) and len(g) == 4 and all(isinstance(v,(int,float)) for v in g):
        x1,y1,x2,y2 = map(int, g)
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        return [x1,y1,x2,y2]
    xs = [int(p[0]) for p in g]
    ys = [int(p[1]) for p in g]
    return [min(xs), min(ys), max(xs), max(ys)]

def _items_from_result(res_any):
    if isinstance(res_any, dict):
        texts  = res_any.get("rec_texts")  or []
        wboxes = _get_first(res_any, "rec_word_boxes","word_boxes","rec_wordbox","wordbox")
        geoms  = wboxes if _is_nonempty(wboxes) else (
                 res_any.get("rec_polys") if _is_nonempty(res_any.get("rec_polys")) else
                 res_any.get("rec_boxes"))
        if not geoms: return []
        out=[]; L=min(len(texts), len(geoms))
        for i in range(L):
            out.append((str(texts[i]), _to_rect(geoms[i])))
        return out
    if hasattr(res_any, "rec_texts"):
        texts = list(getattr(res_any, "rec_texts", []))
        geoms = getattr(res_any, "rec_word_boxes", None)
        if not _is_nonempty(geoms):
            geoms = getattr(res_any, "word_boxes", None)
        if not _is_nonempty(geoms):
            geoms = getattr(res_any, "rec_polys", None)
        if not _is_nonempty(geoms):
            geoms = getattr(res_any, "rec_boxes", None)
        if not _is_nonempty(geoms): return []
        out=[]; L=min(len(texts), len(geoms))
        for i in range(L):
            out.append((str(texts[i]), _to_rect(geoms[i])))
        return out
    items=[]
    if isinstance(res_any, list):
        for it in res_any:
            items.extend(_items_from_result(it))
    return items

def recover_spaces_from_word_boxes(res_any, *, gap_ratio=0.5, min_space_px=8, y_tol_factor=0.6):
    items = _items_from_result(res_any)
    if not items: return ""
    heights = [r[3]-r[1] for _,r in items]
    med_h = float(np.median(heights)) if heights else 0.0
    y_tol = max(8.0, y_tol_factor * med_h)

    def yc(r): return 0.5*(r[1]+r[3])
    items.sort(key=lambda it: yc(it[1]))
    lines=[]; cur=[]; last=None
    for t,r in items:
        cy=yc(r)
        if last is None or abs(cy-last)<=y_tol: cur.append((t,r))
        else: lines.append(cur); cur=[(t,r)]
        last=cy
    if cur: lines.append(cur)

    out=[]
    for line in lines:
        line.sort(key=lambda it: it[1][0])
        parts=[]; prev_t=None; prev_r=None
        lw = [rr[1][2]-rr[1][0] for rr in line]
        med_w=float(np.median(lw)) if lw else 0.0
        for t,r in line:
            if prev_r is None:
                parts.append(t)
            else:
                gap = r[0]-prev_r[2]
                w_prev = prev_r[2]-prev_r[0]
                w_cur  = r[2]-r[0]
                char_w_prev = (w_prev/max(1,len(prev_t))) if prev_t else w_prev
                char_w_cur  = (w_cur /max(1,len(t)))      if t else w_cur
                char_w = 0.5*(char_w_prev+char_w_cur)
                thr = max(min_space_px, gap_ratio*max(w_prev,w_cur), 1.4*char_w, 0.35*med_w)
                if gap > thr and not (prev_t and prev_t.endswith("-")):
                    parts.append(" ")
                if prev_t and prev_t.endswith("-") and gap <= thr:
                    parts[-1] = parts[-1][:-1]
                parts.append(t)
            prev_t, prev_r = t, r
        out.append("".join(parts))
    return "\n".join(out)

def _make_dualpage_viz(orig_bgr, rec):
    H, W = orig_bgr.shape[:2]
    rec_polys, rec_boxes, rec_word_boxes = _pick_boxes(rec)
    left = _draw_overlays(orig_bgr, rec_polys, rec_boxes, rec_word_boxes)

    right = np.full((H, W, 3), 255, np.uint8)
    texts  = rec.get("rec_texts", [])
    # escolha segura da geometria
    geoms = None
    if _is_nonempty(rec_word_boxes):
        geoms = rec_word_boxes
    elif _is_nonempty(rec_polys):
        geoms = rec_polys
    elif _is_nonempty(rec_boxes):
        geoms = rec_boxes
    else:
        geoms = []

    L = min(len(texts), len(geoms))
    for i in range(L):
        t = str(texts[i])
        g = geoms[i]
        g = _tolist(g)
        if isinstance(g, (list, tuple)) and len(g) == 4 and all(isinstance(v,(int,float)) for v in g):
            x1,y1,x2,y2 = map(int, g)
        else:
            # poly -> retângulo
            xs=[int(p[0]) for p in g]; ys=[int(p[1]) for p in g]
            x1,y1,x2,y2 = min(xs), min(ys), max(xs), max(ys)
        cv2.rectangle(right, (x1,y1), (x2,y2), (0,0,0), 1)
        tx, ty = x1+3, max(y1+12, 12)
        cv2.putText(right, t, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)

    canvas = np.full((H, W*2, 3), 255, np.uint8)
    canvas[:, :W]  = left
    canvas[:, W: ] = right
    return canvas

def _save_outputs(res_any, *, original_img_path: Path | None = None):
    out_dir = Path("output"); out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")

    # 1) Tenta caminhos nativos, se existirem
    objs = res_any if isinstance(res_any, list) else [res_any]
    saved_native = False
    for obj in objs:
        if hasattr(obj, "save_to_img"):
            obj.save_to_img(str(out_dir))
            obj.save_to_json(str(out_dir))
            saved_native = True

    # 2) Sempre salva JSON “serializável”
    serializable = []
    for item in objs:
        d = _obj_to_dict(item)
        if d:
            serializable.append(d)
        elif isinstance(item, dict):
            serializable.append(item)
    # se nada deu, ainda assim escreva algo informativo

    compact = _compact_payload(res_any)
    json_path = out_dir / f"ocr_res_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, indent=2)

    # 3) Overlay: use o primeiro item com caixas/polys
    if original_img_path and original_img_path.exists():
        img = cv2.imread(str(original_img_path))
        if img is not None:
            # usa o mesmo “rec0” que você já escolhe para overlay
            rec0 = None
            for it in serializable:
                if any(k in it for k in ("rec_polys","rec_boxes","rec_word_boxes","word_boxes","rec_wordbox","wordbox")):
                    rec0 = it; break
            if rec0:
                # single overlay (como já fazia):
                vis = _draw_overlays(img,
                                     rec_polys=rec0.get("rec_polys"),
                                     rec_boxes=rec0.get("rec_boxes"),
                                     rec_word_boxes=(rec0.get("rec_word_boxes") or rec0.get("word_boxes")))
                cv2.imwrite(str(out_dir / f"ocr_res_{ts}.png"), vis)

                # NOVO: “duas páginas”:
                dual = _make_dualpage_viz(img, rec0)
                cv2.imwrite(str(out_dir / f"ocr_res_{ts}_dual.png"), dual)

                # NOVO: recortes por linha + versões melhoradas
                _export_line_crops(img, rec0, out_dir, ts)

    print(f"💾 outputs dir: {out_dir.resolve()}")
    print(f"   - JSON: {json_path.name}")
    pngs = list(out_dir.glob(f"ocr_res_{ts}.png"))
    if pngs:
        print(f"   - PNG : {pngs[0].name}")


# ── OCR routines ─────────────────────────────────────────────────────────────

def _predict_with_flags(ocr: PaddleOCR, img_any, *, args) -> Any:
    """
    Repassa os kwargs suportados pela pipeline ao *predict*, como na doc:
      - text_rec_score_thresh (default 0.0 = sem corte)
      - use_doc_unwarping / use_textline_orientation (podem ser None/False/True)
    """
    return ocr.predict(
        img_any,
        text_rec_score_thresh=args.text_rec_score_thresh,
        use_doc_unwarping=args.use_doc_unwarping,
        use_textline_orientation=args.use_textline_orientation,
        return_word_box=getattr(args, "return_word_box", True),
    )

def simple_ocr(ocr: PaddleOCR, img: Path, save: bool, *, args) -> str:
    res = _predict_with_flags(ocr, str(img), args=args)
    if save:
        _save_outputs(res, original_img_path=img)
    return texts_from_predict(res)

def adaptive_ocr(ocr: PaddleOCR, img: Path, save: bool, *, args):
    base = cv2.imread(str(img))
    if base is None:
        raise FileNotFoundError(img)

    chosen_txt, chosen_step, chosen_res = "", "normal", None

    # 1ª rodada: mais modos de pré-processamento
    for step in ("normal", "hires", "hires_bin", "binarize", "invert", "upscale", "deskew", "dpi_norm"):
        img_pp = preprocess(base, step) if step != "normal" else base
        res = _predict_with_flags(ocr, img_pp, args=args)
        txt = texts_from_predict(res)
        L = len(txt)
        A = (sum(c.isalpha() for c in txt) / max(1, L)) if L else 0.0
        N = txt.count("\n") + (1 if L else 0)
        print(f"[{step:<8}] len={L:4d} α={A:.2f} lines={N}")

        if L >= 30 and A >= 0.5 and N >= 2:
            chosen_txt, chosen_step, chosen_res = txt, step, res
            break
        if L > len(chosen_txt):
            chosen_txt, chosen_step, chosen_res = txt, step, res

    # 2ª rodada opcional: se temos rec_polys/boxes (dict ou objeto), faz autocrop + hires/dpi_norm
    cand_dict = None
    try:
        cand_dict = chosen_res if isinstance(chosen_res, dict) else _obj_to_dict(chosen_res)
    except Exception:
        cand_dict = None
    if cand_dict:
        cropped = _autocrop_with_polys(base, cand_dict, pad=24)
        for step in ("hires", "dpi_norm", "hires_bin"):
            img2 = preprocess(cropped, step)
            res2 = _predict_with_flags(ocr, img2, args=args)
            txt2 = texts_from_predict(res2)
            if len(txt2) > len(chosen_txt):
                chosen_txt, chosen_step, chosen_res = txt2, f"crop+{step}", res2
                cand_dict = res2 if isinstance(res2, dict) else _obj_to_dict(res2)

    # 3) tenta reconstruir espaços via word_boxes (se houver)
    if cand_dict:
        rebuilt = recover_spaces_from_word_boxes(cand_dict, gap_ratio=0.55, min_space_px=10)
        if rebuilt and len(rebuilt) >= len(chosen_txt) * 0.9:
            chosen_txt = rebuilt

    # 4) SALVA uma vez, com overlay/dual usando a imagem PRÉ-PROCESSADA do step vencedor
    if save and chosen_res is not None:
        out_dir = Path("output"); out_dir.mkdir(exist_ok=True)
        viz_path = img
        if chosen_step != "normal":
            # extrai o step final (após crop+)
            step_final = chosen_step.split("+")[-1]
            base_viz = preprocess(base, step_final)
            viz_path = out_dir / f"viz_{Path(img).stem}_{chosen_step}.png"
            cv2.imwrite(str(viz_path), base_viz)
        _save_outputs(chosen_res, original_img_path=viz_path)

    return chosen_txt, chosen_step
