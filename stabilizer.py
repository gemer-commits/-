"""
Стабилизация дрожащего видео на основе оптического потока.

Конвейер (по шагам):
  1. Предобработка кадра: серый + нормировка яркости + гауссово размытие [фильтрация]
  2. Поиск углов Ши–Томаси (goodFeaturesToTrack)                      [признаки]
  3. Пирамидальный оптический поток Лукаса–Канаде + проверка
     «вперёд-назад» (forward-backward), чтобы выбросить плохие точки   [оптический поток]
  4. Оценка движения камеры между кадрами: преобразование подобия
     (сдвиг + поворот + масштаб) методом RANSAC                       [робастная оценка]
  5. Накопление траектории камеры и её сглаживание                    [фильтрация траектории]
  6. Для каждого кадра — поправка «сглаженная траектория / реальная»
     и warpAffine, плюс автоматический zoom, чтобы убрать чёрные края [геометрия]

Все преобразования считаются в координатах с началом в ЦЕНТРЕ кадра:
так поворот не «разгоняет» сдвиг, и траектория получается честной.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict, replace

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Настройки
# --------------------------------------------------------------------------
@dataclass
class Config:
    method: str = "lk"          # "lk" (оптический поток) или "orb" (сопоставление признаков)
    max_corners: int = 300      # сколько углов искать
    min_distance: int = 15      # минимальное расстояние между углами, px
    quality: float = 0.01       # порог качества Ши–Томаси
    blur: int = 5               # размер гауссова ядра (0 = без размытия)
    clahe: bool = False         # выравнивание контраста CLAHE (опционально)
    normalize: bool = True      # нормировка яркости кадра (среднее/СКО) — защита от мерцания и засветки
    fallback_orb: bool = True   # если поток LK не сработал — запасной вариант: ORB
    fb_thresh: float = 1.0      # порог ошибки forward-backward, px
    ransac_thresh: float = 1.5  # порог RANSAC, px (больше — медленные объекты «просачиваются» в фон)
    min_inliers: int = 10       # меньше — считаем, что оценка не удалась
    smooth_radius: int = 15     # радиус сглаживания траектории (в кадрах)
    smooth_kind: str = "moving_average"   # или "gaussian"
    zoom: object = "auto"       # "auto" или число (например 1.08)
    max_zoom: float = 1.25      # ограничение авто-zoom

    def as_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------
# Геометрия
# --------------------------------------------------------------------------
def params_to_matrix(dx, dy, da, s=1.0):
    """(сдвиг, угол, масштаб) -> матрица 3x3 (в координатах от центра кадра)."""
    c, sn = s * np.cos(da), s * np.sin(da)
    return np.array([[c, -sn, dx], [sn, c, dy], [0, 0, 1.0]])


def matrix_to_params(A):
    """матрица 3x3 -> (dx, dy, da)."""
    return np.array([A[0, 2], A[1, 2], np.arctan2(A[1, 0], A[0, 0])])


def _translation(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1.0]])


# --------------------------------------------------------------------------
# Шаги 1–4: оценка движения между двумя кадрами
# --------------------------------------------------------------------------
def make_preprocessor(cfg: Config):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if cfg.clahe else None

    def prep(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if clahe is not None:
            gray = clahe.apply(gray)
        if cfg.blur:
            gray = cv2.GaussianBlur(gray, (cfg.blur, cfg.blur), 0)
        if cfg.normalize:
            # Метод Лукаса–Канаде предполагает «постоянство яркости». Резкое изменение
            # экспозиции его нарушает, поэтому приводим каждый кадр к одному среднему и контрасту.
            g = gray.astype(np.float32)
            gray = np.clip((g - g.mean()) / (g.std() + 1e-3) * 40 + 128, 0, 255).astype(np.uint8)
        return gray

    return prep


_LK = dict(winSize=(21, 21), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def match_lk(prev, curr, cfg: Config):
    """Углы на prev -> поток Лукаса–Канаде -> фильтр forward-backward. Возвращает пары точек."""
    pts = cv2.goodFeaturesToTrack(prev, maxCorners=cfg.max_corners, qualityLevel=cfg.quality,
                                  minDistance=cfg.min_distance, blockSize=3)
    if pts is None:
        return None, None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, curr, pts, None, **_LK)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(curr, prev, nxt, None, **_LK)
    fb_err = np.linalg.norm(pts - back, axis=2).ravel()
    good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb_err < cfg.fb_thresh)
    return pts[good].reshape(-1, 2), nxt[good].reshape(-1, 2)


_orb = cv2.ORB_create(nfeatures=1000)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def match_orb(prev, curr, cfg: Config):
    """Альтернатива для сравнения: ORB-признаки + brute-force сопоставление."""
    k0, d0 = _orb.detectAndCompute(prev, None)
    k1, d1 = _orb.detectAndCompute(curr, None)
    if d0 is None or d1 is None:
        return None, None
    matches = sorted(_bf.match(d0, d1), key=lambda m: m.distance)[:cfg.max_corners]
    if not matches:
        return None, None
    p0 = np.float32([k0[m.queryIdx].pt for m in matches])
    p1 = np.float32([k1[m.trainIdx].pt for m in matches])
    return p0, p1


def fit_similarity(p0, p1, center, cfg: Config):
    """RANSAC: подбираем преобразование подобия. Движущиеся объекты станут выбросами."""
    if p0 is None or len(p0) < cfg.min_inliers:
        return None, None
    m, inl = cv2.estimateAffinePartial2D(p0 - center, p1 - center, method=cv2.RANSAC,
                                         ransacReprojThreshold=cfg.ransac_thresh,
                                         maxIters=2000, confidence=0.99)
    if m is None or inl is None or int(inl.sum()) < cfg.min_inliers:
        return None, None
    A = np.vstack([m, [0, 0, 1.0]])
    return A, inl.ravel().astype(bool)


def track_pair(prev, curr, cfg: Config, center):
    """Полный шаг 2–4 для пары кадров. Возвращает (A, p0, p1, inliers) или (None, ...)."""
    matcher = match_lk if cfg.method == "lk" else match_orb
    p0, p1 = matcher(prev, curr, cfg)
    A, inl = fit_similarity(p0, p1, center, cfg)
    if A is None and cfg.method == "lk" and cfg.fallback_orb:
        p0, p1 = match_orb(prev, curr, cfg)          # запасной вариант
        A, inl = fit_similarity(p0, p1, center, cfg)
    return A, p0, p1, inl


# --------------------------------------------------------------------------
# Проход 1: оценка движения по всему видео
# --------------------------------------------------------------------------
def estimate_motions(path, cfg: Config, keep_debug=()):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видео: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center = np.array([W / 2, H / 2], np.float32)
    prep = make_preprocessor(cfg)

    T = [np.eye(3)]            # T[i]: кадр i-1 -> кадр i
    ok_flags, n_pts, n_inl, debug = [True], [0], [0], {}
    prev, i = None, 0
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = prep(frame)
        if prev is not None:
            A, p0, p1, inl = track_pair(prev, gray, cfg, center)
            if A is None:                     # запасной вариант: считаем, что движения нет
                T.append(np.eye(3)); ok_flags.append(False)
                n_pts.append(0 if p0 is None else len(p0)); n_inl.append(0)
            else:
                T.append(A); ok_flags.append(True)
                n_pts.append(len(p0)); n_inl.append(int(inl.sum()))
            if i in keep_debug:
                debug[i] = (p0, p1, inl)
        prev, i = gray, i + 1
    cap.release()
    return dict(T=T, ok=np.array(ok_flags), n_pts=np.array(n_pts), n_inliers=np.array(n_inl),
                size=(W, H), fps=fps, time=time.perf_counter() - t0, debug=debug)


# --------------------------------------------------------------------------
# Шаг 5: траектория и сглаживание
# --------------------------------------------------------------------------
def cumulative_path(T):
    """Накопленные матрицы A_i (кадр 0 -> кадр i) и их параметры (dx, dy, da)."""
    A, cur = [], np.eye(3)
    for t in T:
        cur = t @ cur
        A.append(cur.copy())
    P = np.array([matrix_to_params(a) for a in A])
    return A, P


def smooth_path(P, radius, kind="moving_average"):
    if radius <= 0:
        return P.copy()
    if kind == "gaussian":
        x = np.arange(-radius, radius + 1)
        k = np.exp(-0.5 * (x / (radius / 2)) ** 2)
    else:
        k = np.ones(2 * radius + 1)
    k = k / k.sum()
    pad = np.pad(P, ((radius, radius), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(P.shape[1])], axis=1)


# --------------------------------------------------------------------------
# Шаг 6: поправки для кадров
# --------------------------------------------------------------------------
def build_corrections(A, S, size, zoom="auto", max_zoom=1.25):
    """
    Для кадра i его «настоящее» положение — A_i, нужное (сглаженное) — S_i.
    Поправка W_i = S_i · A_i^-1 (в центрированных координатах), затем zoom.
    """
    W, H = size
    Ws = [params_to_matrix(*S[i]) @ np.linalg.inv(A[i]) for i in range(len(A))]

    # минимальный zoom, при котором углы выхода не вылезают за исходный кадр
    corners = np.array([[sx * W / 2, sy * H / 2, 1.0] for sx in (-1, 1) for sy in (-1, 1)]).T
    need = 1.0
    for w in Ws:
        v = np.linalg.inv(w) @ corners
        need = max(need, np.abs(v[0]).max() / (W / 2), np.abs(v[1]).max() / (H / 2))
    s = need * 1.005 if zoom == "auto" else float(zoom)
    capped = s > max_zoom
    s = min(s, max_zoom)

    Z = np.diag([s, s, 1.0])
    to_c, from_c = _translation(W / 2, H / 2), _translation(-W / 2, -H / 2)
    full = [to_c @ Z @ w @ from_c for w in Ws]
    return full, s, capped, need


# --------------------------------------------------------------------------
# Проход 2: рендер
# --------------------------------------------------------------------------
def render(path, out_path, full, fps, size):
    cap = cv2.VideoCapture(path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*("MJPG" if out_path.endswith(".avi") else "mp4v"))
    out = cv2.VideoWriter(out_path, fourcc, fps, size)
    if not out.isOpened():
        raise IOError(f"Не удалось создать выходной файл: {out_path}")
    t0 = time.perf_counter()
    for M in full:
        ok, frame = cap.read()
        if not ok:
            break
        out.write(cv2.warpAffine(frame, M[:2], size, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE))
    cap.release(); out.release()
    return time.perf_counter() - t0


def stabilize_video(in_path, out_path, cfg: Config | None = None, keep_debug=()):
    cfg = cfg or Config()
    est = estimate_motions(in_path, cfg, keep_debug)
    A, P = cumulative_path(est["T"])
    S = smooth_path(P, cfg.smooth_radius, cfg.smooth_kind)
    full, zoom, capped, need = build_corrections(A, S, est["size"], cfg.zoom, cfg.max_zoom)
    t_render = render(in_path, out_path, full, est["fps"], est["size"])
    n = len(full)
    return dict(est=est, A=A, P=P, S=S, full=full, zoom=zoom, zoom_needed=need, zoom_capped=capped,
                n_frames=n, t_estimate=est["time"], t_render=t_render,
                fps_total=n / (est["time"] + t_render), cfg=cfg)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Стабилизация видео (оптический поток Лукаса–Канаде)")
    ap.add_argument("input"); ap.add_argument("output")
    ap.add_argument("--method", default="lk", choices=["lk", "orb"])
    ap.add_argument("--radius", type=int, default=15, help="радиус сглаживания, кадры")
    ap.add_argument("--clahe", action="store_true")
    a = ap.parse_args()
    r = stabilize_video(a.input, a.output, Config(method=a.method, smooth_radius=a.radius, clahe=a.clahe))
    print(f"Готово: {r['n_frames']} кадров, zoom={r['zoom']:.3f}"
          f"{' (ограничен!)' if r['zoom_capped'] else ''}, {r['fps_total']:.1f} к/с")
    print(f"Кадров с неудачной оценкой движения: {int((~r['est']['ok']).sum())}")
