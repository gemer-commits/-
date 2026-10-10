"""
Количественная оценка: прогон всех сценариев, сравнение «оригинал / LK / ORB»,
таблицы в results/metrics.csv и results/ablation.csv.

Метрики:
  ITF        — средний PSNR между соседними кадрами, дБ (чем больше, тем стабильнее картинка).
               Не зависит от нашего алгоритма оценки движения.
  Shake      — остаточное «дрожание» в готовом видео: RMS отклонения траектории камеры от
               ИДЕАЛЬНОЙ (без дрожания), px. Траектория пере-оценивается по самому видео
               (делим на zoom, т.к. zoom увеличивает сдвиги).
  HF-jitter  — высокочастотное дрожание (траектория минус её сглаженная версия, окно ±5 кадров).
               Не требует истины — годится для реальных видео.
  Err_track  — точность оценки движения камеры против истины (px, RMS без константы).
  Err_plan   — насколько итоговая плавная траектория отличается от «желаемой» (px, RMS).
  Oracle     — тот же показатель при ИДЕАЛЬНОЙ оценке движения (потолок для выбранного окна).
  FPS        — скорость обработки целиком (оценка + рендер).
"""
import csv
import json
import os

import cv2
import numpy as np

import stabilizer as st
from make_test_videos import SCENARIOS, W, H

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, RES = os.path.join(HERE, "data"), os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)


# ------------------------------- метрики -----------------------------------
def itf(path, crop=0.1):
    """Среднее PSNR соседних кадров по центральной области (края после zoom не считаем)."""
    cap = cv2.VideoCapture(path)
    prev, vals = None, []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = g.shape
        g = g[int(h * crop):int(h * (1 - crop)), int(w * crop):int(w * (1 - crop))]
        if prev is not None:
            mse = max(float(np.mean((g - prev) ** 2)), 1e-3)
            vals.append(10 * np.log10(255 ** 2 / mse))
        prev = g
    cap.release()
    return float(np.mean(vals))


def jitter_rms(P, radius=5):
    """Высокочастотная часть траектории: (px по x/y, градусы по углу)."""
    hp = P - st.smooth_path(P, radius)
    return float(np.sqrt(np.mean(hp[:, 0] ** 2 + hp[:, 1] ** 2))), float(np.degrees(np.sqrt(np.mean(hp[:, 2] ** 2))))


def gt_params(poses, ref_pose):
    """Истинные параметры (dx,dy,da) «кадр 0 -> кадр i» в центрированных координатах."""
    R = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    C0, a0 = ref_pose[:2], ref_pose[2]
    out = []
    for cx, cy, a in poses:
        t = R(a).T @ (C0 - np.array([cx, cy]))
        out.append([t[0], t[1], a0 - a])
    return np.array(out)


def rms_no_const(err):
    e = err - err.mean(0)
    return float(np.sqrt(np.mean(e[:, 0] ** 2 + e[:, 1] ** 2))), float(np.degrees(np.sqrt(np.mean(e[:, 2] ** 2))))


# ------------------------------- прогон ------------------------------------
def run_scenario(name, cfg, tag):
    src = os.path.join(DATA, f"{name}.avi")
    dst = os.path.join(RES, f"{name}_{tag}.avi")
    r = st.stabilize_video(src, dst, cfg)
    gt = np.load(os.path.join(DATA, f"{name}_gt.npz"))
    n = r["n_frames"]
    P_true = gt_params(gt["real"][:n], gt["real"][0])
    P_ideal = gt_params(gt["smooth"][:n], gt["real"][0])
    e_track = rms_no_const(r["P"] - P_true)
    e_plan = rms_no_const(r["S"] - P_ideal)
    # «потолок»: что получилось бы при ИДЕАЛЬНОЙ оценке движения с тем же сглаживанием
    oracle = rms_no_const(st.smooth_path(P_true, cfg.smooth_radius, cfg.smooth_kind) - P_ideal)
    # пере-оценка траектории по стабилизированному видео
    est2 = st.estimate_motions(dst, st.Config())
    _, P2 = st.cumulative_path(est2["T"])
    jit = jitter_rms(P2)
    P2s = P2.copy(); P2s[:, :2] /= r["zoom"]
    shake = rms_no_const(P2s - P_ideal)
    return dict(scenario=name, method=tag, itf=itf(dst), shake_px=shake[0], shake_deg=shake[1],
                jitter_px=jit[0], jitter_deg=jit[1],
                err_track_px=e_track[0], err_track_deg=e_track[1], err_plan_px=e_plan[0],
                plan_oracle_px=oracle[0], zoom=r["zoom"], zoom_capped=r["zoom_capped"],
                failed_frames=int((~r["est"]["ok"]).sum()),
                mean_inliers=float(r["est"]["n_inliers"][1:].mean()), fps=r["fps_total"]), r


def baseline_row(name):
    src = os.path.join(DATA, f"{name}.avi")
    est = st.estimate_motions(src, st.Config())
    _, P = st.cumulative_path(est["T"])
    jit = jitter_rms(P)
    gt = np.load(os.path.join(DATA, f"{name}_gt.npz"))
    n = len(P)
    shake = rms_no_const(P - gt_params(gt["smooth"][:n], gt["real"][0]))
    return dict(scenario=name, method="original", itf=itf(src), shake_px=shake[0], shake_deg=shake[1],
                jitter_px=jit[0], jitter_deg=jit[1],
                err_track_px=np.nan, err_track_deg=np.nan, err_plan_px=np.nan, plan_oracle_px=np.nan, zoom=1.0,
                zoom_capped=False, failed_frames=0, mean_inliers=np.nan, fps=np.nan)


def save_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    rows = []
    for name in SCENARIOS:
        print(f"== {name}")
        rows.append(baseline_row(name))
        for tag, cfg in [("lk", st.Config(method="lk")),
                         ("lk_basic", st.Config(method="lk", normalize=False, fallback_orb=False, ransac_thresh=3.0)),
                         ("orb", st.Config(method="orb"))]:
            row, _ = run_scenario(name, cfg, tag)
            rows.append(row)
            print(f"   {tag:9s} ITF={row['itf']:.2f} shake={row['shake_px']:.2f}px "
                  f"track={row['err_track_px']:.2f}px fail={row['failed_frames']} zoom={row['zoom']:.3f}")
    save_csv(rows, os.path.join(RES, "metrics.csv"))

    # --- влияние радиуса сглаживания (базовый сценарий) ---
    abl = []
    for kind in ("moving_average", "gaussian"):
        for radius in (0, 3, 7, 15, 30, 60):
            row, _ = run_scenario("baseline", st.Config(smooth_radius=radius, smooth_kind=kind), f"r{radius}{kind[0]}")
            row.update(radius=radius, kind=kind)
            abl.append(row)
            print(f"   ablation {kind:14s} r={radius:2d} ITF={row['itf']:.2f} shake={row['shake_px']:.2f} "
                  f"plan={row['err_plan_px']:.2f} zoom={row['zoom']:.3f}")
    save_csv(abl, os.path.join(RES, "ablation.csv"))


if __name__ == "__main__":
    main()
