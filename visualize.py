"""
Визуализация результатов: рисунки для отчёта и демо-видео «до / после».
Запускать ПОСЛЕ evaluate.py (нужны results/*.avi и results/*.csv).
"""
import csv
import os
import subprocess

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import stabilizer as st
from evaluate import gt_params, DATA, RES

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 110})
CFG = st.Config()


def read_frames(path, idx):
    cap, out = cv2.VideoCapture(path), {}
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in idx:
            out[i] = f
        i += 1
    cap.release()
    return out


def rgb(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------
def fig_pipeline(scenario="moving_objects", i=120):
    """Этапы конвейера на одном кадре: фильтрация -> углы -> поток -> RANSAC -> результат."""
    src = os.path.join(DATA, f"{scenario}.avi")
    fr = read_frames(src, {i - 1, i})
    prep = st.make_preprocessor(CFG)
    prev, cur = fr[i - 1], fr[i]
    g_prev, g_cur = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), prep(cur)
    h, w = g_cur.shape
    center = np.array([w / 2, h / 2], np.float32)

    gp = prep(prev)
    corners = cv2.goodFeaturesToTrack(gp, CFG.max_corners, CFG.quality, CFG.min_distance, blockSize=3).reshape(-1, 2)
    A, p0, p1, inl = st.track_pair(gp, g_cur, CFG, center)

    flow = rgb(cur).copy()
    for a, b, ok in zip(p0, p1, inl):
        col = (0, 200, 0) if ok else (255, 0, 0)
        cv2.arrowedLine(flow, tuple(int(v) for v in a), tuple(int(v) for v in a + (b - a) * 4), col, 1, tipLength=0.3)
        cv2.circle(flow, tuple(int(v) for v in a), 2, col, -1)

    corn = rgb(prev).copy()
    for x, y in corners:
        cv2.circle(corn, (int(x), int(y)), 3, (255, 255, 0), -1)

    stab = os.path.join(RES, f"{scenario}_lk.avi")
    after = read_frames(stab, {i})[i]

    panels = [("1. Исходный кадр", rgb(cur)),
              ("2. Серый + гауссово размытие", cv2.cvtColor(g_cur, cv2.COLOR_GRAY2RGB)),
              (f"3. Углы Ши–Томаси ({len(corners)})", corn),
              (f"4. Поток LK + RANSAC (×4)\nзелёные={int(inl.sum())}, красные={int((~inl).sum())}", flow),
              ("5. После стабилизации", rgb(after))]
    fig, axs = plt.subplots(2, 3, figsize=(15, 6.6))
    for ax, (t, im) in zip(axs.ravel(), panels):
        ax.imshow(im); ax.set_title(t, fontsize=10); ax.axis("off")
    axs.ravel()[-1].axis("off")
    axs.ravel()[-1].text(0.02, 0.5,
                         "Красные стрелки — точки на движущихся\nобъектах: RANSAC считает их выбросами,\n"
                         "поэтому они не влияют на оценку\nдвижения камеры.", fontsize=11, va="center")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_pipeline.png")); plt.close(fig)


def fig_trajectory(scenario="baseline"):
    src = os.path.join(DATA, f"{scenario}.avi")
    est = st.estimate_motions(src, CFG)
    A, P = st.cumulative_path(est["T"])
    S = st.smooth_path(P, CFG.smooth_radius)
    gt = np.load(os.path.join(DATA, f"{scenario}_gt.npz"))
    n = len(P)
    ideal = gt_params(gt["smooth"][:n], gt["real"][0])
    true = gt_params(gt["real"][:n], gt["real"][0])
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["Сдвиг по X, px", "Сдвиг по Y, px", "Поворот, градусы"]
    for j, ax in enumerate(axs):
        k = 1 if j < 2 else np.degrees(1)
        ax.plot(P[:, j] * k, color="tab:red", lw=1.3, label="оценённая траектория (с дрожанием)")
        ax.plot(true[:, j] * k, "k:", lw=1, label="истинная траектория")
        ax.plot(S[:, j] * k, color="tab:green", lw=2.2, label=f"сглаженная (окно ±{CFG.smooth_radius})")
        ax.plot(ideal[:, j] * k, color="tab:blue", lw=1, ls="--", label="желаемая (без дрожания)")
        ax.set_ylabel(labels[j])
    axs[0].legend(ncol=2, fontsize=8, loc="upper right")
    axs[-1].set_xlabel("номер кадра")
    axs[0].set_title("Траектория камеры: красная линия вычитается, зелёная — то, что остаётся")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_trajectory.png")); plt.close(fig)


def xt_slice(path, y):
    cap, rows = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        rows.append(f[y])
    cap.release()
    return rgb(np.array(rows))


def fig_xt(scenario="baseline"):
    """Срез «время × x»: в дрожащем видео строки пилообразные, в стабилизированном — плавные."""
    a = xt_slice(os.path.join(DATA, f"{scenario}.avi"), 180)
    b = xt_slice(os.path.join(RES, f"{scenario}_lk.avi"), 180)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, im, t in zip(axs, (a, b), ("До: линии дёргаются", "После: линии плавные")):
        ax.imshow(im, aspect="auto"); ax.set_title(t); ax.set_xlabel("x, px"); ax.set_ylabel("время, кадры"); ax.grid(False)
    fig.suptitle("Срез видео по одной строке (y=180) во времени", y=1.0)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_xt_slice.png")); plt.close(fig)


def fig_before_after(scenario="strong_jitter", i=100):
    a = read_frames(os.path.join(DATA, f"{scenario}.avi"), {i, i + 1})
    b = read_frames(os.path.join(RES, f"{scenario}_lk.avi"), {i, i + 1})
    d = lambda fr: cv2.absdiff(cv2.cvtColor(fr[i], cv2.COLOR_BGR2GRAY), cv2.cvtColor(fr[i + 1], cv2.COLOR_BGR2GRAY))
    da, db = d(a), d(b)
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    axs[0, 0].imshow(rgb(a[i])); axs[0, 0].set_title("До (кадр %d)" % i)
    axs[0, 1].imshow(da, cmap="gray", vmin=0, vmax=120); axs[0, 1].set_title("До: разность соседних кадров (mean=%.1f)" % da.mean())
    axs[1, 0].imshow(rgb(b[i])); axs[1, 0].set_title("После (кадр %d)" % i)
    axs[1, 1].imshow(db, cmap="gray", vmin=0, vmax=120); axs[1, 1].set_title("После: разность соседних кадров (mean=%.1f)" % db.mean())
    for ax in axs.ravel():
        ax.axis("off")
    fig.suptitle("Сценарий «сильное дрожание»: чем темнее разность, тем стабильнее видео")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_before_after.png")); plt.close(fig)


# --------------------------------------------------------------------------
def load_csv(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fig_metrics():
    rows = load_csv("metrics.csv")
    scen = list(dict.fromkeys(r["scenario"] for r in rows))
    methods = [("original", "оригинал", "tab:gray"), ("lk", "LK (наш метод)", "tab:green"),
               ("lk_basic", "LK базовый (без доработок)", "tab:orange"), ("orb", "ORB (альтернатива)", "tab:blue")]
    get = lambda s, m, k: float(next(r[k] for r in rows if r["scenario"] == s and r["method"] == m))
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    x, wd = np.arange(len(scen)), 0.2
    for k, (key, lab, col) in enumerate(methods):
        axs[0].bar(x + (k - 1.5) * wd, [get(s, key, "itf") for s in scen], wd, label=lab, color=col)
        axs[1].bar(x + (k - 1.5) * wd, [get(s, key, "jitter_px") for s in scen], wd, label=lab, color=col)
    for ax, t in zip(axs, ("ITF (средний PSNR соседних кадров), дБ — больше лучше",
                           "Высокочастотное дрожание, px — меньше лучше")):
        ax.set_xticks(x); ax.set_xticklabels(scen, rotation=30, ha="right"); ax.set_title(t, fontsize=10)
    axs[0].set_ylim(10, None); axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_metrics.png")); plt.close(fig)


def fig_ablation():
    rows = load_csv("ablation.csv")
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for kind, col in (("moving_average", "tab:green"), ("gaussian", "tab:purple")):
        rr = [r for r in rows if r["kind"] == kind]
        rad = [int(r["radius"]) for r in rr]
        axs[0].plot(rad, [float(r["jitter_px"]) for r in rr], "o-", color=col, label=kind)
        axs[1].plot(rad, [float(r["err_plan_px"]) for r in rr], "o-", color=col, label=kind)
        axs[2].plot(rad, [float(r["zoom"]) for r in rr], "o-", color=col, label=kind)
    axs[0].set_title("Остаточное ВЧ-дрожание, px"); axs[1].set_title("Отклонение от желаемой траектории, px")
    axs[2].set_title("Необходимый zoom (потеря поля зрения)")
    for ax in axs:
        ax.set_xlabel("радиус сглаживания, кадры"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_ablation.png")); plt.close(fig)


def fig_inliers():
    """Сколько точек остаётся после RANSAC по ходу видео — видно, где метод «страдает»."""
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    for name in ("baseline", "noise", "occlusion", "brightness", "blur_lowlight", "low_texture"):
        src = os.path.join(DATA, f"{name}.avi")
        axs[0].plot(st.estimate_motions(src, st.Config())["n_inliers"], lw=1, label=name)
    axs[0].set_title("Число «хороших» точек (inliers) по кадрам — LK"); axs[0].set_xlabel("кадр"); axs[0].legend(fontsize=8)
    src = os.path.join(DATA, "brightness.avi")
    for norm, col in ((False, "tab:red"), (True, "tab:green")):
        e = st.estimate_motions(src, st.Config(normalize=norm, fallback_orb=False))
        axs[1].plot(e["n_inliers"], color=col, lw=1.2, label="с нормировкой яркости" if norm else "без нормировки")
    axs[1].axvspan(100, 140, color="orange", alpha=0.2, label="засветка (кадры 100–139)")
    axs[1].set_title("Сценарий «brightness»: влияние нормировки яркости"); axs[1].set_xlabel("кадр"); axs[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_inliers.png")); plt.close(fig)


def fig_brightness_steps():
    """Межкадровый сдвиг в ГОТОВОМ видео: у хорошей стабилизации он маленький и плавный.
    Если на засветке оценка движения потеряна, в этих кадрах видно резкий «рывок»."""
    probe = st.Config()          # одинаковый «измеритель» для всех видео
    fig, ax = plt.subplots(figsize=(11, 4))
    for path, lab, col, lw in ((os.path.join(DATA, "brightness.avi"), "оригинал", "tab:gray", 1),
                               (os.path.join(RES, "brightness_lk_basic.avi"), "LK базовый", "tab:orange", 1.4),
                               (os.path.join(RES, "brightness_lk.avi"), "LK итоговый", "tab:green", 1.8)):
        T = st.estimate_motions(path, probe)["T"]
        ax.plot([np.hypot(t[0, 2], t[1, 2]) for t in T], color=col, lw=lw, label=lab)
    ax.axvline(100, color="orange", ls=":"); ax.axvline(140, color="orange", ls=":")
    ax.set_xlabel("кадр"); ax.set_ylabel("сдвиг между соседними кадрами, px"); ax.legend()
    ax.set_title("Сценарий «brightness»: рывки в готовом видео (пунктир — резкое изменение яркости)")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_brightness_steps.png")); plt.close(fig)


# --------------------------------------------------------------------------
def make_demo(scenario="moving_objects", out="demo_before_after.mp4"):
    """Видео: слева оригинал с точками потока, справа стабилизированное."""
    src = os.path.join(DATA, f"{scenario}.avi")
    stab = os.path.join(RES, f"{scenario}_lk.avi")
    a, b = cv2.VideoCapture(src), cv2.VideoCapture(stab)
    prep = st.make_preprocessor(CFG)
    tmp = os.path.join(RES, "_demo_tmp.avi")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 25, (1280, 360))
    center, prev = None, None
    while True:
        ok1, fa = a.read(); ok2, fb = b.read()
        if not (ok1 and ok2):
            break
        g = prep(fa)
        if center is None:
            center = np.array([g.shape[1] / 2, g.shape[0] / 2], np.float32)
        left = fa.copy()
        if prev is not None:
            _, p0, p1, inl = st.track_pair(prev, g, CFG, center)
            if p0 is not None and inl is not None:
                for p, q, ok in zip(p0, p1, inl):
                    col = (0, 200, 0) if ok else (0, 0, 255)
                    cv2.arrowedLine(left, tuple(int(v) for v in p), tuple(int(v) for v in p + (q - p) * 4), col, 1, tipLength=0.3)
        prev = g
        for img, t in ((left, "DO (zelenye - fon, krasnye - dvizh. obekty)"), (fb, "POSLE stabilizacii")):
            cv2.putText(img, t, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3)
            cv2.putText(img, t, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        vw.write(np.hstack([left, fb]))
    vw.release()
    dst = os.path.join(RES, out)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "23", dst], check=True)
    os.remove(tmp)
    return dst


if __name__ == "__main__":
    fig_pipeline(); fig_trajectory(); fig_xt(); fig_before_after(); fig_metrics(); fig_ablation(); fig_inliers(); fig_brightness_steps()
    print(make_demo())
