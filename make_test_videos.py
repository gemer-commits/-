"""
Генерация тестовых «дрожащих» видео с ИЗВЕСТНОЙ истиной.

Берём большую текстурную картинку («мир») и двигаем по ней виртуальную камеру:
  плавное движение (то, что оператор хотел)  +  дрожание (то, что надо убрать).
Так мы точно знаем, как «должна» двигаться камера, и можем честно померять точность.

Сценарии проверяют устойчивость: шум, засветка, движущиеся объекты, перекрытие, размытие,
бедная текстура, сильное дрожание.
"""
import json
import os

import cv2
import numpy as np

W, H, N, FPS = 640, 360, 240, 25
OUT = os.path.join(os.path.dirname(__file__), "data")


def make_world(seed=0, texture=1.0, fine_noise=10.0, cw=2200, ch=1400):
    rng = np.random.default_rng(seed)
    low = rng.random((ch // 120 + 2, cw // 120 + 2, 3)).astype(np.float32)
    world = (60 + 140 * cv2.resize(low, (cw, ch), interpolation=cv2.INTER_CUBIC)).astype(np.uint8)
    for _ in range(int(700 * texture)):
        x, y = int(rng.integers(0, cw)), int(rng.integers(0, ch))
        col = tuple(int(v) for v in rng.integers(0, 255, 3))
        size, kind = int(rng.integers(8, 60)), int(rng.integers(0, 4))
        if kind == 0:
            cv2.circle(world, (x, y), size, col, -1)
        elif kind == 1:
            cv2.rectangle(world, (x, y), (x + size * 2, y + size), col, -1)
        elif kind == 2:
            dx, dy = (int(v) for v in rng.integers(-150, 150, 2))
            cv2.line(world, (x, y), (x + dx, y + dy), col, int(rng.integers(2, 6)))
        else:
            cv2.putText(world, str(int(rng.integers(0, 999))), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, size / 25, col, 2)
    if fine_noise > 0:
        noise = cv2.GaussianBlur(rng.normal(0, fine_noise, (ch, cw)).astype(np.float32), (0, 0), 1.0)
        world = np.clip(world.astype(np.float32) + noise[..., None], 0, 255).astype(np.uint8)
    return world


def camera_poses(n=N, jitter_xy=6.0, jitter_deg=0.7, seed=1):
    """Возвращает (реальные позы, «желаемые» позы без дрожания). Поза = (cx, cy, угол)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    sx = 1100 + 250 * np.sin(2 * np.pi * t / n * 0.8)
    sy = 700 + 90 * np.sin(2 * np.pi * t / n * 0.5 + 1.0)
    sa = np.deg2rad(3) * np.sin(2 * np.pi * t / n * 0.3)

    def jit(rms):
        w = np.convolve(rng.normal(0, 1, n + 8), np.ones(5) / 5, "same")[4:-4]
        return rms * w / w.std()

    smooth = np.stack([sx, sy, sa], 1)
    real = smooth + np.stack([jit(jitter_xy), jit(jitter_xy), jit(np.deg2rad(jitter_deg))], 1)
    return real, smooth


def render_view(world, pose):
    cx, cy, a = pose
    c, s = np.cos(a), np.sin(a)
    M = np.array([[c, -s, cx - (c * W / 2 - s * H / 2)],
                  [s, c, cy - (s * W / 2 + c * H / 2)]])
    return cv2.warpAffine(world, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT)


def draw_moving_objects(world, i, n=N):
    """Рисуем движущиеся объекты прямо в «мире» — они тоже трясутся вместе с камерой."""
    w = world.copy()
    specs = [(-250, -60, 4.0, (30, 30, 200)), (200, 80, -3.0, (200, 60, 30)),
             (-60, 110, 2.5, (30, 160, 30)), (120, -90, -4.5, (160, 30, 160))]
    for k, (ox, oy, v, col) in enumerate(specs):
        x = int(1100 + ox + v * (i - n / 2))
        y = int(700 + oy + 20 * np.sin(i / 15 + k))
        cv2.circle(w, (x, y), 55, col, -1)
        cv2.circle(w, (x, y), 30, tuple(255 - c for c in col), -1)
        cv2.circle(w, (x, y), 10, col, -1)
    return w


# --- эффекты, которые накладываются на кадр уже после съёмки ------------------
def fx_noise(frame, rng, sigma):
    return np.clip(frame + rng.normal(0, sigma, frame.shape), 0, 255).astype(np.uint8)


def fx_brightness(frame, i):
    gain = 1.0 + 0.30 * np.sin(i / 6.0)
    if 100 <= i < 140:                      # сильная засветка
        gain, off = 2.4, 40
    else:
        off = 0
    return np.clip(frame.astype(np.float32) * gain + off, 0, 255).astype(np.uint8)


def fx_occlusion(frame, i):
    if 80 <= i < 160:                       # крупный объект закрывает ~40% кадра
        x0 = int(W * 0.15 + (i - 80) * 2)
        frame = frame.copy()
        cv2.rectangle(frame, (x0, 0), (x0 + int(W * 0.40), H), (20, 20, 20), -1)
    return frame


def fx_blur_lowlight(frame, rng):
    frame = cv2.blur(frame, (9, 1)).astype(np.float32) * 0.40
    return np.clip(frame + rng.normal(0, 8, frame.shape), 0, 255).astype(np.uint8)


SCENARIOS = {
    # имя: (описание, параметры мира, параметры дрожания, эффект)
    "baseline":       ("обычное дрожание",                      dict(),                      dict(), None),
    "strong_jitter":  ("сильное дрожание и повороты",           dict(),                      dict(jitter_xy=10, jitter_deg=1.5), None),
    "noise":          ("сильный шум (σ=20)",                    dict(),                      dict(), "noise"),
    "brightness":     ("мерцание яркости + засветка",           dict(),                      dict(), "brightness"),
    "moving_objects": ("4 движущихся объекта",                  dict(),                      dict(), "objects"),
    "occlusion":      ("перекрытие ~40% кадра",                 dict(),                      dict(), "occlusion"),
    "blur_lowlight":  ("размытие + тёмное изображение",         dict(),                      dict(), "blur"),
    "low_texture":    ("бедная текстура сцены",                 dict(texture=0.12, fine_noise=0), dict(), None),
}


def make_all():
    os.makedirs(OUT, exist_ok=True)
    for name, (desc, wkw, jkw, fx) in SCENARIOS.items():
        world = make_world(**wkw)
        real, smooth = camera_poses(**jkw)
        rng = np.random.default_rng(42)
        path = os.path.join(OUT, f"{name}.avi")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
        for i in range(N):
            src = draw_moving_objects(world, i) if fx == "objects" else world
            f = render_view(src, real[i])
            if fx == "noise": f = fx_noise(f, rng, 20)
            elif fx == "brightness": f = fx_brightness(f, i)
            elif fx == "occlusion": f = fx_occlusion(f, i)
            elif fx == "blur": f = fx_blur_lowlight(f, rng)
            vw.write(f)
        vw.release()
        np.savez(os.path.join(OUT, f"{name}_gt.npz"), real=real, smooth=smooth)
        print(f"{name:15s} — {desc}")
    json.dump({k: v[0] for k, v in SCENARIOS.items()}, open(os.path.join(OUT, "scenarios.json"), "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    make_all()
