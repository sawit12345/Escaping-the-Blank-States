from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ObjectObservation:
    centroid: np.ndarray
    area: float
    bbox: np.ndarray
    mean_color: np.ndarray
    mean_motion: float


@dataclass
class SceneObservation:
    objects: list[ObjectObservation]
    frame_shape: tuple[int, int]
    foreground_ratio: float


def _connected_components(mask: np.ndarray, rgb: np.ndarray, motion: np.ndarray, min_area: int, max_area: int) -> list[ObjectObservation]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    ys, xs = np.where(mask)
    objects: list[ObjectObservation] = []

    for y, x in zip(ys.tolist(), xs.tolist(), strict=False):
        if visited[y, x]:
            continue

        stack = [(y, x)]
        visited[y, x] = 1

        area = 0
        sx = 0.0
        sy = 0.0
        sr = 0.0
        sg = 0.0
        sb = 0.0
        sm = 0.0
        min_x = x
        max_x = x
        min_y = y
        max_y = y

        while stack:
            cy, cx = stack.pop()
            area += 1
            sx += cx
            sy += cy
            sr += float(rgb[cy, cx, 0])
            sg += float(rgb[cy, cx, 1])
            sb += float(rgb[cy, cx, 2])
            sm += float(motion[cy, cx])

            if cx < min_x:
                min_x = cx
            if cx > max_x:
                max_x = cx
            if cy < min_y:
                min_y = cy
            if cy > max_y:
                max_y = cy

            ny = cy - 1
            if ny >= 0 and mask[ny, cx] and not visited[ny, cx]:
                visited[ny, cx] = 1
                stack.append((ny, cx))
            ny = cy + 1
            if ny < h and mask[ny, cx] and not visited[ny, cx]:
                visited[ny, cx] = 1
                stack.append((ny, cx))
            nx = cx - 1
            if nx >= 0 and mask[cy, nx] and not visited[cy, nx]:
                visited[cy, nx] = 1
                stack.append((cy, nx))
            nx = cx + 1
            if nx < w and mask[cy, nx] and not visited[cy, nx]:
                visited[cy, nx] = 1
                stack.append((cy, nx))

        if area < min_area or area > max_area:
            continue

        inv_area = 1.0 / float(area)
        objects.append(
            ObjectObservation(
                centroid=np.array([sx * inv_area, sy * inv_area], dtype=np.float32),
                area=float(area),
                bbox=np.array([min_x, min_y, max_x, max_y], dtype=np.float32),
                mean_color=np.array([sr * inv_area, sg * inv_area, sb * inv_area], dtype=np.float32),
                mean_motion=float(sm * inv_area),
            )
        )

    objects.sort(key=lambda obj: obj.area, reverse=True)
    return objects


def _binary_dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, ((1, 1), (1, 1)), mode="constant", constant_values=False)
    out = np.zeros_like(mask)
    for dy in range(3):
        for dx in range(3):
            out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def _binary_erode(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, ((1, 1), (1, 1)), mode="constant", constant_values=False)
    out = np.ones_like(mask)
    for dy in range(3):
        for dx in range(3):
            out &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


class CoreKnowledgePerceptor:
    def __init__(self, background_alpha: float = 0.01, diff_threshold: float = 16.0) -> None:
        self.background_alpha = float(background_alpha)
        self.diff_threshold = float(diff_threshold)
        self.last_frame: np.ndarray | None = None
        self.background: np.ndarray | None = None

    def _foreground_mask(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rgb = frame.astype(np.float32)
        if self.background is None:
            self.background = rgb.copy()

        bg = self.background
        if bg is None:
            bg = rgb.copy()
            self.background = bg
        diff = np.abs(rgb - bg).mean(axis=2)

        if self.last_frame is None:
            motion = np.zeros_like(diff, dtype=np.float32)
        else:
            motion = np.abs(frame.astype(np.float32) - self.last_frame.astype(np.float32)).mean(axis=2)

        motion_threshold = 8.0
        dynamic_threshold = float(np.percentile(diff, 75))
        threshold = max(self.diff_threshold, 0.65 * dynamic_threshold)

        bg_color = np.median(bg.reshape(-1, 3), axis=0)
        color_dist = np.sqrt(np.sum((rgb - bg_color[None, None, :]) ** 2, axis=2))
        color_threshold = max(8.0, 0.9 * float(np.percentile(color_dist, 65)))

        structural = color_dist > color_threshold
        dynamic = (diff > threshold) | (motion > motion_threshold)
        mask = structural | dynamic
        mask = _binary_erode(_binary_dilate(mask))

        static_mask = ~dynamic
        updated_bg = bg.copy()
        updated_bg[static_mask] = (1.0 - self.background_alpha) * bg[static_mask] + self.background_alpha * rgb[static_mask]
        self.background = updated_bg

        return mask, motion

    def extract(self, observation: np.ndarray) -> SceneObservation:
        h, w, _ = observation.shape
        mask, motion = self._foreground_mask(observation)

        min_area = max(2, int(h * w * 0.00008))
        max_area = max(min_area + 1, int(h * w * 0.22))
        objects = _connected_components(mask, observation, motion, min_area=min_area, max_area=max_area)
        if len(objects) > 64:
            objects = objects[:64]

        self.last_frame = observation.copy()
        return SceneObservation(
            objects=objects,
            frame_shape=(h, w),
            foreground_ratio=float(mask.mean()),
        )
