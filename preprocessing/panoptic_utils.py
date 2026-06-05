"""Small panoptic-control rendering utilities."""

from __future__ import annotations

import colorsys
from collections import namedtuple

import numpy as np

Label = namedtuple("Label", ["name", "trainId", "category", "color"])

_LABELS = [
    Label("road", 0, "flat", (128, 64, 128)),
    Label("sidewalk", 1, "flat", (244, 35, 232)),
    Label("building", 2, "construction", (70, 70, 70)),
    Label("wall", 3, "construction", (102, 102, 156)),
    Label("fence", 4, "construction", (190, 153, 153)),
    Label("pole", 5, "object", (153, 153, 153)),
    Label("traffic light", 6, "object", (250, 170, 30)),
    Label("traffic sign", 7, "object", (220, 220, 0)),
    Label("vegetation", 8, "nature", (107, 142, 35)),
    Label("terrain", 9, "nature", (152, 251, 152)),
    Label("sky", 10, "sky", (70, 130, 180)),
    Label("person", 11, "human", (220, 20, 60)),
    Label("rider", 12, "human", (255, 0, 0)),
    Label("car", 13, "vehicle", (0, 0, 142)),
    Label("truck", 14, "vehicle", (0, 0, 70)),
    Label("bus", 15, "vehicle", (0, 60, 100)),
    Label("train", 16, "vehicle", (0, 80, 100)),
    Label("motorcycle", 17, "vehicle", (0, 0, 230)),
    Label("bicycle", 18, "vehicle", (119, 11, 32)),
]

trainId2label = {label.trainId: label for label in _LABELS}


def _palette_from_hue_band(n: int, hue_start_deg: float, hue_end_deg: float, seed: int) -> list[tuple[int, int, int]]:
    rng = np.random.default_rng(seed)
    if hue_end_deg >= hue_start_deg:
        hues = np.linspace(hue_start_deg, hue_end_deg, n, endpoint=False)
    else:
        span = hue_end_deg + 360 - hue_start_deg
        hues = np.mod(hue_start_deg + np.linspace(0, span, n, endpoint=False), 360.0)
    rng.shuffle(hues)
    return [tuple(int(round(255 * c)) for c in colorsys.hsv_to_rgb(h / 360.0, 0.85, 0.95)) for h in hues]


HUMAN_COLORS = _palette_from_hue_band(256, 340, 60, seed=13)
VEHICLE_COLORS = _palette_from_hue_band(512, 100, 300, seed=29)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter) / float(np.logical_or(a, b).sum())


class SimpleMaskTracker:
    """Greedy IoU tracker used only for stable instance colors."""

    def __init__(self, iou_thresh: float = 0.5, max_age: int = 30):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.next_id = 1
        self.tracks: list[dict] = []

    def _prune(self) -> None:
        self.tracks = [track for track in self.tracks if track["age"] <= self.max_age]

    def update(self, masks: list[np.ndarray], labels: list[Label]) -> list[int]:
        for track in self.tracks:
            track["age"] += 1

        candidates = []
        for track_idx, track in enumerate(self.tracks):
            for det_idx, (mask, label) in enumerate(zip(masks, labels)):
                if track["label"] == label:
                    iou = _mask_iou(track["mask"], mask)
                    if iou >= self.iou_thresh:
                        candidates.append((iou, track_idx, det_idx))
        candidates.sort(reverse=True, key=lambda item: item[0])

        matched_tracks = set()
        matched_dets = set()
        track_ids = [-1] * len(masks)
        for _, track_idx, det_idx in candidates:
            if track_idx in matched_tracks or det_idx in matched_dets:
                continue
            self.tracks[track_idx]["mask"] = masks[det_idx]
            self.tracks[track_idx]["age"] = 0
            track_ids[det_idx] = self.tracks[track_idx]["id"]
            matched_tracks.add(track_idx)
            matched_dets.add(det_idx)

        for det_idx, (mask, label) in enumerate(zip(masks, labels)):
            if det_idx in matched_dets:
                continue
            track_id = self.next_id
            self.next_id += 1
            self.tracks.append({"id": track_id, "label": label, "mask": mask, "age": 0})
            track_ids[det_idx] = track_id

        self._prune()
        return track_ids


def draw_panoptic(frame: np.ndarray, panoptic: np.ndarray, segments_info: list[dict]) -> np.ndarray:
    seg = np.asarray(panoptic, dtype=np.int32)
    overlay = np.zeros_like(frame, dtype=np.uint8)
    tracker = getattr(draw_panoptic, "_tracker", None)
    if tracker is None:
        tracker = SimpleMaskTracker(iou_thresh=0.5, max_age=15)
        setattr(draw_panoptic, "_tracker", tracker)

    thing_masks, thing_labels = [], []
    for segment in segments_info:
        label = trainId2label.get(segment["label_id"])
        if label is None:
            continue
        mask = seg == segment["id"]
        if not mask.any():
            continue
        if label.category in {"vehicle", "human"}:
            thing_masks.append(mask)
            thing_labels.append(label)
        else:
            overlay[mask] = label.color

    track_ids = tracker.update(thing_masks, thing_labels) if thing_masks else []
    for mask, track_id, label in zip(thing_masks, track_ids, thing_labels):
        palette = HUMAN_COLORS if label.category == "human" else VEHICLE_COLORS
        overlay[mask] = palette[track_id % len(palette)]

    return overlay
