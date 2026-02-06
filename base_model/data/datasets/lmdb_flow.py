import lmdb
import pickle
import zstandard as zstd
import numpy as np
import os
import random
from typing import Any, Optional, Tuple, List
from .extended import ExtendedVisionDataset

class LMDBFlowDataset(ExtendedVisionDataset):
    """
    LMDB에 저장된 optical-flow clip을 읽어오는 Dataset.

    기대 포맷 (make_lmdb_multi.py와 동일):
      - key: ASCII string, ex: 'examA_0000'
      - value: zstd.compress(pickle.dumps(np_array))
      - np_array: dtype=uint8, shape = (T, H, W, 2)  # 채널0 = fx(+128), 채널1 = fy(+128)

    entries_path: CSV or pickle list-of-dicts. CSV expected columns:
      key, exam_id, clip_id, label, label_index
    """
    def __init__(
        self,
        *,
        lmdb_path: str,
        entries: Optional[List[dict]] = None,
        entries_path: Optional[str] = None,
        transforms: Optional[callable] = None,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        past_offset_range=(0.05, 0.15),
        current_range=(0.3, 0.7),
        future_offset_range=(0.05, 0.15),
    ):
        super().__init__(root=lmdb_path, transforms=transforms, transform=transform, target_transform=target_transform)
        self.lmdb_path = lmdb_path
        self._env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, max_readers=64)
        self._decompressor = zstd.ZstdDecompressor()
        self._entries = None

        if entries is not None:
            self._entries = entries
        elif entries_path is not None:
            if not os.path.exists(entries_path):
                raise FileNotFoundError(f"entries_path not found: {entries_path}")
            # CSV 또는 pickle 처리
            if entries_path.endswith('.csv'):
                import csv
                arr = []
                with open(entries_path, 'r') as f:
                    reader = csv.DictReader(f)
                    # Accept file with header key,exam_id,clip_id,label,label_index
                    for row in reader:
                        if not row:
                            continue
                        key = row.get('key') or row.get('Key')
                        if key is None:
                            continue
                        label_index = row.get('label_index') or row.get('label_index'.upper()) or row.get('label')
                        try:
                            label_index = int(label_index) if label_index not in (None, '') else -1
                        except Exception:
                            label_index = -1
                        exam_id = row.get('exam_id', '')
                        clip_id = row.get('clip_id', '')
                        label = row.get('label', '')
                        arr.append({"key": key, "exam_id": exam_id, "clip_id": clip_id, "label": label, "class_index": label_index})
                self._entries = arr
            else:
                # assume pickle
                with open(entries_path, 'rb') as f:
                    self._entries = pickle.load(f)

        if self._entries is None:
            # LMDB 내부 '__keys__' 존재 시 로드
            with self._env.begin() as txn:
                buf = txn.get(b"__keys__")
                if buf is not None:
                    keys = pickle.loads(buf)
                    self._entries = [{"key": k, "class_index": -1} for k in keys]
                else:
                    raise RuntimeError("No entries provided and __keys__ not found in LMDB. Provide entries or __keys__.")

        self._past_offset_range = past_offset_range
        self._current_range = current_range
        self._future_offset_range = future_offset_range

    def _get_entries(self):
        return self._entries

    def __len__(self) -> int:
        return len(self._get_entries())

    def _read_clip_by_key(self, key: str) -> np.ndarray:
        with self._env.begin() as txn:
            buf = txn.get(key.encode('ascii'))
            if buf is None:
                raise KeyError(f"Key not found in LMDB: {key}")
        raw = self._decompressor.decompress(buf)
        clip = pickle.loads(raw)
        clip = np.asarray(clip)
        return clip  # (T,H,W,2) uint8

    def get_dataset(self):
        return "LMDBFlow"

    def get_target(self, index: int) -> Optional[int]:
        entry = self._get_entries()[index]
        if isinstance(entry, dict):
            ci = entry.get("class_index", None)
            return int(ci) if ci is not None and ci >= 0 else None
        return None

    def _choose_indices(self, total_frames: int) -> Tuple[int,int,int]:
        def calc_range(rng):
            s = int(rng[0] * total_frames)
            e = int(rng[1] * total_frames)
            return s, e
        cur_s, cur_e = calc_range(self._current_range)
        if cur_e <= cur_s:
            current_idx = total_frames // 2
        else:
            current_idx = random.randint(cur_s, max(cur_s, cur_e - 1))
        past_start = max(0, current_idx - int(self._past_offset_range[1] * total_frames))
        past_end = max(0, current_idx - int(self._past_offset_range[0] * total_frames))
        future_start = min(total_frames - 1, current_idx + int(self._future_offset_range[0] * total_frames))
        future_end = min(total_frames - 1, current_idx + int(self._future_offset_range[1] * total_frames))

        valid = (past_start < past_end <= current_idx <= future_start < future_end) if (past_end>past_start and future_end>future_start) else False
        if not valid:
            idx = random.randint(0, total_frames - 1)
            return idx, idx, idx
        past_idx = random.randint(past_start, max(past_start, past_end))
        future_idx = random.randint(future_start, max(future_start, future_end))
        return past_idx, current_idx, future_idx

    def __getitem__(self, index: int):
        entries = self._get_entries()
        entry = entries[index]
        key = entry["key"] if isinstance(entry, dict) else entry
        clip = self._read_clip_by_key(key)  # (T,H,W,2)
        total_frames = int(clip.shape[0])
        if total_frames == 0:
            raise RuntimeError(f"Clip {key} has zero frames")

        past_idx, current_idx, future_idx = self._choose_indices(total_frames)

        past_frame = clip[past_idx]
        current_frame = clip[current_idx]
        future_frame = clip[future_idx]

        frame_name_prefix = key
        past_name = frame_name_prefix + f"#past_{past_idx:04d}"
        current_name = frame_name_prefix + f"#current_{current_idx:04d}"
        future_name = frame_name_prefix + f"#future_{future_idx:04d}"

        # transform (flow 전용 transform)과 target_transform 기대
        if self.transform is not None and self.target_transform is not None:
            past_out, past_pos = self.transform(past_name, past_frame, pos_tuple_list=None, center_crop=True)
            current_out, current_pos = self.transform(current_name, current_frame, pos_tuple_list=None, center_crop=False)
            future_out, future_pos = self.transform(future_name, future_frame, pos_tuple_list=past_pos, center_crop=True)
            target = self.get_target(index)
            target = self.target_transform(target) if target is not None else None
            images = [past_out, current_out, future_out]
            return images, target
        else:
            raise RuntimeError("LMDBFlowDataset requires both transform and target_transform to be set and flow-aware.")