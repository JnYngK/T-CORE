import torch
import logging

from enum import Enum
from torch.utils.data import Sampler
from typing import Any, Callable, List, Optional, TypeVar
from .datasets import Kinetics
from .samplers import EpochSampler, InfiniteSampler, ShardedInfiniteSampler

logger = logging.getLogger("TCoRe")


class SamplerType(Enum):
    DISTRIBUTED = 0
    EPOCH = 1
    INFINITE = 2
    SHARDED_INFINITE = 3
    SHARDED_INFINITE_NEW = 4


def _parse_dataset_str(dataset_str: str):
    tokens = dataset_str.split(":")

    name = tokens[0]
    kwargs = {}

    for token in tokens[1:]:
        key, value = token.split("=")
        assert key in ("root", "extra", "split", "lmdb", "entries")
        kwargs[key] = value

    if name == "Kinetics":
        class_ = Kinetics
        if "split" in kwargs:
            kwargs["split"] = Kinetics.Split[kwargs["split"]]
    elif name == "LMDBFlow":
        # LMDBFlow dataset string format:
        #   "LMDBFlow:lmdb=/path/to/flow_dataset.lmdb:entries=/path/to/entries.csv"
        # Supported keys: lmdb (required), entries (optional path to CSV/pickle)
        try:
            from .datasets.lmdb_flow import LMDBFlowDataset
        except Exception as e:
            raise ImportError("LMDBFlowDataset not found. Ensure base_model/data/datasets/lmdb_flow.py exists.") from e
        class_ = LMDBFlowDataset
        # map parsed kwargs to constructor expected names
        if "lmdb" in kwargs:
            kwargs["lmdb_path"] = kwargs.pop("lmdb")
        if "entries" in kwargs:
            kwargs["entries_path"] = kwargs.pop("entries")
    else:
        raise ValueError(f'Unsupported dataset "{name}"')

    return class_, kwargs


def make_dataset_for_videos(
    *,
    dataset_str: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    past_offset_range = (0.05, 0.15),
    current_range = (0.3, 0.7),
    future_offset_range = (0.05, 0.15)
):
    logger.info(f'using dataset: "{dataset_str}"')

    class_, kwargs = _parse_dataset_str(dataset_str)

    # If LMDBFlow is requested and the user did not provide transforms,
    # set default FlowDataAugmentation and a simple target_transform that
    # converts label_index -> torch.tensor or None for -1.
    if dataset_str.startswith("LMDBFlow:") or getattr(class_, "__name__", "") == "LMDBFlowDataset":
        # import lazily to avoid circular import problems
        try:
            from .augmentations_flow import FlowDataAugmentation
        except Exception:
            # If not available, raise with hint
            raise ImportError(
                "FlowDataAugmentation not found. Ensure base_model/data/augmentations_flow.py exists."
            )

        if transform is None:
            transform = FlowDataAugmentation(
                global_crops_scale=(0.4, 1.0),
                local_crops_scale=(0.05, 0.4),
                local_crops_number=6,
                global_crops_size=224,
                local_crops_size=96,
            )
        if target_transform is None:
            import torch as _torch
            target_transform = lambda x: None if x is None or int(x) < 0 else _torch.tensor(int(x), dtype=_torch.long)

    dataset = class_(transform=transform, target_transform=target_transform,
                     past_offset_range=past_offset_range, current_range=current_range, future_offset_range=future_offset_range,
                     **kwargs)

    logger.info(f"# of dataset samples: {len(dataset):,d}")

    # preserve original behaviour: set attributes if missing
    if not hasattr(dataset, "transform"):
        setattr(dataset, "transform", transform)
    if not hasattr(dataset, "target_transform"):
        setattr(dataset, "target_transform", target_transform)

    return dataset


def _make_sampler(
    *,
    dataset,
    type: Optional[SamplerType] = None,
    shuffle: bool = False,
    seed: int = 0,
    size: int = -1,
    advance: int = 0,
) -> Optional[Sampler]:
    sample_count = len(dataset)

    if type == SamplerType.INFINITE:
        logger.info("sampler: infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        return InfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
        )
    elif type in (SamplerType.SHARDED_INFINITE, SamplerType.SHARDED_INFINITE_NEW):
        logger.info("sampler: sharded infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        use_new_shuffle_tensor_slice = type == SamplerType.SHARDED_INFINITE_NEW
        return ShardedInfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
            use_new_shuffle_tensor_slice=use_new_shuffle_tensor_slice,
        )
    elif type == SamplerType.EPOCH:
        logger.info("sampler: epoch")
        if advance > 0:
            raise NotImplementedError("sampler advance > 0 is not supported")
        size = size if size > 0 else sample_count
        logger.info(f"# of samples / epoch: {size:,d}")
        return EpochSampler(
            size=size,
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
        )
    elif type == SamplerType.DISTRIBUTED:
        logger.info("sampler: distributed")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        if advance > 0:
            raise ValueError("sampler advance > 0 is invalid")
        return torch.utils.data.DistributedSampler(
            dataset=dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )

    logger.info("sampler: none")
    return None


T = TypeVar("T")


def make_data_loader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 0,
    sampler_type: Optional[SamplerType] = SamplerType.INFINITE,
    sampler_size: int = -1,
    sampler_advance: int = 0,
    drop_last: bool = True,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable[[List[T]], Any]] = None,
    worker_init_fn: Optional[Callable[[int], None]] = None,
):
    """
    DataLoader factory with LMDB-friendly worker init behaviour.

    If worker_init_fn is None, a default worker_init_fn is provided that re-opens
    lmdb environments in worker processes when dataset has attribute 'lmdb_path'.
    This prevents child processes from attempting to use a forked lmdb.Environment.
    """

    sampler = _make_sampler(
        dataset=dataset,
        type=sampler_type,
        shuffle=shuffle,
        seed=seed,
        size=sampler_size,
        advance=sampler_advance,
    )

    # Default LMDB-aware worker_init_fn
    if worker_init_fn is None:
        def _default_worker_init_fn(worker_id: int):
            # get worker info and dataset reference
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                return
            ds = worker_info.dataset
            # If dataset exposes lmdb_path, re-open lmdb env per worker to avoid fork problems.
            if hasattr(ds, "lmdb_path"):
                try:
                    import lmdb
                    # Close parent's env if present
                    try:
                        env = getattr(ds, "_env", None)
                        if env is not None:
                            try:
                                env.close()
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # Open a fresh, read-only environment in worker
                    # Recommended options: readonly=True, lock=False, readahead=False, max_readers tuned for concurrency
                    ds._env = lmdb.open(ds.lmdb_path, readonly=True, lock=False, readahead=False, max_readers=64)
                except Exception as e:
                    logger.warning(f"Failed to open lmdb in worker {worker_id}: {e}")
        worker_init = _default_worker_init_fn
    else:
        worker_init = worker_init_fn

    logger.info("using PyTorch data loader")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init,
    )

    try:
        logger.info(f"# of batches: {len(data_loader):,d}")
    except TypeError:
        logger.info("infinite data loader")
    return data_loader