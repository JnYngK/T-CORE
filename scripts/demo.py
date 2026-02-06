# scripts/test_lmdb_dataset.py
import torch
from base_model.data.loaders import make_dataset_for_videos, make_data_loader
from base_model.data.collate import collate_data_and_cast_with_aux_use_past_future_frames

dataset = make_dataset_for_videos(
    dataset_str="LMDBFlow:lmdb=/data2/flow_dataset.lmdb:entries=label_example.csv"
)

# test one sample
print("Dataset size:", len(dataset))
img, target = dataset[0]
print("Sample structure:", type(img), len(img), "target:", target)

# dataloader test (small)
dl = make_data_loader(dataset=dataset, batch_size=2, num_workers=2, shuffle=False, collate_fn=lambda x: collate_data_and_cast_with_aux_use_past_future_frames(x, mask_ratio_tuple=(0.1,0.5), mask_probability=0.5, dtype=torch.float32, n_tokens=196, mask_generator=None))
for batch in dl:
    print("Got batch keys:", batch.keys())
    break