import torch
from base_model.data.loaders import make_dataset_for_videos, make_data_loader
from base_model.data.collate import collate_data_and_cast_with_aux_use_past_future_frames
from base_model.data.masking import MaskingGenerator
# 경로를 실제 환경에 맞게 바꾸세요
LMDB_PATH = "/data2/flow_dataset.lmdb"
ENTRIES_CSV = "/data2/flow_dataset.labels.csv"

dataset = make_dataset_for_videos(
    dataset_str=f"LMDBFlow:lmdb={LMDB_PATH}:entries={ENTRIES_CSV}"
)

print("Dataset size:", len(dataset))
# 샘플 1개 확인
img, target = dataset[0]
print("Sample images (past,current,future):", type(img), len(img))
print("Sample target:", target)

# DataLoader 테스트: collate 함수에 n_tokens=196(14x14) 전달, mask_generator는 None으로 두어도 내부에서 생성됨
mask_generator = MaskingGenerator(
        input_size=(256 // 14, 256 // 14),
        max_num_patches= 256 // 14 * 256 // 14,
    )

dl = make_data_loader(
    dataset=dataset,
    batch_size=2,
    num_workers=2,
    shuffle=False,
    collate_fn=lambda x: collate_data_and_cast_with_aux_use_past_future_frames(
        x,
        mask_ratio_tuple=(0.1, 0.5),
        mask_probability=0.5,
        dtype=torch.float32,
        n_tokens=196,
        mask_generator=mask_generator
    ),
    persistent_workers=False
)

for batch in dl:
    print("Batch keys:", list(batch.keys()))
    print("collated_global_crops shape:", batch["collated_global_crops"].shape)
    break