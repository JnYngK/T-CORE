import lmdb
import numpy as np
import cv2
import pickle
import glob
import math
import multiprocessing as mp
import zstandard as zstd
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor
import json


PROGRESS_PATH = '/data2/flow_build_progress.json'

def load_progress():
    if not os.path.exists(PROGRESS_PATH):
        return 0  # 아직 아무 것도 안 함
    with open(PROGRESS_PATH, 'r') as f:
        obj = json.load(f)
    return obj.get('last_done_index', 0)

def save_progress(idx):
    with open(PROGRESS_PATH, 'w') as f:
        json.dump({'last_done_index': idx}, f)
                  
def load_gray(path):
    return cv2.imread(path, 0)

# -----------------------------------------------------------------------------
# 1. Pre-scan: 태스크 생성 및 ID 미리 할당 (Pass 1)
# -----------------------------------------------------------------------------
def prepare_tasks(exam_path_list):
    """
    모든 비디오 폴더의 파일 개수를 미리 파악하여,
    각 비디오가 맡아야 할 '시작 클립 ID'를 부여한 태스크 리스트를 생성합니다.
    """
    tasks = []
    print("Step 1: Scanning directories to assign Clip IDs...")
    
    for exam_path in tqdm(exam_path_list):
        exam_id = exam_path.split('/')[-1]
        
        # 비디오 목록 정렬 (매우 중요: 이 순서대로 ID가 부여됨)
        video_paths = sorted(glob.glob(f'{exam_path}/*'))
        
        if len(video_paths) < 100:
            continue
            
        current_clip_id = 0
        
        for v_path in video_paths:
            # glob은 느릴 수 있으므로 os.listdir나 fast scandir 고려 가능하지만,
            # 정확성을 위해 기존 패턴 유지하되 개수만 셉니다.
            # (실제 이미지 로드는 안 하므로 비교적 빠릅니다)
            # flow_x 파일 개수 파악
            # 주의: 여기서 glob을 또 하면 느리니, len만 빠르게 체크하거나
            # 정확한 계산을 위해선 어쩔 수 없이 glob을 써야 합니다.
            # 속도 최적화를 위해 파일명 패턴을 안다면 os.listdir로 카운팅하는게 낫습니다.
            
            # 여기서는 안전하게 glob 사용 (메타데이터 로딩 시간은 전체의 1% 미만임)
            # 개수만 셉니다.
            n_files = len(os.listdir(v_path))/2
            if n_files == 0:
                continue
                
            n_clips = math.ceil(n_files / 150)
            
            # 태스크 정보: (비디오경로, 검사ID, 시작클립ID)
            tasks.append({
                'video_path': v_path,
                'exam_id': exam_id,
                'start_id': current_clip_id
            })
            
            # 다음 비디오를 위해 ID 증가
            current_clip_id += n_clips
            
    return tasks

import subprocess

def prepare_tasks_fast(root_paths):
    """
    Linux 'find' 명령어를 사용하여 NAS상의 모든 이미지 파일을 초고속으로 스캔합니다.
    """
    tasks = []
    print("Scanning files using 'find' command (Fast Mode)...")
    
    # 여러 경로를 검색해야 하므로 find 명령어 인자 구성
    # find /path1 /path2 -name "flow_x_*.jpg"
    cmd = ['find'] + root_paths + ['-name', 'flow_x_*.jpg']
    
    # subprocess로 실행하고 결과를 파이프로 받음
    # encoding='utf-8'로 바로 문자열 수신
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # 전체 파일 리스트를 메모리에 로드 (경로 문자열이므로 수십만 개도 OK)
    # communicate()는 프로세스가 끝날 때까지 기다렸다가 한번에 받음
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"Error in find command: {stderr}")
        return []
    
    # 줄바꿈으로 분리하여 리스트 생성
    all_files = stdout.strip().split('\n')
    
    if not all_files or all_files == ['']:
        return []

    print(f"Found {len(all_files)} files. Grouping by video...")
    
    # 파일들을 비디오 경로 기준으로 그룹화
    # 딕셔너리 구조: { 'video_path': count }
    video_counts = {}
    
    for file_path in tqdm(all_files):
        # /path/to/video/flow_x_0001.jpg -> /path/to/video 추출
        video_path = os.path.dirname(file_path)
        
        # 카운팅 (딕셔너리 접근은 O(1)이라 매우 빠름)
        if video_path in video_counts:
            video_counts[video_path] += 1
        else:
            video_counts[video_path] = 1
            
    # 그룹화된 정보를 바탕으로 최종 Task 생성
    # 검사 폴더 기준으로 정렬하여 ID 부여 순서 보장 (중요)
    sorted_video_paths = sorted(video_counts.keys())
    
    current_clip_id = 0
    for v_path in sorted_video_paths:
        n_files = video_counts[v_path]
        
        # 검사 ID 추출 (video_path의 상위 폴더 이름)
        # 예: /.../Patient_001/Video_01 -> Patient_001
        exam_id = os.path.basename(os.path.dirname(v_path))
        
        n_clips = math.ceil(n_files / 150)
        
        tasks.append({
            'video_path': v_path,
            'exam_id': exam_id,
            'start_id': current_clip_id
        })
        
        current_clip_id += n_clips
        
    return tasks

def load_existing_keys(lmdb_path):
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    existing = set()
    with env.begin() as txn:
        buf = txn.get(b'__keys__')
        if buf is not None:
            keys = pickle.loads(buf)
            existing.update(keys)
        else:
            # __keys__가 아직 없으면, 커서로 전체 키 스캔
            cursor = txn.cursor()
            for k, _ in tqdm(cursor):
                if k == b'__keys__':
                    continue
                existing.add(k.decode('ascii'))
    env.close()
    return existing
# -----------------------------------------------------------------------------
# 2. Worker Process (Queue 기반)
# -----------------------------------------------------------------------------
def worker_process(input_queue, result_queue):
    compressor = zstd.ZstdCompressor(level=6)

    while True:
        task_idx, task = input_queue.get()
        if task is None:  # 종료 신호(Sentinel)
            break
            
        video_path = task['video_path']
        exam_id = task['exam_id']
        start_id = task['start_id']
        
        # 파일 읽기 및 처리
        flow_x_files = sorted(glob.glob(f'{video_path}/flow_x_*.jpg'))
        flow_y_files = sorted(glob.glob(f'{video_path}/flow_y_*.jpg'))
        
        if not flow_x_files:
            continue

        try:
            # I/O 최적화를 위해 여기서도 ThreadPool 사용 가능 (생략)
            sample = cv2.imread(flow_x_files[0], 0)
            if sample is None: continue
            
            
            num_clips = math.ceil(len(flow_x_files) / 150)
            with ThreadPoolExecutor(max_workers=32) as pool:
                # 전체 프레임을 미리 읽어두기
                xs = list(pool.map(load_gray, flow_x_files))
                ys = list(pool.map(load_gray, flow_y_files))
            
            # 클립 생성
            for i in range(num_clips):
                actual_clip_id = start_id + i
                start_idx = i * 150
                end_idx = min(start_idx + 150, len(flow_x_files))
                key_str = f'{exam_id}_{actual_clip_id:04d}'

                # if key_str in existing_keys:
                #     continue

                clip_data = np.full((150, 256, 256, 2), 128, dtype=np.uint8)
                valid = True
                
                # 데이터 로드
                for j, idx in enumerate(range(start_idx, end_idx)):
                    fx = xs[idx]
                    fy = ys[idx]
                    if fx is None or fy is None:
                        valid = False
                        break
                    target_size = (256, 256)
                    fx = cv2.resize(fx, target_size, interpolation=cv2.INTER_LINEAR)
                    fy = cv2.resize(fy, target_size, interpolation=cv2.INTER_LINEAR)
                    clip_data[j, :, :, 0] = fx
                    clip_data[j, :, :, 1] = fy
                
                if valid:
                    
                    # 결과 큐에 넣기 (큐가 꽉 차면 여기서 멈춤 -> 메모리 폭발 방지)
                    pickled = pickle.dumps(clip_data)
                    compressed = compressor.compress(pickled)
                    result_queue.put((task_idx, key_str, compressed))
                    
        except Exception as e:
            print(f"Error in {video_path}: {e}")
            continue

# -----------------------------------------------------------------------------
# 3. Main Process (Writer)
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # 경로 설정
    root_paths = glob.glob('/data2/flow*/*') + glob.glob('/jinyong_NAS/flows/*/*')
    
    # 태스크 생성 (Pass 1)
    if os.path.exists('/jinyong_NAS/tasks.pickle'):
        all_tasks = pickle.load(open('/jinyong_NAS/tasks.pickle', 'rb'))
    else:
        all_tasks = prepare_tasks(root_paths)
        with open('/jinyong_NAS/tasks.pickle', 'wb') as f:
            pickle.dump(all_tasks, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Total Tasks: {len(all_tasks)}")
    lmdb_path = '/data2/flow_dataset.lmdb'

    start_idx = max(0, load_progress() - 50)
    print(f"Resuming from task index {start_idx}")

    # 1) 이미 존재하는 키 로딩
    # existing_keys = load_existing_keys(lmdb_path)
    # print(f"Found {len(existing_keys)} existing clips.")
    # all_keys = set(existing_keys)

    # 큐 설정 (핵심: maxsize 제한)
    # result_queue 크기를 작게 잡아서 메모리 사용량을 강제로 제어합니다.
    # 예: 비디오 30개 분량 데이터만 대기 가능. 그 이상은 Worker가 멈춤.
    task_queue = mp.Queue(maxsize=1000)
    result_queue = mp.Queue(maxsize=30) 
    
    # Worker 실행
    num_workers = min(mp.cpu_count(), 12)  # 코어 수에 맞게 조절
    workers = []
    
    for _ in range(num_workers):
        p = mp.Process(target=worker_process, args=(task_queue, result_queue))
        p.start()
        workers.append(p)
    
    # 태스크 주입 (별도 스레드나 프로세스로 하면 좋지만, 단순화를 위해 미리 채움)
    # 태스크가 너무 많으면 Queue가 꽉 찰 수 있으니, 별도 Feeder Thread 사용 권장
    # 여기서는 간단하게 구현하기 위해 Feeder Thread 사용
    import threading
    def feeder():
        for i in range(start_idx, len(all_tasks)):
            task = all_tasks[i]
            task_queue.put((i,task))
        # 종료 신호
        for _ in range(num_workers):
            task_queue.put(None)
            
    feeder_thread = threading.Thread(target=feeder)
    feeder_thread.start()
    
    # LMDB 쓰기 루프
    env = lmdb.open('/data2/flow_dataset.lmdb', map_size=int(5e12))
    # all_keys = []
    
    # 전체 클립 수 예측 (Progress bar용)
    # 정확하진 않지만 태스크 수 * 평균 클립 수로 추정하거나 그냥 tqdm 없이 돌림
    # 여기서는 worker 종료를 감지하는 방식으로 진행
    
    finished_workers = 0
    count = 0
    
    with env.begin(write=True) as txn:
        pbar = tqdm(total=len(all_tasks) * 4, initial = start_idx*4) # 대략적인 진행바

        last_done_idx = start_idx - 1
        
        while True:
            # 타임아웃을 두어 데드락 방지
            try:
                # 큐에서 데이터를 하나 가져옴
                task_index, key_str, val_bytes = result_queue.get(timeout=1) 
                
                txn.put(key_str.encode('ascii'), val_bytes)
                # all_keys.append(key_str)
                count += 1
                pbar.update(1)

                if task_index > last_done_idx:
                    last_done_idx = task_index
                    # 너무 자주 쓰지 않도록 간단히 샘플링
                    if last_done_idx % 50 == 0:
                        save_progress(last_done_idx)
                
                if count % 500 == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                
                # if count % 50000 == 0:
                    # all_keys.sort()
                    # txn.put(b'__keys__', pickle.dumps(all_keys))
                    
            except mp.queues.Empty:
                # 큐가 비었을 때, 모든 워커가 죽었는지 확인
                alive_workers = sum(1 for p in workers if p.is_alive())
                if alive_workers == 0 and result_queue.empty():
                    break
        
        pbar.close()
        
        # 마무리 저장
        # all_keys.sort()
        # txn.put(b'__keys__', pickle.dumps(all_keys))
        
    env.close()
    
    feeder_thread.join()
    for p in workers:
        p.join()
        
    # print(f"Done. Saved {len(all_keys)} clips.")

