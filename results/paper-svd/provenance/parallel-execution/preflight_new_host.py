import hashlib
import json
import shutil
import time
from pathlib import Path

import torch

from gradient_interp.confirmation import verify_files
from gradient_interp.paper_svd_replication import preflight, validate_preflight
from gradient_interp.study import atomic_json, file_hash

repo = Path('/home/ubuntu/gradient-interp')
tools = repo / 'runs/analysis-tools/svd-parallel'
amendment_path = tools / 'execution-amendment.json'
amendment = json.loads(amendment_path.read_text())
torch.set_num_threads(8)
for assignment in amendment['assignments']:
    root = Path(assignment['original_root'])
    spec = json.loads((root / 'specification.json').read_text())
    plan = json.loads((root / 'plan.json').read_text())
    verify_files(repo, plan['source_files_sha256'])
    verify_files(repo, assignment['runtime_asset_hashes'])
    temp = repo / f"runs/paper-svd-third-preflight-seed{assignment['seed']}"
    if temp.exists():
        raise ValueError('Preserve any previous new-host preflight attempt')
    temp.mkdir()
    for name in ['specification.json', 'selection-lock.json', 'directions.pt', 'data.json',
                 'data-overlap-audit.json', 'candidates.json', 'selection.json', 'fit.json']:
        shutil.copyfile(root / name, temp / name)
    atomic_json(temp / 'hardware-preflight-plan.json', dict(
        written_unix=time.time(), execution_host=amendment['execution_host'],
        source_commit=amendment['frozen_source_commit'],
        amendment_sha256=file_hash(amendment_path),
        original_preflight_kernel_sha256=file_hash(repo / 'src/gradient_interp/paper_svd_replication.py'),
        training_input_only=True, scientific_outcomes_read=False))
    started = time.time()
    preflight(temp)
    record = json.loads((temp / 'preflight.json').read_text())
    validate_preflight(record, spec)
    record.update(execution_host=amendment['execution_host'],
                  amendment_sha256=file_hash(amendment_path),
                  helper_sha256=amendment['helper_sha256'],
                  hardware_preflight_started_unix=started,
                  torch_version=torch.__version__,
                  gpu=torch.cuda.get_device_name(0),
                  preflight_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target = tools / f"tail-preflight-seed{assignment['seed']}.json"
    if target.exists():
        raise ValueError('Preserve the previous bound hardware preflight')
    atomic_json(target, record)
    print(json.dumps(dict(seed=assignment['seed'], preflight='passed',
                         peak_gpu_bytes=record['peak_gpu_bytes'],
                         completed_unix=record['completed_unix'])), flush=True)
