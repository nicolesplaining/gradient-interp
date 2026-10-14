import fcntl
import json
import os
import subprocess
import time
from pathlib import Path
from gradient_interp.study import atomic_json,file_hash
repo=Path('/home/ubuntu/gradient-interp'); prefix=repo/'runs/paper-svd-seed42'; root=repo/'runs/paper-magnitude-seed42'; meta=repo/'runs/analysis-tools/svd-parallel'
expected='946cb3ae803d67b0426ae1bb3b4ac7fc1b6c2afae5b89e654030d31ce928f590'
with (meta/'magnitude42-continuation.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (meta/'magnitude42-continuation.json').exists():raise ValueError('Continuation already attempted')
 state={'status':'waiting_for_prefix','pid':os.getpid(),'started_unix':time.time(),'original_magnitude_plan_sha256':expected,'execution_only_queue_amendment':True}
 atomic_json(meta/'magnitude42-continuation.json',state)
 deadline=json.loads((meta/'execution-amendment.json').read_text())['assignments'][0]['original_deadline_unix']
 while not (prefix/'prefix-completion-receipt.json').exists():
  if time.time()>=deadline:raise TimeoutError('Original prefix deadline reached')
  time.sleep(2)
 receipt=json.loads((prefix/'prefix-completion-receipt.json').read_text())
 if not receipt['all_processes_dead'] or receipt['seed']!=42:raise ValueError('Invalid prefix death receipt')
 if file_hash(root/'plan.json')!=expected or (root/'controller.json').exists():raise ValueError('Magnitude controller exists or original plan changed')
 state.update(status='launching',launched_unix=time.time());atomic_json(meta/'magnitude42-continuation.json',state)
 with (meta/'magnitude42-continuation-worker.log').open('a') as log:
  result=subprocess.run(['/home/ubuntu/gradient-interp/.venv-atoms/bin/python','-u','-m','gradient_interp.development_run','--root',str(root)],cwd=repo,stdout=log,stderr=subprocess.STDOUT)
 controller=json.loads((root/'controller.json').read_text())
 state.update(status=controller['status'],returncode=result.returncode,completed_unix=time.time(),controller_sha256=file_hash(root/'controller.json'))
 atomic_json(meta/'magnitude42-continuation.json',state)
 if result.returncode or state['status']!='complete':raise RuntimeError('Original magnitude42 controller did not complete')
