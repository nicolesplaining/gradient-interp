import fcntl,json,os,subprocess,time
from pathlib import Path
from gradient_interp.study import atomic_json,file_hash
repo=Path('/home/ubuntu/gradient-interp');meta=repo/'runs/analysis-tools/svd-parallel'
amend=meta/'execution-amendment.json';value=json.loads(amend.read_text())
with (meta/'tail-queue.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 target=meta/'tail-queue-state.json'
 if target.exists():raise ValueError('Tail queue already attempted; preserve it')
 state={'status':'running','pid':os.getpid(),'started_unix':time.time(),'amendment_sha256':file_hash(amend),'attempts':[]};atomic_json(target,state)
 for seed in value['execution_order_new_host']:
  assignment=next(a for a in value['assignments'] if a['seed']==seed)
  if file_hash(meta/'run_paper_svd_tail.py')!=value['helper_sha256']:raise ValueError('Frozen helper changed')
  attempt={'seed':seed,'started_unix':time.time()};state['attempts'].append(attempt);state['current_seed']=seed;atomic_json(target,state)
  with (meta/f'tail-seed{seed}.log').open('a') as log:
   result=subprocess.run(['/home/ubuntu/gradient-interp/.venv-atoms/bin/python','-u',str(meta/'run_paper_svd_tail.py'),'run','--root',assignment['original_root'],'--output',assignment['tail_root'],'--amendment',str(amend),'--preflight',str(meta/f'tail-preflight-seed{seed}.json')],cwd=repo,stdout=log,stderr=subprocess.STDOUT)
  attempt.update(returncode=result.returncode,completed_unix=time.time())
  if result.returncode:
   state.update(status='failed',completed_unix=time.time());atomic_json(target,state);raise RuntimeError('Tail helper failed; no automatic repair')
  subprocess.run(['/home/ubuntu/gradient-interp/.venv-atoms/bin/python',str(meta/'run_paper_svd_tail.py'),'receipt','--root',assignment['tail_root'],'--output',str(Path(assignment['tail_root'])/'tail-completion-receipt.json')],cwd=repo,check=True)
  atomic_json(target,state)
 state.update(status='complete',completed_unix=time.time());atomic_json(target,state)
