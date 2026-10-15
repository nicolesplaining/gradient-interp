from pathlib import Path
import json,hashlib,time
r=Path('/home/ubuntu/gradient-interp/runs/loracle-trajectory-short-prompt')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
c=read(r/'controller.json')
assert c['status']=='complete', 'Wait for completion'
p=Path(f"/proc/{c['pid']}/stat")
assert not p.exists() or p.read_text().split(') ')[-1][0]=='Z', 'Worker still alive'
m=read(r/'manifest.json');plan=read(r/'plan.json')
assert c['plan_sha256']==m['plan_sha256']==sha(r/'plan.json')
assert m['worker_sha256']==plan['worker_sha256']==sha(r/'run_loracle_short_prompt.py')
assert m['finished_unix']<=c['finished_unix']<=c['deadline_unix']
assert m['response_count']==10 and m['responses_sha256']==sha(r/'responses.json')
assert m['original_inputs_unchanged']
old=Path(plan['original_root']);lock=read(old/'predictions-lock.json')
for name,digest in lock['token_sha256'].items():assert sha(old/'tokens'/name)==digest
for name in ['controller.json','integrity.json','predictions-lock.json','training-manifest.json','manifest.json']:
 assert sha(old/name)==plan['input_files_sha256'][str(old/name)]
rows=read(r/'responses.json')
assert len(rows)==10
assert {(x['case'],x['encoding']) for x in rows}=={(a,b) for a in plan['cases'] for b in plan['encodings']}
assert all(x['question']==plan['question'] and isinstance(x['response'],str) for x in rows)
a={'passed':True,'checked_unix':time.time(),'worker_complete_and_dead':True,'all10responses':True,'unchanged_original_tokens':True,'within_original_deadline':True,'files_sha256':{n:sha(r/n) for n in ['plan.json','manifest.json','controller.json','launch.json','responses.json','run_loracle_short_prompt.py']}}
(r/'integrity.json').write_text(json.dumps(a,indent=2)+'\n')
print(json.dumps({k:v for k,v in a.items() if k!='files_sha256'}))
