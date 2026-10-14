import hashlib
import json
import time
from pathlib import Path
root=Path('/home/ubuntu/gradient-interp')
a=json.loads((root/'runs/analysis-tools/svd-parallel/execution-amendment.json').read_text())
hashes={}
for assignment in a['assignments']:
 hashes.update(assignment['runtime_asset_hashes'])
 source=Path(assignment['original_root'])
 hashes.update(json.loads((source/'plan.json').read_text())['source_files_sha256'])
 hashes.update({str(source/n):h for n,h in json.loads((source/'selection-lock.json').read_text())['files_sha256'].items()})
mismatches=[]
for name,expected in hashes.items():
 p=Path(name);h=hashlib.sha256()
 with p.open('rb') as f:
  while data:=f.read(8388608):h.update(data)
 if h.hexdigest()!=expected:mismatches.append(name)
r=dict(checked_unix=time.time(),verified_files=len(hashes),mismatches=mismatches,scientific_outputs_read=False,transfer_directory_metadata_warnings_only=not mismatches)
(root/'runs/analysis-tools/svd-parallel/staging-validation.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r));assert not mismatches
