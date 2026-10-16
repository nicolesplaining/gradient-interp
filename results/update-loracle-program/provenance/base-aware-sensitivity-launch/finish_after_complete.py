"""One-shot audit and analysis after the frozen worker normally completes; never retry."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

BASE = Path('/home/ubuntu/gradient-interp')
ROOT = BASE / 'runs/update-loracle-sensitivity-v4'
PYTHON = BASE / '.venv-atoms/bin/python'
PLAN_SHA = '3a4d4dfaa8391bd418c438fcc6e4c64acb033e3d9c54f78cee96c957c9a3c000'
SOURCE = 'f72323ce12f7871d0cf00d4c483d5f7791ffb61e'


def read(path):
    return json.loads(path.read_text())


def alive(pid):
    p = Path('/proc') / str(pid) / 'stat'
    return bool(pid and p.exists() and p.read_text().rsplit(')', 1)[-1].split()[0] != 'Z')


def record(stage, **extra):
    path = ROOT / 'audit-handoff.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps({'pid': os.getpid(), 'stage': stage,
                               'checked_unix': time.time(), 'automatic_retry': False,
                               **extra}, indent=2) + '\n')
    temp.replace(path)


def main():
    with (ROOT / 'audit-handoff.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (ROOT / 'audit-handoff.json').exists(), 'Never repeat this handoff'
        assert hashlib.sha256((ROOT / 'plan.json').read_bytes()).hexdigest() == PLAN_SHA
        record('waiting_for_normal_completion')
        while True:
            c = read(ROOT / 'controller.json')
            assert c['plan_sha256'] == PLAN_SHA
            if c['status'] == 'complete' and not alive(c['controller_pid']):
                assert c['worker_pid'] is None
                assert len(c['attempts']) == 1
                assert c['attempts'][0]['returncode'] == 0
                assert not alive(c['attempts'][0]['pid'])
                break
            if c['status'] not in ('running', 'complete') or time.time() > c['deadline_unix'] + 60:
                raise RuntimeError('No normal completion; no scientific outputs released')
            time.sleep(30)
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=BASE,
                                       text=True).strip() == SOURCE
        assert not (ROOT / 'integrity.json').exists() and not (ROOT / 'analysis').exists()
        env = os.environ.copy()
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'LOKY_MAX_CPU_COUNT'):
            env[key] = '8'
        for stage in ('audit', 'analyze'):
            record(stage)
            args = [str(PYTHON), '-u', '-m',
                    'gradient_interp.update_loracle_sensitivity_analysis',
                    stage, '--root', str(ROOT)]
            if stage == 'analyze':
                assert read(ROOT / 'integrity.json')['passed']
                args += ['--output', str(ROOT / 'analysis')]
            with (ROOT / f'{stage}.log').open('xb') as log:
                subprocess.run(args, cwd=BASE, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        record('complete', scientific_worker_never_resumed=True,
               audit_passed=read(ROOT / 'integrity.json')['passed'])


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        record('failed', error=repr(exc), no_automatic_retry=True)
        raise
