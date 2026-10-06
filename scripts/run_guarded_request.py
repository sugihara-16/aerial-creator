#!/usr/bin/env python3
"""Run a request collection/update in an owned, bounded systemd cgroup.

Example: python scripts/run_guarded_request.py --output artifacts/run/guard \
  --cpu-quota 1200 --memory-gib 44 --wall-seconds 3600 -- \
  /path/to/isaaclab/python scripts/run_request_collection.py ...
No automatic retry, hardware-limit change, or training restart is performed.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import grp
import json
import os
import pwd
import shutil
import signal
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
LIMITS = dict(cpu_c=85., gpu_c=75., gpu_memory_mib=20000., ram_min_gib=8.,
              disk_min_gib=40., gpu_power_limit_w=250.)
MAX_CPU_CORES = 20  # Host has 24 physical cores; retain capacity for the desktop.


def sample(output):
    temperatures=[]
    for directory in Path('/sys/class/hwmon').glob('hwmon*'):
        if (directory/'name').read_text().strip() == 'coretemp':
            temperatures.extend(float(p.read_text())/1000 for p in directory.glob('temp*_input'))
    if not temperatures:
        raise RuntimeError('CPU temperature unavailable')
    gpu=subprocess.check_output(['nvidia-smi','-i','0',
        '--query-gpu=temperature.gpu,memory.used,power.limit,power.draw',
        '--format=csv,noheader,nounits'],text=True,timeout=5).strip().split(',')
    return dict(utc=datetime.now(timezone.utc).isoformat(),cpu_c=max(temperatures),
        gpu_c=float(gpu[0]),gpu_memory_mib=float(gpu[1]),gpu_power_limit_w=float(gpu[2]),gpu_power_w=float(gpu[3]),
        ram_available_gib=next(int(line.split()[1])/1024**2 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')),
        disk_free_gib=shutil.disk_usage(output).free/1024**3,
        kernel_taint=int(Path('/proc/sys/kernel/tainted').read_text()),
        no_turbo=int(Path('/sys/devices/system/cpu/intel_pstate/no_turbo').read_text()))


def trip_reason(value):
    if value['kernel_taint'] & ((1<<7)|(1<<4)):
        return 'kernel Oops or machine-check taint'
    for name in ('cpu_c','gpu_c','gpu_memory_mib'):
        if value[name] >= LIMITS[name]:
            return name+' threshold'
    if value['ram_available_gib'] < LIMITS['ram_min_gib']:
        return 'RAM floor reached'
    if value['disk_free_gib'] < LIMITS['disk_min_gib']:
        return 'disk floor reached'
    if value['no_turbo'] != 1 or value['gpu_power_limit_w'] > LIMITS['gpu_power_limit_w']+.1:
        return 'existing host power/frequency limits changed'
    return None


def require_bounded_worker_environment():
    """Reject an accidentally unbounded multi-process training launch."""
    unit=os.environ.get('AMSRR_RESOURCE_GUARD_ACTIVE')
    if not unit:
        raise RuntimeError('parallel simulation requires scripts/run_guarded_request.py')
    group=next(line.split(':',2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    if Path(group).name != unit:
        raise RuntimeError('resource guard does not own the collection cgroup')
    root=Path('/sys/fs/cgroup')/group.lstrip('/')
    quota,period=(root/'cpu.max').read_text().split()
    memory=(root/'memory.max').read_text().strip()
    if (quota == 'max' or int(quota)/int(period)>MAX_CPU_CORES or memory=='max'
            or int(memory)>44*1024**3 or (root/'memory.swap.max').read_text().strip()!='0'):
        raise RuntimeError('parallel worker cgroup is not safely bounded')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu-quota',type=int,default=200,help='Percent of one CPU; aggregate over all children')
    parser.add_argument('--cpu-period-ms',type=int,choices=(10,100),default=100,
                        help='Quota accounting period; aggregate CPU limit is unchanged')
    parser.add_argument('--memory-gib',type=int,default=16)
    parser.add_argument('--wall-seconds',type=int,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command or not 100 <= args.cpu_quota <= MAX_CPU_CORES*100 or not 12 <= args.memory_gib <= 44 or args.wall_seconds <= 0:
        parser.error('invalid command or bounded resource budget')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'guard_result.json').exists() or (output/'process.log').exists():
        raise FileExistsError('guard output already contains a run')
    first=sample(output)
    reason=trip_reason(first)
    if reason:
        raise RuntimeError(reason)
    unit=f'amsrr-request-{time.time_ns()}.service'
    user=pwd.getpwuid(os.getuid()).pw_name
    group=grp.getgrgid(os.getgid()).gr_name
    arguments=['sudo','-n','systemd-run','--uid='+user,'--gid='+group,'--unit='+unit,
        '--wait','--collect','--pipe','-p',f'CPUQuota={args.cpu_quota}%',
        '-p',f'CPUQuotaPeriodSec={args.cpu_period_ms}ms',
        '-p',f'MemoryHigh={args.memory_gib-4}G','-p',f'MemoryMax={args.memory_gib}G',
        '-p','MemorySwapMax=0','-p','TasksMax=2048','-p','Nice=10','-p','LimitCORE=0',
        '-p','KillMode=control-group','-p','TimeoutStopSec=15','-p','Restart=no',
        '-p',f'RuntimeMaxSec={args.wall_seconds}','--working-directory='+str(REPO),
        '/usr/bin/env','PYTHONPATH='+str(REPO),'OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',
        'MKL_NUM_THREADS=1','TORCHINDUCTOR_COMPILE_THREADS=1','OMNI_KIT_ACCEPT_EULA=YES',
        'AMSRR_RESOURCE_GUARD_ACTIVE='+unit,*command]
    def write(name,value):
        (output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    write('guard_binding.json',dict(command=command,unit=unit,limits=LIMITS,
        cpu_quota=args.cpu_quota,cpu_period_ms=args.cpu_period_ms,
        memory_gib=args.memory_gib,wall_seconds=args.wall_seconds,
        first_sample=first,automatic_retry=False))
    def interrupt(*_):
        raise RuntimeError('guard interrupted')
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,interrupt)
    started=time.monotonic();epoch=time.time();process=None;error=None;code=1
    next_journal=0.
    try:
        with (output/'process.log').open('x') as log:
            process=subprocess.Popen(arguments,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                value=sample(output);reason=trip_reason(value)
                cgroup=Path('/sys/fs/cgroup/system.slice')/unit
                if cgroup.exists():
                    try:
                        stats=dict(line.split() for line in (cgroup/'memory.stat').read_text().splitlines())
                        value['cgroup_memory_gib'] = int((cgroup/'memory.current').read_text())/1024**3
                        value['cgroup_cpu_max'] = (cgroup/'cpu.max').read_text().strip()
                        value['cgroup_memory_peak_gib'] = int((cgroup/'memory.peak').read_text())/1024**3
                        value['cgroup_anon_gib'] = int(stats['anon'])/1024**3
                        value['cgroup_file_gib'] = int(stats['file'])/1024**3
                        value['cgroup_memory_events'] = {k:int(v) for k,v in
                            (line.split() for line in (cgroup/'memory.events').read_text().splitlines())}
                    except FileNotFoundError:
                        pass  # systemd collected a just-completed owned unit.
                if time.monotonic() >= next_journal:
                    journal=subprocess.check_output(['journalctl','-b','-k','--no-pager',
                        '--since','@'+str(int(epoch)),'-o','cat'],text=True,timeout=5)
                    markers=('BUG:','Oops:','NVRM: Xid','Hardware Error','Out of memory','oom-kill')
                    errors=[line for line in journal.splitlines() if any(marker in line for marker in markers)]
                    if errors:
                        reason='new kernel/GPU error';value['kernel_errors']=errors[-10:]
                    next_journal=time.monotonic()+15
                value['trip']=reason
                with (output/'health.jsonl').open('a') as health:
                    health.write(json.dumps(value)+'\n')
                write('health.json',value)
                if reason:
                    raise RuntimeError(reason)
                if time.monotonic()-started >= args.wall_seconds:
                    raise TimeoutError('declared wall budget reached')
                time.sleep(5)
            code=process.returncode
    except BaseException as exc:
        error=repr(exc)
        raise
    finally:
        subprocess.run(['sudo','-n','systemctl','stop',unit],stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,timeout=35)
        if process is not None and process.poll() is None:
            process.wait(timeout=35)
        write('guard_result.json',dict(exit_code=code,error=error,seconds=time.monotonic()-started,
            command=command,unit=unit,automatic_retry=False))
    return code


if __name__=='__main__':
    raise SystemExit(main())
