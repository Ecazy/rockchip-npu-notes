#!/usr/bin/env python3
"""Exclusive, scoped function-graph capture on the matching RK3588 kernel."""
import fcntl, json, os, pathlib, re, signal, subprocess, sys
from schedule_kernel_audit import audit
T=pathlib.Path('/sys/kernel/tracing')
FUNCTIONS=['rknpu_submit_ioctl','rknpu_job_schedule','rknpu_job_next.part.0','rknpu_get_task_number','rknpu_core0_irq_handler','rknpu_core1_irq_handler','rknpu_core2_irq_handler','ktime_get']
ROOTS=[FUNCTIONS[0]]+FUNCTIONS[4:7]
def read(name):return (T/name).read_text()
def write(name,value): (T/name).write_text(str(value)+'\n')
def interrupted(signum,frame):raise RuntimeError(f'interrupted by signal {signum}')
def main():
    binary=pathlib.Path(sys.argv[1]).resolve(); out=pathlib.Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False)
    with open('/tmp/rk3588-custom-ops.lock','w') as lock, open('/tmp/rknpu-global-trace.lock','w') as trace_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(trace_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        audit(out)
        if read('current_tracer').strip()!='nop' or read('events/enable').strip()!='0':raise RuntimeError('global tracing already in use')
        available=set(read('available_filter_functions').splitlines())
        missing=[f for f in FUNCTIONS if not any(line.split()[0]==f for line in available)]
        if missing:raise RuntimeError(f'missing functions {missing}')
        saved={k:read(k) for k in ['current_tracer','tracing_on','trace_clock','buffer_size_kb','set_ftrace_filter','set_graph_function']}
        (out/'previous-trace.txt').write_text(read('trace'))
        (out/'saved-state.json').write_text(json.dumps(saved,indent=2))
        options=[]
        for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,interrupted)
        try:
            write('tracing_on',0);write('current_tracer','function_graph')
            options=read('trace_options').splitlines()
            (out/'saved-options.json').write_text(json.dumps(options))
            write('trace_clock','mono');write('buffer_size_kb',8192)
            write('set_ftrace_filter','\n'.join(FUNCTIONS));write('set_graph_function','\n'.join(ROOTS))
            for opt in ['funcgraph-abstime','funcgraph-proc','funcgraph-cpu','funcgraph-duration','funcgraph-tail','funcgraph-irqs','sleep-time']:
                write('trace_options',opt)
            (out/'capture-options.json').write_text(json.dumps(read('trace_options').splitlines()))
            specs=[('serial',1,6,1),('same',4,6,64),('split',3,6,64),('candidates',4,6,64),('auto',4,6,64)]
            for scenario,workers,rounds,tasks in specs:
                write('tracing_on',0);write('trace','')
                env=os.environ.copy();env.update(RS_PROFILE_JSON=str(out/f'{scenario}.jsonl'),RS_TRACE_MARKER=str(T/'trace_marker'))
                write('tracing_on',1)
                try:
                    result=subprocess.run([str(binary),scenario,str(workers),str(rounds),str(tasks)],env=env,capture_output=True,text=True,timeout=90)
                finally:write('tracing_on',0)
                (out/f'{scenario}.trace').write_text(read('trace'))
                stats={p.parent.name:p.read_text() for p in (T/'per_cpu').glob('cpu*/stats')}
                (out/f'{scenario}.stats.json').write_text(json.dumps(stats,indent=2))
                (out/f'{scenario}.log').write_text(result.stdout+result.stderr)
                print(result.stdout.strip(),flush=True)
                if result.returncode:raise RuntimeError(f'{scenario} exit={result.returncode}')
                env.update(RS_PROFILE_JSON=str(out/f'{scenario}.baseline.jsonl'),RS_TRACE_MARKER='/dev/null')
                write('current_tracer','nop')
                try:
                    baseline=subprocess.run([str(binary),scenario,str(workers),str(rounds),str(tasks)],env=env,capture_output=True,text=True,timeout=90)
                finally:write('current_tracer','function_graph')
                (out/f'{scenario}.baseline.log').write_text(baseline.stdout+baseline.stderr)
                if baseline.returncode:raise RuntimeError(f'{scenario} baseline exit={baseline.returncode}')
            (out/'system.txt').write_text(subprocess.check_output(['uname','-a'],text=True))
        finally:
            write('tracing_on',0)
            for opt in options:write('trace_options',opt)
            (out/'restored-options.json').write_text(json.dumps(read('trace_options').splitlines()))
            write('current_tracer','nop')
            for name in ['set_ftrace_filter','set_graph_function']:
                write(name,'\n'.join(l for l in saved[name].splitlines() if not l.startswith('#')))
            write('buffer_size_kb',saved['buffer_size_kb'].strip())
            write('trace_clock',re.search(r'\[([^]]+)\]',saved['trace_clock'])[1])
            write('tracing_on',saved['tracing_on'].strip())
            (out/'restored-state.json').write_text(json.dumps({k:read(k) for k in saved},indent=2))
if __name__=='__main__':main()
