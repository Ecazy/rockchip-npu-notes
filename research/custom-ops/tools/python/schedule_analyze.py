#!/usr/bin/env python3
"""Fail closed: correlate controlled FIFO jobs, bound enqueue/ktime stamps."""
import collections, decimal, json, pathlib, re, statistics, sys
D=decimal.Decimal
LINE=re.compile(r'^\s*([0-9]+\.[0-9]+)\s*\|\s*(\d+)\)\s*(.*?)\s*\|([^|]*)\|(.*)$')
CALL=re.compile(r'([\w.]+)\(\)\s*([;{])')
IRQ=re.compile(r'rknpu_core([012])_irq_handler')
def require(condition,message):
    if not condition:raise ValueError(message)
def stamp(time,duration=0):return [time,time+1000+duration+1]
def analyze(records,trace,stats):
    require(records,'empty records');require(stats,'missing CPU trace stats')
    jobs={p['id']:dict(p) for p in records};require(len(jobs)==len(records),'duplicate IDs')
    for p in jobs.values():
        require(p['result']==0 and p['bad']==0,'failed numeric/device result')
        mask=p['actual_mask'];require(mask in (1,2,4),'invalid actual mask')
        require(p['requested_mask'] in (0,mask),'binding mismatch')
        require(mask & p['allowed_mask'],'outside allowed set')
        require(p['int_status']&0x300 and p['driver_service_ns']>0,'missing device evidence')
        require(p['host_begin_ns']<=p['ioctl_begin_ns']<=p['ioctl_end_ns']<=p['host_done_ns'],'host clock order')
        p['core']=mask.bit_length()-1
    for cpu,info in stats.items():
        for name in ['overrun','commit overrun','dropped events']:
            match=re.search(r'^'+name+r': (\d+)$',info,re.M)
            require(match and int(match[1])==0,f'{cpu} {name}: trace loss/unknown')
    stacks=collections.defaultdict(list);current={};markers=collections.defaultdict(dict);events=[];used_cpus=set()
    for line in trace.splitlines():
        m=LINE.match(line)
        if not m:
            require(not re.search(r'LOST|overrun|overflow',line,re.I),'trace overflow')
            continue
        time=int(D(m[1])*1000000000);cpu=int(m[2]);used_cpus.add(cpu);tid_match=re.search(r'-(\d+)\s*$',m[3]);require(tid_match,'bad trace task')
        tid=int(tid_match[1]);key=tid if tid else ('idle',cpu);stack=stacks[key];body=m[5].strip();depth=len(m[5])-len(m[5].lstrip())
        marker=re.search(r'RS (request|ioctl|return|done) id=(\d+) tid=(\d+) mask=(\d+) mono=(\d+)',body)
        if marker:
            name,id,mtid,mask,mono=marker.groups();id=int(id);mono=int(mono)
            require(id in jobs and int(mtid)==tid==jobs[id]['tid'] and int(mask)==jobs[id]['requested_mask'],'foreign/mismatched marker')
            require(name not in markers[id],'duplicate marker');markers[id][name]=mono
            require(time+1000>=mono,'trace/host clocks differ')
            if name=='ioctl':require(tid not in current,'overlapping request on TID');current[tid]=id
            if name=='return':require(current.pop(tid,None)==id,'unmatched return')
            continue
        end=re.match(r'} /\* ([\w.]+) \*/',body)
        if end:
            require(stack and stack[-1]['name']==end[1],f'unbalanced graph {line}')
            node=stack.pop()
            if node['name']=='rknpu_job_schedule':require(node.get('count')==1,'ambiguous enqueue')
            if IRQ.fullmatch(node['name']):require(node.get('completion')==1 and node.get('task_number_calls')==2,'unexpected/intermediate IRQ')
            continue
        call=CALL.fullmatch(body)
        require(call is not None,f'unparsed graph line: {line}')
        name,kind=call.groups();node={'name':name,'time':time,'depth':depth}
        if name=='rknpu_submit_ioctl':require(tid in current,'unobserved submit');node['id']=current[tid];p=jobs[node['id']];require('driver_entry_ns' not in p,'duplicate submit');p['driver_entry_ns']=stamp(time)
        if name=='rknpu_job_schedule':
            require(tid in current,'unobserved schedule');node['id']=current[tid];jobs[node['id']]['enqueue_ns']=[time,None]
        if name=='rknpu_get_task_number' and stack and stack[-1]['name']=='rknpu_job_schedule':
            parent=stack[-1];parent['count']=parent.get('count',0)+1;id=parent['id'];p=jobs[id]
            require(p['enqueue_ns'][1] is None,'multiple enqueue bounds');p['enqueue_ns'][1]=time+1000
            events.append((time,'enqueue',p['core'],id,stamp(time),(cpu,tid)))
        if name=='rknpu_get_task_number' and stack and IRQ.fullmatch(stack[-1]['name']) and depth==stack[-1]['depth']+2:
            stack[-1]['task_number_calls']=stack[-1].get('task_number_calls',0)+1
        if name=='ktime_get' and stack and depth==stack[-1]['depth']+2:
            duration=re.search(r'([0-9.]+) us',m[4]);require(kind==';' and duration,'ktime leaf required')
            interval=stamp(time,int(D(duration[1])*1000))
            if stack[-1]['name']=='rknpu_job_next.part.0':
                require(not stack[-1].get('commit'),'multiple direct commit stamps');stack[-1]['commit']=1
                irq=next((IRQ.fullmatch(n['name']) for n in reversed(stack) if IRQ.fullmatch(n['name'])),None)
                core=int(irq[1]) if irq else jobs[current[tid]]['core']
                events.append((time,'start',core,None,interval,(cpu,tid)))
            elif IRQ.fullmatch(stack[-1]['name']):
                parent=stack[-1];parent['completion']=parent.get('completion',0)+1
                events.append((time,'complete',int(IRQ.fullmatch(parent['name'])[1]),None,interval,(cpu,tid)))
        if kind=='{':stack.append(node)
    require(all(f'cpu{cpu}' in stats for cpu in used_cpus),'missing traced CPU stats')
    require(not current and all(not s for s in stacks.values()),'truncated graph')
    pending=[collections.deque() for _ in range(3)];active=[None]*3;last_enqueue=[None]*3;last_event=[None]*3
    # Same CPU/TID preserves program order; equal timestamps across contexts are ambiguous.
    for time,kind,core,id,interval,context in sorted(events,key=lambda e:e[0]):
        require(last_event[core] is None or time>last_event[core][0] or context==last_event[core][1],f'ambiguous event order core{core}')
        last_event[core]=(time,context)
        if kind=='enqueue':
            require(last_enqueue[core] is None or time>last_enqueue[core],'ambiguous FIFO insertion order');last_enqueue[core]=time;pending[core].append(id)
        elif kind=='start':
            require(active[core] is None and pending[core],f'unmatched start at {time} core{core}, active={active[core]}, pending={list(pending[core])}');id=pending[core].popleft();active[core]=id;jobs[id]['start_ns']=interval
        else:
            require(active[core] is not None,'unmatched completion IRQ');id=active[core];active[core]=None;jobs[id]['completion_ns']=interval;jobs[id]['irq_core']=core
    require(not any(pending) and not any(x is not None for x in active),'unfinished jobs')
    for id,p in jobs.items():
        require(markers[id]=={'request':p['host_begin_ns'],'ioctl':p['ioctl_begin_ns'],'return':p['ioctl_end_ns'],'done':p['host_done_ns']},'missing/mismatched markers')
        require('start_ns' in p and 'completion_ns' in p,'missing timing')
        a,b=p['start_ns'];c,d=p['completion_ns'];service=p['driver_service_ns']
        require(c-b<=service<=d-a,'driver service inconsistent with trace clock/correlation')
        # Exact driver duration tightens two enclosing intervals without claiming exact stamps.
        a=max(a,c-service);b=min(b,d-service);c=max(c,a+service);d=min(d,b+service)
        p['start_ns']=[a,b];p['completion_ns']=[c,d]
        e,f=p['enqueue_ns']
        p['pre_submit_ns']=p['ioctl_begin_ns']-p['host_begin_ns']
        l,h=p['driver_entry_ns'];p['host_to_driver_ns']=[l-p['ioctl_begin_ns'],h-p['ioctl_begin_ns']]
        p['driver_admission_ns']=[max(0,e-h),max(0,f-l)]
        p['queue_ns']=[max(0,a-f),max(0,b-e)]
        require(d<=p['ioctl_end_ns']+1000,'completion after host observation')
        p['host_observed_ns']=p['ioctl_end_ns']
        response=p['host_observed_ns']-p['host_begin_ns'];p['response_ns']=response
        p['api_response_ns']=p['host_done_ns']-p['host_begin_ns']
        qlo,qhi=p['queue_ns'];p['other_ns']=[response-service-qhi,response-service-qlo]
        require(p['other_ns'][0]>=0,'negative other')
        p['completion_to_host_ns']=[max(0,p['ioctl_end_ns']-d),max(0,p['ioctl_end_ns']-c)]
    return sorted(jobs.values(),key=lambda p:p['id'])
def summary(rows):
    return dict(requests=len(rows),tasks=sum(p['task_count'] for p in rows),cores=sorted(set(p['irq_core'] for p in rows)),
        positive_queue_lower=sum(p['queue_ns'][0]>0 for p in rows),
        queue_lower_median_us=statistics.median(p['queue_ns'][0]/1000 for p in rows),
        queue_upper_median_us=statistics.median(p['queue_ns'][1]/1000 for p in rows),
        max_queue_lower_us=max(p['queue_ns'][0]/1000 for p in rows),
        queue_width_max_us=max((p['queue_ns'][1]-p['queue_ns'][0])/1000 for p in rows),
        service_median_us=statistics.median(p['driver_service_ns']/1000 for p in rows),
        response_median_us=statistics.median(p['response_ns']/1000 for p in rows))
def main(path):
    root=pathlib.Path(path);result={}
    from schedule_kernel_audit import DIGEST
    require(json.loads((root/'kernel-audit.json').read_text())['code_sha256']==DIGEST,'unaudited kernel')
    for scenario in ['serial','same','split','candidates','auto']:
        rows=analyze([json.loads(l) for l in (root/f'{scenario}.jsonl').read_text().splitlines()],(root/f'{scenario}.trace').read_text(),json.loads((root/f'{scenario}.stats.json').read_text()))
        expected_workers={'serial':1,'same':4,'split':3,'candidates':4,'auto':4}[scenario]
        expected_tasks=1 if scenario=='serial' else 64
        require(len(rows)==expected_workers*6,'incomplete campaign requests')
        require({(p['worker'],p['round']) for p in rows}=={(w,r) for w in range(expected_workers) for r in range(6)},'missing/duplicate worker round')
        for p in rows:
            require(p['scenario']==scenario and p['task_count']==expected_tasks,'wrong workload')
            expected_mask={'serial':1<<(p['round']%3),'same':1,'split':1<<(p['worker']%3),'candidates':1<<((p['worker']+p['round'])%2),'auto':0}[scenario]
            require(p['requested_mask']==expected_mask,'wrong requested routing policy')
        (root/f'{scenario}.profile.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in rows));result[scenario]=summary(rows)
        baseline=root/f'{scenario}.baseline.jsonl'
        if baseline.exists():
            base=[json.loads(l) for l in baseline.read_text().splitlines()]
            require(len(base)==len(rows) and all(p['result']==0 and p['bad']==0 for p in base),'baseline failed')
            result[scenario]['untraced_service_median_us']=statistics.median(p['driver_service_ns']/1000 for p in base)
            result[scenario]['untraced_response_median_us']=statistics.median((p['ioctl_end_ns']-p['host_begin_ns'])/1000 for p in base)
    require(result['serial']['cores']==[0,1,2],'missing strict core coverage')
    require(result['same']['max_queue_lower_us']>100,'failed to observe actual queueing')
    require(result['same']['cores']==[0] and result['split']['cores']==[0,1,2] and result['candidates']['cores']==[0,1],'routing coverage mismatch')
    if (root/'restored-options.json').exists():
        require(json.loads((root/'saved-options.json').read_text())==json.loads((root/'restored-options.json').read_text()),'trace options not restored')
    require(json.loads((root/'saved-state.json').read_text())==json.loads((root/'restored-state.json').read_text()),'tracing state not restored')
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('SCHEDULING_EVIDENCE PASS')
if __name__=='__main__':main(sys.argv[1])
