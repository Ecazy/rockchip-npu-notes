#!/usr/bin/env python3
"""v2 executable plans; historical v1 draft retained in handoff evidence."""
import json
from device_cost_analyze import integer, program_features, require, validate_run
SCHEMA_VERSION = 2

def validate_config(config):
    require(integer(config.get('schema_version')) and config['schema_version'] == 2, 'schema version')
    require(integer(config.get('blocks')) and config['blocks'] >= 1, 'block budget')
    require(isinstance(config.get('cells'), list) and config['cells'], 'empty matrix')
    seen=set()
    for c in config['cells']:
        require(isinstance(c.get('id'), str) and c['id'] not in seen, 'missing/duplicate ID');seen.add(c['id'])
        require(c.get('workload') in ('lut','matmul','mm_bias_hsig','lut+matmul','matmul+lut'), 'workload')
        require(len(c['dims'])==3 and all(integer(x) and 1<=x<=256 for x in c['dims']), 'geometry')
        for name,lo,hi in [('programs',1,64),('workers',1,8),('operations',1,100000),('warmup',0,10000)]:
            require(integer(c.get(name)) and lo<=c[name]<=hi, f'invalid {name}')
        require(c['operations']>=c['workers'], 'zero worker quota')
        require(c['strategy'] in ('fixed','split','poll','auto'), 'strategy')
        require(integer(c['mask']) and c['mask'] in (1,2,4), 'fixed mask')
        require(integer(c['allowed']) and 1<=c['allowed']<=7, 'candidate set')
        require(c['strategy']!='auto' or c['allowed']==7, 'vendor AUTO cannot restrict candidate cores')
        require(c['mode'] in ('chain','separate') and c['path'] in ('device_ready','host_io'), 'execution path')
        require(c.get('capture','immediate') in ('immediate','deferred'), 'verification placement')
        require(c.get('capture') != 'deferred' or (c['programs']==1 and c['mode']=='chain' and c['path']=='device_ready' and '+' not in c['workload']), 'unsupported deferred capture')
        if c.get('capture')=='deferred':
            m,k,n=c['dims'];y=((max(m,12)+3)//4*4)*((n+31)//32*32)*2
            count=len(program_features(c));buffers=3 if c['workload']=='mm_bias_hsig' else 1
            memory=(c['operations']+c['warmup']*c['workers'])*(buffers*(y+32768+64+4096)+count*17000)
            require(memory<=1073741824,'deferred batch exceeds 1 GiB memory budget')
        require(c['start'] in ('continuous','burst'), 'start policy')
        require(c['start']!='burst' or c['operations']%c['workers']==0, 'unequal burst quota')
        require(c['background'] is None or c['background'] in ('compute:1','compute:2','compute:4','stream:1','stream:2','stream:4'), 'background')
        require(integer(c['timeout_s']) and 1<=c['timeout_s']<=3600, 'timeout')
        require(integer(c['seed']) and 0<=c['seed']<=0xffffffff, 'seed')
        require(integer(c['affinity_base']) and -1<=c['affinity_base']<=7, 'CPU affinity')
        require(isinstance(c['trace'],bool), 'trace flag')
        require(not c['trace'] or not c['background'], 'trace background')
        for wi in range(c['workers']):require(1<=len(program_features(c,wi))<=64, 'program count')
    return config

def enumerate_config(config):
    validate_config(config)
    counts=dict(measured_operations=0,measured_submits=0,measured_tasks=0,warmup_operations=0,warmup_submits=0,warmup_tasks=0)
    for c in config['cells']:
        for wi in range(c['workers']):
            budget=c['operations']//c['workers']+(wi<c['operations']%c['workers'])
            tasks=len(program_features(c,wi));submits=1 if c['mode']=='chain' else tasks
            for prefix,n in [('measured',budget),('warmup',c['warmup'])]:
                counts[prefix+'_operations']+=n;counts[prefix+'_submits']+=n*submits;counts[prefix+'_tasks']+=n*tasks
    return {k:v*config['blocks'] for k,v in counts.items()}

def load_json(path):
    def reject(value):raise ValueError('nonfinite JSON number '+value)
    with open(path) as file:return json.load(file,parse_constant=reject)
