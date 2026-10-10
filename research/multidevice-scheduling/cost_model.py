#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Explicit residency/admission and conservative Host-staged migration costs."""
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
from dataset import JobCostDataset
from simulator import transfer_ns, validate_scenario, simulate_all

def nonnegative(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be non-negative integer bytes/ns')
    return value

@dataclass(frozen=True)
class CacheIdentity:
    device_id: str
    model_sha256: str
    layout: str
    dtype: str
    runtime_version: str
    state_generation: int
    def __post_init__(self):
        if not all(type(x) is str and x for x in (self.device_id, self.model_sha256,
                                                 self.layout, self.dtype, self.runtime_version)):
            raise ValueError('cache identity fields must be explicit')
        nonnegative(self.state_generation, 'state generation')

@dataclass(frozen=True)
class LinkCost:
    bandwidth_bytes_per_s: int
    latency_ns: int
    evidence: str = 'configured'
    def duration(self, size):
        return transfer_ns(size, self.bandwidth_bytes_per_s, self.latency_ns)
    def __post_init__(self):
        self.duration(0)
        if self.evidence not in ('configured', 'measured'):
            raise ValueError('link evidence must be explicit')

@dataclass(frozen=True)
class MemoryBudget:
    capacity_bytes: int
    resident_bytes: int = 0
    live_tensor_bytes: int = 0
    workspace_bytes: int = 0
    output_bytes: int = 0
    staging_bytes: int = 0
    def required(self):
        for key, value in self.__dict__.items():
            nonnegative(value, key)
        return self.resident_bytes + self.live_tensor_bytes + self.workspace_bytes + self.output_bytes + self.staging_bytes
    def admitted(self):
        return self.required() <= self.capacity_bytes

@dataclass(frozen=True)
class HostStagedMigration:
    download: LinkCost
    upload: LinkCost
    host_copy_bandwidth_bytes_per_s: int
    setup_ns: int
    layout_ns: int | None = None
    identical_layout: bool = False
    def estimate(self, download_bytes, upload_bytes=None, host_copy_bytes=None):
        upload_bytes = download_bytes if upload_bytes is None else upload_bytes
        host_copy_bytes = download_bytes if host_copy_bytes is None else host_copy_bytes
        nonnegative(self.setup_ns, 'setup')
        nonnegative(download_bytes, 'download bytes'); nonnegative(upload_bytes, 'upload bytes')
        nonnegative(host_copy_bytes, 'host copy bytes')
        if self.layout_ns is None and not self.identical_layout:
            raise ValueError('layout conversion cost unknown')
        if self.identical_layout and download_bytes != upload_bytes:
            raise ValueError('identical layout requires equal transfer byte counts')
        layout = 0 if self.layout_ns is None else nonnegative(self.layout_ns, 'layout cost')
        phases = {'download_ns':self.download.duration(download_bytes),
                  'host_copy_ns':transfer_ns(host_copy_bytes, self.host_copy_bandwidth_bytes_per_s, 0),
                  'layout_ns':layout, 'upload_ns':self.upload.duration(upload_bytes), 'setup_ns':self.setup_ns}
        return {'duration_ns':sum(phases.values()), 'phases':phases,
                'bytes':{'download':download_bytes, 'upload':upload_bytes, 'host_copy':host_copy_bytes},
                'scope':'conservative serial Host staging; resource queueing external',
                'evidence':'configured' if 'configured' in (self.download.evidence,self.upload.evidence)
                           else 'mixed link measurements and configured Host overhead'}

def migration_decision(*, source_queue_ns, source_service_ns, target_queue_ns,
                       target_service_ns, migration, source_budget, target_budget):
    for name, value in [('source queue',source_queue_ns),('source service',source_service_ns),
                        ('target queue',target_queue_ns),('target service',target_service_ns)]:
        nonnegative(value,name)
    transfer = nonnegative(migration['duration_ns'],'migration')
    local = source_queue_ns + source_service_ns
    remote = target_queue_ns + transfer + target_service_ns
    fits = source_budget.admitted() and target_budget.admitted()
    return {'migrate':fits and remote < local, 'source_finish_ns':local, 'target_finish_ns':remote,
            'savings_ns':local-remote, 'source_required_bytes':source_budget.required(),
            'target_required_bytes':target_budget.required(), 'both_memory_budgets_fit':fits,
            'scope':'serial target queue + transfer + service; conservative, no assumed overlap'}

def measured_cost(condition, *, dataset=None, percentile=50, resident=True, migration=None,
                  cold_model_load_ns=None):
    cost = (dataset or JobCostDataset()).cost(condition,percentile)
    if not resident and migration is None:
        raise ValueError('nonresident cost requires explicit migration; unknown is not zero')
    extra = 0 if resident else migration['duration_ns']
    nonnegative(extra,'migration')
    if cold_model_load_ns is not None:
        nonnegative(cold_model_load_ns,'cold model load')
    return dict(cost, data_residency='resident' if resident else 'migrating',
                ready_service_plus_transfer_ns=cost['duration_ns']+extra,
                cold_model_load_ns=cold_model_load_ns,
                cold_start_total_ns=None if cold_model_load_ns is None else cost['duration_ns']+extra+cold_model_load_ns,
                residency_scope='device-visible DDR BO; no implicit CBUF state')

def example(dataset=None):
    dataset = dataset or JobCostDataset()
    cells = dataset.match(campaign='cost-20261010-165249', workload='lut', workers=1,
                          residency='device_ready', tracing=False, actual_mask=1, mode='chain')
    cells.sort(key=lambda c:(c['condition']['task_count'],c['condition']['shape_mkn']))
    if len(cells)<2:raise ValueError('need measured cells')
    chosen = [cells[0],cells[-1]]
    document={'schema_version':1,'meta':{'name':'measured-RK3588-costs-hypothetical-two-device-topology',
                'note':'Two independently configured logical devices use actual RK3588 costs. This does not model isolated DDR domains of RK3588 cores or measured RK182X.',
                'cost_cell_ids':[c['id'] for c in chosen]},
              'devices':[{'id':d,'memory_capacity_bytes':1<<24,'measured':False} for d in ('A','B')],
              'links':[{'src':src,'dst':dst,'bandwidth_bytes_per_s':1000000000,'latency_ns':2000}
                       for src,dst in [('host','A'),('host','B'),('A','B'),('B','A')]],'tensors':[],'jobs':[]}
    for i in range(8):
        c=chosen[i%2];condition=c['condition'];cost=dataset.cost(condition)
        # Full observed operation BO allocation used as workspace; these independent
        # jobs carry no separately counted tensor BOs, so footprint is not doubled.
        document['jobs'].append({'id':f'job{i}','arrival_ns':0,'task_count':condition['task_count'],
                                'workspace_bytes':c['memory']['allocated_tensor_bo_bytes'],
                                'durations_ns':{d:cost['duration_ns'] for d in ('A','B')},
                                'depends_on':[]})
    results=simulate_all(validate_scenario(document),strict=True)
    return document, {'scenario':document['meta'],'cost_evidence':'measured',
                      'topology_evidence':'configured','policies':results}

def main():
    p=argparse.ArgumentParser();p.add_argument('--example',action='store_true');args=p.parse_args()
    if not args.example:p.error('use --example')
    doc,results=example();here=Path(__file__).resolve().parent
    (here/'examples/measured-rk3588.json').write_text(json.dumps(doc,indent=2)+'\n')
    (here/'results/measured-baselines.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({k:r['makespan_ns'] for k,r in results['policies'].items()}))
if __name__=='__main__':main()
