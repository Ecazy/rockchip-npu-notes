// SPDX-License-Identifier: GPL-3.0-or-later
#define _GNU_SOURCE
#include "co_schedule.h"
#include "rocket_chain.h"
#include "rknpu_uapi.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

static uint64_t mono(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000ull+t.tv_nsec;}
static unsigned long command(unsigned nr,size_t size){return _IOC(_IOC_READ|_IOC_WRITE,'d',0x40+nr,size);}
int co_schedule_mask_valid(unsigned mask){return mask==0||mask==1||mask==2||mask==4;}
int co_schedule_pick(unsigned allowed,unsigned ordinal,unsigned *out){
    if(!out||!allowed||(allowed&~7u))return ROCKET_E_SHAPE;
    unsigned count=0;for(unsigned i=0;i<3;i++)count+=(allowed>>i)&1;
    unsigned pick=ordinal%count;
    for(unsigned i=0;i<3;i++)if(allowed&(1u<<i)){if(!pick){*out=1u<<i;return 0;}pick--;}
    return ROCKET_E_SHAPE;
}
static void marker(int fd,const char *name,co_schedule_profile *p,uint64_t time){
    if(fd<0)return;
    char text[192];int n=snprintf(text,sizeof(text),"RS %s id=%llu tid=%u mask=%u mono=%llu\n",name,
        (unsigned long long)p->request_id,p->tid,p->requested_mask,(unsigned long long)time);
    if(write(fd,text,(size_t)n)!=n)p->result=ROCKET_E_DEVICE;
}
int co_schedule_execute(int fd,const co_program *p,unsigned n,unsigned mask,int tracefd,
                        uint64_t id,co_schedule_profile *out){
    if(!out)return ROCKET_E_SHAPE;
    memset(out,0,sizeof(*out));out->request_id=id;out->tid=(uint32_t)syscall(SYS_gettid);
    out->requested_mask=mask;out->task_count=n;out->host_begin_ns=mono();
    if(fd<0||!p||!n||n>CO_MAX_PROGRAMS||!co_schedule_mask_valid(mask))return out->result=ROCKET_E_SHAPE;
    for(unsigned i=0;i<n;i++){
        if(!p[i].count||p[i].count>CO_MAX_WORDS||p[i].flags||!p[i].output.buffer||
           p[i].output.buffer->fd!=fd)return out->result=ROCKET_E_SHAPE;
        for(unsigned j=0;j<p[i].nreads;j++)if(!p[i].reads[j].buffer||p[i].reads[j].buffer->fd!=fd)
            return out->result=ROCKET_E_SHAPE;
    }
    marker(tracefd,"request",out,out->host_begin_ns);
    rocket_bo rcbo={0};struct rknpu_mem_create task={0};void *mapped=MAP_FAILED;
    int rc=0;size_t offsets[CO_MAX_PROGRAMS],words=0;
    rocket_task_desc descriptors[CO_MAX_PROGRAMS];
    for(unsigned i=0;i<n;i++){offsets[i]=words;words+=rkt_chain_words(p[i].count);}
    if((rc=rocket_bo_alloc32(fd,words*8,&rcbo)))goto cleanup;
    if((rc=rocket_bo_prep(fd,&rcbo,1,0)))goto cleanup;
    memset(rcbo.ptr,0,rcbo.size);
    for(unsigned i=0;i<n;i++){
        rc=rkt_chain_pack_at(&rcbo,descriptors,(int)i,offsets[i],i+1<n?offsets[i+1]:offsets[i],
                             p[i].words,p[i].count,i+1<n?p[i+1].count:p[i].count);
        if(rc)break;
    }
    if(!rc)rc=rkt_chain_verify(&rcbo,descriptors,(int)n);
    {int end=rocket_bo_fini(fd,&rcbo);if(!rc)rc=end;}if(rc)goto cleanup;
    task.size=n*sizeof(struct rknpu_task);
    task.flags=RKNPU_MEM_NON_CONTIGUOUS|RKNPU_MEM_CACHEABLE|RKNPU_MEM_KERNEL_MAPPING|RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT;
    if(ioctl(fd,command(RKNPU_MEM_CREATE_NR,sizeof(task)),&task)<0){rc=-errno;goto cleanup;}
    struct rknpu_mem_map map={.handle=task.handle};
    if(ioctl(fd,command(RKNPU_MEM_MAP_NR,sizeof(map)),&map)<0){rc=-errno;goto cleanup;}
    mapped=mmap(NULL,task.size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,map.offset);
    if(mapped==MAP_FAILED){rc=-errno;goto cleanup;}
    struct rknpu_task *t=mapped;memset(mapped,0,task.size);
    for(unsigned i=0;i<n;i++){
        t[i].regcmd_addr=descriptors[i].regcmd;t[i].regcfg_amount=descriptors[i].regcmd_count-4;
        t[i].int_mask=RKNPU_TASK_INT_MASK_DPU;
    }
    struct rknpu_mem_sync sync={.obj_addr=task.obj_addr,.size=task.size,.flags=RKNPU_MEM_SYNC_TO_DEVICE};
    if(ioctl(fd,command(RKNPU_MEM_SYNC_NR,sizeof(sync)),&sync)<0){rc=-errno;goto cleanup;}
    struct rknpu_submit s={.flags=RKNPU_JOB_PC|RKNPU_JOB_BLOCK,.timeout=2000,
        .task_number=n,.task_obj_addr=task.obj_addr,.core_mask=mask,.fence_fd=-1};
    for(unsigned i=0;i<3;i++){s.subcore_task[i].task_start=0;s.subcore_task[i].task_number=n;}
    out->ioctl_begin_ns=mono();marker(tracefd,"ioctl",out,out->ioctl_begin_ns);
    if(out->result){rc=out->result;goto cleanup;}
    if(ioctl(fd,command(RKNPU_SUBMIT_NR,sizeof(s)),&s)<0)rc=-errno;
    out->ioctl_end_ns=mono();out->actual_mask=s.core_mask;out->driver_service_ns=s.hw_elapse_time;
    out->int_status=t[n-1].int_status;
    marker(tracefd,"return",out,out->ioctl_end_ns);
    if(!rc&&(!s.core_mask||!co_schedule_mask_valid(s.core_mask)||
             (mask&&s.core_mask!=mask)||s.hw_elapse_time<=0||(out->int_status&0x300)==0))rc=ROCKET_E_DEVICE;
    /* Submit is blocking on this provider. Read synchronization and observation
     * are measured separately from the driver service envelope. */
    for(unsigned i=0;!rc&&i<n;i++){
        rc=rocket_bo_prep(fd,&p[i].output.buffer->bo,0,2000000000ull);
        if(!rc)rc=rocket_bo_fini(fd,&p[i].output.buffer->bo);
    }
cleanup:
    if(mapped!=MAP_FAILED)munmap(mapped,task.size);
    if(task.obj_addr){struct rknpu_mem_destroy d={.handle=task.handle,.obj_addr=task.obj_addr};
        if(ioctl(fd,command(RKNPU_MEM_DESTROY_NR,sizeof(d)),&d)<0&&!rc)rc=-errno;}
    rocket_bo_free(fd,&rcbo);
    out->host_done_ns=mono();marker(tracefd,"done",out,out->host_done_ns);
    out->result=rc?rc:out->result;
    return out->result;
}
