// SPDX-License-Identifier: GPL-3.0-or-later
/* Read-only LD_PRELOAD observer for the vendor ABI. No register/counter ioctls.
 * Driver-written descriptor int_status exposes the actual completion bank bits. */
#define _GNU_SOURCE
#include <stddef.h>
#include "rknpu_uapi.h"
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

typedef struct {unsigned id;int fd;uint32_t handle;uint64_t obj,dma,size,offset;int domain;void *ptr;size_t length;} allocation;
static allocation mem[128];
static unsigned serial,submit_serial,event_serial,current_round;
static char current_arm='?';static int logfd=-1;
static const char *dir;static __thread int busy;
static int (*next_ioctl)(int,unsigned long,...);
static void *(*next_mmap)(void *,size_t,int,int,int,off_t);
static int (*next_munmap)(void *,size_t);
static uint64_t clock_ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000ull+t.tv_nsec;}
static void event(const char *fmt,...) {
    if(logfd<0)return;
    char b[2048];
    int n=snprintf(b,sizeof(b),"{\"event_id\":%u,",++event_serial);
    va_list a;va_start(a,fmt);int k=vsnprintf(b+n,sizeof(b)-(size_t)n,fmt,a);va_end(a);
    if(k<0||(size_t)k>=sizeof(b)-(size_t)n)_exit(90);
    n+=k;if(write(logfd,b,(size_t)n)!=n)_exit(91);
}
static allocation *object(int fd,uint64_t obj){for(unsigned i=0;i<128;i++)if(mem[i].id&&mem[i].fd==fd&&mem[i].obj==obj)return &mem[i];return NULL;}
static allocation *handle(int fd,uint32_t h){for(unsigned i=0;i<128;i++)if(mem[i].id&&mem[i].fd==fd&&mem[i].handle==h)return &mem[i];return NULL;}
static allocation *address(int fd,uint64_t dma,size_t bytes){for(unsigned i=0;i<128;i++)if(mem[i].id&&mem[i].fd==fd&&dma>=mem[i].dma&&dma-mem[i].dma<=mem[i].size&&bytes<=mem[i].size-(dma-mem[i].dma))return &mem[i];return NULL;}
static unsigned live(void){unsigned n=0;for(unsigned i=0;i<128;i++)n+=mem[i].id!=0;return n;}
__attribute__((constructor)) static void init(void) {
    busy=1;void *s=dlsym(RTLD_NEXT,"ioctl");memcpy(&next_ioctl,&s,sizeof(s));s=dlsym(RTLD_NEXT,"mmap");memcpy(&next_mmap,&s,sizeof(s));s=dlsym(RTLD_NEXT,"munmap");memcpy(&next_munmap,&s,sizeof(s));
    dir=getenv("TT_TRACE_DIR");const char *p=getenv("TT_TRACE_JSON");if(p)logfd=open(p,O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC,0644);busy=0;
}
__attribute__((destructor)) static void finish(void){event("\"event\":\"end\",\"live_allocations\":%u}\n",live());if(logfd>=0)close(logfd);}
void tt_stage(unsigned round,const char *arm){current_round=round;current_arm=arm[0];event("\"event\":\"stage\",\"round\":%u,\"arm\":\"%c\"}\n",round,current_arm);}

void *mmap(void *a,size_t len,int prot,int flags,int fd,off_t offset) {
    void *p=next_mmap(a,len,prot,flags,fd,offset);if(busy||logfd<0||p==MAP_FAILED)return p;
    for(unsigned i=0;i<128;i++)if(mem[i].id&&mem[i].fd==fd&&mem[i].offset==(uint64_t)offset) {
        mem[i].ptr=p;mem[i].length=len;event("\"event\":\"mmap\",\"id\":%u,\"address\":%llu,\"length\":%zu}\n",mem[i].id,(unsigned long long)(uintptr_t)p,len);break;
    }return p;
}
void *mmap64(void *a,size_t len,int prot,int flags,int fd,off64_t offset){return mmap(a,len,prot,flags,fd,(off_t)offset);}
int munmap(void *a,size_t len) {
    int rc=next_munmap(a,len);if(!busy&&logfd>=0&&rc==0)for(unsigned i=0;i<128;i++)if(mem[i].id&&mem[i].ptr==a){event("\"event\":\"unmap\",\"id\":%u,\"address\":%llu,\"length\":%zu}\n",mem[i].id,(unsigned long long)(uintptr_t)a,len);mem[i].ptr=NULL;mem[i].length=0;break;}return rc;
}

int ioctl(int fd,unsigned long request,...) {
    va_list args;va_start(args,request);void *arg=va_arg(args,void *);va_end(args);
    if(busy||logfd<0)return next_ioctl(fd,request,arg);
    busy=1;unsigned nr=_IOC_NR(request),kind=_IOC_TYPE(request),seq=0;
    allocation *taskbo=NULL;struct rknpu_submit before={0};unsigned refs[16]={0},nrefs=0;
    unsigned char *descriptor_snapshot=NULL;
    int submit=kind=='d'&&nr==0x41&&_IOC_SIZE(request)==sizeof(struct rknpu_submit);
    if(submit) {
        memcpy(&before,arg,sizeof(before));seq=++submit_serial;taskbo=object(fd,before.task_obj_addr);
        size_t off=(size_t)before.task_start*sizeof(struct rknpu_task),bytes=(size_t)before.task_number*sizeof(struct rknpu_task);
        event("\"event\":\"submit_begin\",\"submit_id\":%u,\"fd\":%d,\"round\":%u,\"arm\":\"%c\",\"flags\":%u,\"count\":%u,\"task_obj\":%llu,\"task_id\":%u,\"task_offset\":%zu,\"task_bytes\":%zu,\"task_size\":40,\"task_start\":%u,\"core_mask\":%u,\"domain\":%u,\"base\":%llu}\n",seq,fd,current_round,current_arm,before.flags,before.task_number,(unsigned long long)before.task_obj_addr,taskbo?taskbo->id:0,off,bytes,before.task_start,before.core_mask,before.iommu_domain_id,(unsigned long long)before.task_base_addr);
        unsigned core=before.core_mask==1?0:(before.core_mask==2?1:2);
        event("\"event\":\"subcore\",\"submit_id\":%u,\"index\":%u,\"start\":%u,\"count\":%u}\n",seq,core,before.subcore_task[core].task_start,before.subcore_task[core].task_number);
        if(taskbo&&taskbo->ptr&&off<=taskbo->length&&bytes<=taskbo->length-off&&before.task_number<=2) {
            descriptor_snapshot=malloc(taskbo->length);
            if(!descriptor_snapshot)_exit(94);
            memcpy(descriptor_snapshot,taskbo->ptr,taskbo->length);
            refs[nrefs++]=taskbo->id;
            for(unsigned i=0;i<before.task_number;i++) {
                struct rknpu_task t;memcpy(&t,(char *)taskbo->ptr+off+i*40,40);
                unsigned words=t.regcfg_amount+4;size_t fetch=(size_t)((words+1)&~1u)*8;
                allocation *r=address(fd,t.regcmd_addr,fetch);uint32_t src=0,dst=0,sd=0,sr=0,width=0,enable=0;
                uint32_t next=0,amount=0;int valid=0;
                if(r&&r->ptr&&t.regcmd_addr-r->dma+fetch<=r->length&&words<=CO_MAX_WORDS) {
                    const uint64_t *ops=(const uint64_t *)((char *)r->ptr+(t.regcmd_addr-r->dma));
                    for(unsigned j=0;j<words;j++) {uint16_t reg=(uint16_t)ops[j];uint32_t val=(uint32_t)(ops[j]>>16);
                        if(reg==0x5018)src=val;
                        if(reg==0x4020)dst=val;
                        if(reg==0x4004)sd=val;
                        if(reg==0x5004)sr=val;
                        if(reg==0x4030)width=val;
                    }
                    next=(uint16_t)(ops[words-4]>>48)==0x101?(uint32_t)(ops[words-4]>>16):0;
                    amount=(uint32_t)(ops[words-3]>>16);enable=(uint32_t)(ops[words-1]>>16);
                    valid=1;refs[nrefs++]=r->id;
                    if(dir&&seq<=15) {char path[1024];snprintf(path,sizeof(path),"%s/program-%04u-%u.bin",dir,seq,i);FILE *out=fopen(path,"wb");if(!out||fwrite(ops,8,words,out)!=words||fclose(out))_exit(92);}
                }
                size_t data=(size_t)(width+1)*16;allocation *s=address(fd,src,data),*d=address(fd,dst,data);
                if(s)refs[nrefs++]=s->id;
                if(d)refs[nrefs++]=d->id;
                event("\"event\":\"program\",\"submit_id\":%u,\"index\":%u,\"desc_offset\":%zu,\"desc_bytes\":40,\"regcmd\":%llu,\"amount\":%u,\"words\":%u,\"rc_id\":%u,\"rc_offset\":%llu,\"fetch_bytes\":%zu,\"int_mask\":%u,\"sptr_dpu\":%u,\"sptr_rdma\":%u,\"src\":%u,\"dst\":%u,\"src_id\":%u,\"dst_id\":%u,\"data_bytes\":%zu,\"next\":%u,\"next_amount\":%u,\"enable\":%u,\"valid\":%s}\n",seq,i,off+i*40,(unsigned long long)t.regcmd_addr,t.regcfg_amount,words,r?r->id:0,(unsigned long long)(r?t.regcmd_addr-r->dma:0),fetch,t.int_mask,sd,sr,src,dst,s?s->id:0,d?d->id:0,data,next,amount,enable,valid&&s&&d?"true":"false");
            }
        }
    }
    uint64_t start=clock_ns();int rc=next_ioctl(fd,request,arg),saved=errno;uint64_t duration=clock_ns()-start;
    if(submit) {
        struct rknpu_submit *after=arg;uint32_t status=0;
        if(taskbo&&taskbo->ptr&&before.task_number&&before.task_start+before.task_number<=taskbo->length/40) {
            struct rknpu_task t;memcpy(&t,(char *)taskbo->ptr+(before.task_start+before.task_number-1)*40,40);status=t.int_status;
        }
        int all=nrefs>0;for(unsigned j=0;j<nrefs;j++){int found=0;for(unsigned i=0;i<128;i++)if(mem[i].id==refs[j])found=1;if(!found)all=0;}
        int unchanged=0;
        if(descriptor_snapshot&&taskbo&&taskbo->ptr&&before.task_number) {
            size_t status_off=(size_t)(before.task_start+before.task_number-1)*40+offsetof(struct rknpu_task,int_status);
            if(status_off+4<=taskbo->length) {
                memcpy(descriptor_snapshot+status_off,&status,4);
                unchanged=memcmp(descriptor_snapshot,taskbo->ptr,taskbo->length)==0;
            }
        }
        event("\"event\":\"submit_end\",\"submit_id\":%u,\"result\":%d,\"errno\":%d,\"wall_ns\":%llu,\"hw_ns\":%lld,\"last_status\":%u,\"live\":%s,\"descriptor_guard\":%s}\n",seq,rc,rc?saved:0,(unsigned long long)duration,(long long)after->hw_elapse_time,status,all?"true":"false",unchanged?"true":"false");
        free(descriptor_snapshot);
    }else if(kind=='d'&&rc==0) {
        if(nr==0x42&&_IOC_SIZE(request)==sizeof(struct rknpu_mem_create)) {
            struct rknpu_mem_create *a=arg;allocation *r=NULL;
            for(unsigned i=0;i<128;i++)if(!mem[i].id){r=&mem[i];break;}
            if(!r)_exit(93);
            *r=(allocation){.id=++serial,.fd=fd,.handle=a->handle,.obj=a->obj_addr,.dma=a->dma_addr,.size=a->size,.domain=a->iommu_domain_id};
            event("\"event\":\"create\",\"fd\":%d,\"id\":%u,\"handle\":%u,\"obj\":%llu,\"dma\":%llu,\"size\":%llu,\"domain\":%d}\n",fd,r->id,r->handle,(unsigned long long)r->obj,(unsigned long long)r->dma,(unsigned long long)r->size,r->domain);
        }else if(nr==0x43&&_IOC_SIZE(request)==sizeof(struct rknpu_mem_map)) {
            struct rknpu_mem_map *a=arg;allocation *r=handle(fd,a->handle);if(r){r->offset=a->offset;event("\"event\":\"map\",\"fd\":%d,\"id\":%u,\"offset\":%llu}\n",fd,r->id,(unsigned long long)r->offset);}
        }else if(nr==0x44&&_IOC_SIZE(request)==sizeof(struct rknpu_mem_destroy)) {
            struct rknpu_mem_destroy *a=arg;allocation *r=object(fd,a->obj_addr);event("\"event\":\"destroy\",\"fd\":%d,\"id\":%u,\"obj\":%llu,\"result\":%d}\n",fd,r?r->id:0,(unsigned long long)a->obj_addr,rc);if(r)memset(r,0,sizeof(*r));
        }else if(nr==0x45&&_IOC_SIZE(request)==sizeof(struct rknpu_mem_sync)) {
            struct rknpu_mem_sync *a=arg;allocation *r=object(fd,a->obj_addr);event("\"event\":\"sync\",\"fd\":%d,\"id\":%u,\"offset\":%llu,\"size\":%llu,\"flags\":%u,\"result\":%d}\n",fd,r?r->id:0,(unsigned long long)a->offset,(unsigned long long)a->size,a->flags,rc);
        }
    }
    busy=0;errno=saved;return rc;
}
