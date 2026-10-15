// SPDX-License-Identifier: GPL-3.0-or-later
/* Experimental RK3588 module-level regcmd composition. No production emitter changes.
 * Instance/reference helpers adapted from overlap_probe.c. Raw submit deliberately
 * separates stream length from hardware task count to test completion accounting. */
#define _GNU_SOURCE
#include "co_schedule.h"
#include "npu_hw.h"
#include "npu_matmul.h"
#include "rknpu_uapi.h"
#include "rocket_chain.h"
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#define OV_GEMM_M 256u
#define OV_K 14336u
#define OV_N 256u
enum { INST_MM, INST_LUT };
static int matched_common_weights;
static size_t matched_row_offset,matched_b_offset;
static uint32_t mix32(uint32_t x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
static _Float16 hs(double x){return (_Float16)fmin(1,fmax(0,x/6+0.5));}
static uint64_t mono(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000ull+t.tv_nsec;}

static int irqs(uint64_t out[3]) {
    FILE *f=fopen("/proc/interrupts","r"); if(!f)return -1;
    char line[2048]; unsigned found=0; memset(out,0,3*sizeof(*out));
    while(fgets(line,sizeof(line),f)) if(strstr(line,"fdab0000.npu")) {
        if(found>=3){fclose(f);return -1;}
        char *p=strchr(line,':'); if(!p){fclose(f);return -1;} p++;
        for(;;) {
            while(*p==' '||*p=='\t')p++;
            if(!isdigit((unsigned char)*p))break;
            char *end; out[found]+=strtoull(p,&end,10); p=end;
        }
        found++;
    }
    fclose(f); return found==3?0:-1;
}

/* ---- instances --------------------------------------------------------- */

typedef struct { co_tensor t; co_buffer b; co_view v; } slot;

static int slot_init(int fd,const co_tensor *t,slot *s) {
    memset(s,0,sizeof(*s)); s->t=*t;
    int rc=co_buffer_alloc(fd,t->bytes,&s->b);
    if(!rc)rc=co_view_init(&s->v,&s->b,0,t);
    return rc;
}
static void slot_free(slot *s){co_buffer_free(&s->b);}

typedef struct inst_t {
    int kind;
    unsigned rows,k,cols,physical_rows;
    slot x,w,y;
    _Float16 *hx,*hw,*ref;
    size_t nx,nw,ny;
    uint64_t last_hash;
    co_program prog;
} inst_t;

typedef struct { int kind; unsigned rows,k,cols; uint32_t seed; } inst_def;

static int inst_init(int fd,inst_t *in,const inst_def *d) {
    memset(in,0,sizeof(*in));
    in->kind=d->kind; in->rows=d->rows; in->k=d->k; in->cols=d->cols;
    co_tensor xt,yt,wt; int rc;
    if(d->kind==INST_MM) {
        if(d->rows>OV_GEMM_M||!d->k||d->k>OV_K||!d->cols||d->cols>OV_N)return ROCKET_E_SHAPE;
        if(d->k<=CO_MAX_K){if((rc=co_tensor_input(&xt,d->rows,d->k)))return rc;}
        else {
            if((rc=co_tensor_matrix(&xt,d->rows,32,CO_F16)))return rc;
            xt.cols=d->k;xt.physical_cols=(d->k+31u)&~31u;
            xt.bytes=(size_t)xt.physical_rows*xt.physical_cols*2;
        }
        if(d->k<=256){if((rc=co_tensor_weight(&wt,d->cols,d->k)))return rc;}
        else {
            /* Experimental wide-K weights: bypass only the production <=256 K
             * view contract; retain explicit allocation, bounds and independent reference. */
            unsigned pn=(d->cols+31u)&~31u,pk=(d->k+31u)&~31u;
            wt=(co_tensor){.type=CO_F16,.layout=CO_WEIGHT,.rows=d->cols,.cols=d->k,
                .physical_rows=pn,.physical_cols=pk,.atom=32,.bytes=(size_t)pn*pk*2};
            if(xt.bytes>11u*32768u)return ROCKET_E_SHAPE;
        }
        if((rc=co_tensor_matrix(&yt,d->rows,d->cols,CO_F16)))return rc;
        in->physical_rows=xt.physical_rows;
        in->nx=(size_t)xt.physical_rows*(size_t)xt.physical_cols;
        in->nw=(size_t)d->cols*d->k;
        in->ny=(size_t)d->rows*d->cols;
        in->hx=malloc(in->nx*2); in->hw=malloc(in->nw*2); in->ref=malloc(in->ny*2);
        if(!in->hx||!in->hw||!in->ref)return ROCKET_E_NOMEM;
        for(size_t i=0;i<in->nx;i++)in->hx[i]=(_Float16)(((int)(mix32(matched_common_weights?17u+(uint32_t)(matched_row_offset*d->k+i):d->seed+(uint32_t)i)%17)-8)/64.0);
        for(size_t i=0;i<in->nw;i++)in->hw[i]=(_Float16)(((int)(mix32((matched_common_weights?48u:d->seed+31)+(uint32_t)i)%9)-4)/128.0);
        for(unsigned r=0;r<d->rows;r++)for(unsigned c=0;c<d->cols;c++) {
            double a=0;
            for(unsigned q=0;q<d->k;q++)a+=(double)in->hx[(size_t)r*d->k+q]*(double)in->hw[(size_t)c*d->k+q];
            in->ref[(size_t)r*d->cols+c]=(_Float16)a;
        }
        if((rc=slot_init(fd,&yt,&in->y)))return rc;
        if(d->k<=CO_MAX_K){if((rc=slot_init(fd,&xt,&in->x)))return rc;}
        else {
            in->x.t=xt;
            if((rc=co_buffer_alloc(fd,xt.bytes,&in->x.b)))return rc;
            if(in->x.b.bo.dma_address>UINT32_MAX||xt.bytes-1>UINT32_MAX-in->x.b.bo.dma_address)return ROCKET_E_SHAPE;
            in->x.v=(co_view){.buffer=&in->x.b,.tensor=xt};
        }
        if(d->k<=256){
            if((rc=slot_init(fd,&wt,&in->w)))return rc;
            if((rc=co_upload_f16(&in->w.v,in->hw)))return rc;
        } else {
            in->w.t=wt;
            if((rc=co_buffer_alloc(fd,wt.bytes,&in->w.b)))return rc;
            if(in->w.b.bo.dma_address>UINT32_MAX||wt.bytes-1>UINT32_MAX-in->w.b.bo.dma_address)return ROCKET_E_SHAPE;
            in->w.v=(co_view){.buffer=&in->w.b,.tensor=wt};
            if((rc=rocket_bo_prep(fd,&in->w.b.bo,1,0)))return rc;
            _Float16 *dst=in->w.b.bo.ptr;
            for(unsigned r=0;r<wt.rows;r++)for(unsigned c=0;c<wt.cols;c++)
                dst[co_index(&wt,r,c)]=in->hw[(size_t)r*wt.cols+c];
            if((rc=rocket_bo_fini(fd,&in->w.b.bo)))return rc;
        }
        if(d->k<=CO_MAX_K){if((rc=co_upload_f16(&in->x.v,in->hx)))return rc;}
        else {
            if((rc=rocket_bo_prep(fd,&in->x.b.bo,1,0)))return rc;
            _Float16 *dst=in->x.b.bo.ptr;
            for(unsigned r=0;r<xt.rows;r++)for(unsigned c=0;c<xt.cols;c++)
                dst[co_index(&xt,r,c)]=in->hx[(size_t)r*xt.cols+c];
            if((rc=rocket_bo_fini(fd,&in->x.b.bo)))return rc;
        }
        if(d->k<=256)rc=co_emit_mm(&in->prog,in->x.v,in->w.v,in->y.v);
        else {
            matmul_params_t a={.m=xt.physical_rows,.k=xt.physical_cols,.n=yt.physical_cols,
                .input_dma=in->x.b.bo.dma_address,.weights_dma=in->w.b.bo.dma_address,
                .output_dma=in->y.b.bo.dma_address,.tasks=in->prog.words,.fp32tofp16=1};
            rc=gen_matmul_fp16(&a);in->prog.count=a.task_count;
            in->prog.reads[0]=in->x.v;in->prog.reads[1]=in->w.v;
            in->prog.nreads=2;in->prog.output=in->y.v;
        }
    } else {
        unsigned r=d->rows,c=d->cols;
        in->k=0; in->physical_rows=r;
        if(r==1){
            if((rc=co_tensor_linear(&xt,c)))return rc;
            yt=xt;
        } else {
            if((rc=co_tensor_matrix(&xt,r,c,CO_F16)))return rc;
            if((rc=co_tensor_matrix(&yt,r,c,CO_F16)))return rc;
        }
        in->nx=in->ny=(size_t)r*c;
        in->hx=malloc(in->nx*2); in->ref=malloc(in->ny*2);
        if(!in->hx||!in->ref)return ROCKET_E_NOMEM;
        for(size_t i=0;i<in->nx;i++) {
            in->hx[i]=(_Float16)(((int)(mix32(matched_common_weights?114u+(uint32_t)(matched_b_offset+i):d->seed+7+(uint32_t)i)%33)-16)/4.0);
            in->ref[i]=hs((double)in->hx[i]);
        }
        if((rc=slot_init(fd,&yt,&in->y)))return rc;
        if(d->k<=CO_MAX_K){if((rc=slot_init(fd,&xt,&in->x)))return rc;}
        else {
            in->x.t=xt;
            if((rc=co_buffer_alloc(fd,xt.bytes,&in->x.b)))return rc;
            if(in->x.b.bo.dma_address>UINT32_MAX||xt.bytes-1>UINT32_MAX-in->x.b.bo.dma_address)return ROCKET_E_SHAPE;
            in->x.v=(co_view){.buffer=&in->x.b,.tensor=xt};
        }
        if((rc=co_upload_f16(&in->x.v,in->hx)))return rc;
        rc=co_emit_lut(&in->prog,in->x.v,in->y.v,CO_HARDSIGMOID);
    }
    return rc;
}
static void inst_free(inst_t *in) {
    slot_free(&in->x); slot_free(&in->w); slot_free(&in->y);
    free(in->hx); free(in->hw); free(in->ref);
}
static double inst_tol(const inst_t *in){return in->kind==INST_LUT?0.002:0.02;}


static unsigned long command(unsigned nr,size_t size){return _IOC(_IOC_READ|_IOC_WRITE,'d',0x40+nr,size);}
static int raw_execute(int fd,const co_program *p,unsigned n,unsigned streams,co_schedule_profile *out){
    if(!out)return ROCKET_E_SHAPE;
    memset(out,0,sizeof(*out));out->request_id=0;out->tid=(uint32_t)getpid();
    out->requested_mask=1;out->task_count=n;out->host_begin_ns=mono();
    if(fd<0||!p||!n||n>CO_MAX_PROGRAMS||!streams||streams>CO_MAX_PROGRAMS||
       (streams!=n&&!(streams==1&&n==2)))return out->result=ROCKET_E_SHAPE;
    for(unsigned i=0;i<streams;i++)if(p[i].count<4||p[i].count>CO_MAX_WORDS)return out->result=ROCKET_E_SHAPE;
    rocket_bo rcbo={0};struct rknpu_mem_create task={0};void *mapped=MAP_FAILED;
    int rc=0;size_t words=0,offsets[CO_MAX_PROGRAMS];
    for(unsigned i=0;i<streams;i++){offsets[i]=words;words+=rkt_chain_words(p[i].count);}
    if((rc=rocket_bo_alloc32(fd,words*8,&rcbo)))goto cleanup;
    if((rc=rocket_bo_prep(fd,&rcbo,1,0)))goto cleanup;
    memset(rcbo.ptr,0,rcbo.size);
    rocket_task_desc descriptors[CO_MAX_PROGRAMS];
    if(streams>1) {
        for(unsigned i=0;i<streams&&!rc;i++){
            unsigned next=i+1<streams?i+1:i;
            rc=rkt_chain_pack_at(&rcbo,descriptors,(int)i,offsets[i],offsets[next],p[i].words,p[i].count,p[next].count);
        }
        if(!rc)rc=rkt_chain_verify(&rcbo,descriptors,(int)streams);
    } else memcpy(rcbo.ptr,p->words,(size_t)p->count*8);
    if(rc)goto cleanup;
    const char *dump_path=getenv("RG_DUMP_PATH");
    if(dump_path){FILE *dump=fopen(dump_path,"wx");if(dump){
        fprintf(dump,"# hw_tasks=%u streams=%u iova=%llx count=%zu pp=%s\n",n,streams,
            (unsigned long long)rcbo.dma_address,words,getenv("RG_PC_PP")?getenv("RG_PC_PP"):"0");
        for(size_t j=0;j<words;j++)fprintf(dump,"%04zu %016llx\n",j,(unsigned long long)((uint64_t *)rcbo.ptr)[j]);
        fclose(dump);
    }}
    if((rc=rocket_bo_fini(fd,&rcbo)))goto cleanup;
    task.size=n*sizeof(struct rknpu_task);
    task.flags=RKNPU_MEM_NON_CONTIGUOUS|RKNPU_MEM_CACHEABLE|RKNPU_MEM_KERNEL_MAPPING|RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT;
    if(ioctl(fd,command(RKNPU_MEM_CREATE_NR,sizeof(task)),&task)<0){rc=-errno;goto cleanup;}
    struct rknpu_mem_map map={.handle=task.handle};
    if(ioctl(fd,command(RKNPU_MEM_MAP_NR,sizeof(map)),&map)<0){rc=-errno;goto cleanup;}
    mapped=mmap(NULL,task.size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,map.offset);
    if(mapped==MAP_FAILED){rc=-errno;goto cleanup;}
    struct rknpu_task *t=mapped;memset(mapped,0,task.size);
    for(unsigned i=0;i<n;i++){
        t[i].regcmd_addr=rcbo.dma_address+offsets[streams>1?i:0]*8;t[i].regcfg_amount=p[streams>1?i:0].count-4;
        t[i].int_mask=RKNPU_TASK_INT_MASK_DPU;
    }
    struct rknpu_mem_sync sync={.obj_addr=task.obj_addr,.size=task.size,.flags=RKNPU_MEM_SYNC_TO_DEVICE};
    if(ioctl(fd,command(RKNPU_MEM_SYNC_NR,sizeof(sync)),&sync)<0){rc=-errno;goto cleanup;}
    unsigned pp=getenv("RG_PC_PP")?(unsigned)atoi(getenv("RG_PC_PP")):0;
    struct rknpu_submit s={.flags=RKNPU_JOB_PC|RKNPU_JOB_BLOCK|(pp?4u:0u),.timeout=2000,
        .task_number=n,.task_obj_addr=task.obj_addr,.core_mask=1,.fence_fd=-1};
    for(unsigned i=0;i<3;i++){s.subcore_task[i].task_start=0;s.subcore_task[i].task_number=n;}
    out->ioctl_begin_ns=mono();
    if(out->result){rc=out->result;goto cleanup;}
    if(ioctl(fd,command(RKNPU_SUBMIT_NR,sizeof(s)),&s)<0)rc=-errno;
    out->ioctl_end_ns=mono();out->actual_mask=s.core_mask;out->driver_service_ns=s.hw_elapse_time;
    out->int_status=t[n-1].int_status;

    if(!rc&&(!s.core_mask||s.core_mask!=1||s.hw_elapse_time<=0||(out->int_status&0x300)==0))rc=ROCKET_E_DEVICE;
cleanup:
    if(mapped!=MAP_FAILED)munmap(mapped,task.size);
    if(task.obj_addr){struct rknpu_mem_destroy d={.handle=task.handle,.obj_addr=task.obj_addr};
        if(ioctl(fd,command(RKNPU_MEM_DESTROY_NR,sizeof(d)),&d)<0&&!rc)rc=-errno;}
    rocket_bo_free(fd,&rcbo);
    out->host_done_ns=mono();
    out->result=rc?rc:out->result;
    return out->result;
}

static void put(co_program *p,uint64_t w) {
    if(p->count>=CO_MAX_WORDS){fprintf(stderr,"stream capacity exceeded\n");exit(2);}
    p->words[p->count++]=w;
}
/* Filter config by target block; never touch CNA/CORE pointers owned by driver. */
static void config(co_program *p,const co_program *src,int blocks) {
    for(unsigned i=0;i<src->count-4;i++) {
        unsigned op=(unsigned)(src->words[i]>>48),target=op&0xff00;
        int take=blocks==0 || (blocks==1&&(target==BLOCK_CNA||target==BLOCK_CORE)) ||
            (blocks==2&&(target==BLOCK_DPU||target==BLOCK_DPU_RDMA));
        if(take)put(p,src->words[i]);
    }
}
static void launch(co_program *p,unsigned bits,int tagged) {
    static const unsigned regs[]={0x1008,0x3008,0x4008,0x5008};
    static const unsigned targets[]={BLOCK_CNA,BLOCK_CORE,BLOCK_DPU,BLOCK_DPU_RDMA};
    for(unsigned j=0;j<4;j++)if(bits&(1u<<j))
        put(p,NPUOP(targets[j]|PC_OP_01|(tagged?PC_OP_ENABLE:0),1,regs[j]));
}
static void trailer(co_program *p,unsigned enable) {
    if(p->count&1)put(p,0);
    put(p,0);put(p,NPUOP(OP_REG_PC,0,PC_REGISTER_AMOUNTS));
    put(p,NPUOP(OP_40,0,0));put(p,NPUOP(OP_ENABLE,enable,PC_OPERATION_ENABLE));
}
static int compose(co_program *p,const char *arm,const co_program *a,const co_program *b,int tagged) {
    char base[40];int banks=strstr(arm,"-banks")!=NULL;
    if(banks){size_t len=(size_t)(strstr(arm,"-banks")-arm);if(len>=sizeof(base))return -1;
        memcpy(base,arm,len);base[len]=0;arm=base;}
    memset(p,0,sizeof(*p));
    if(!strcmp(arm,"direct-a")){config(p,a,0);launch(p,15,tagged);trailer(p,0);}
    else if(!strcmp(arm,"direct-b")){config(p,b,0);launch(p,12,tagged);trailer(p,0);}
    else if(!strcmp(arm,"raw-ab")){config(p,a,0);launch(p,15,tagged);config(p,b,0);trailer(p,0x18);}
    else if(!strcmp(arm,"raw-ba")){config(p,b,0);launch(p,12,tagged);config(p,a,0);trailer(p,0x1d);}
    else if(!strcmp(arm,"split")){
        config(p,a,1);config(p,b,0);
        /* Start independent CNA/CORE(A) and flying DPU(B). DPU(A) is queued next. */
        launch(p,15,tagged);config(p,a,2);trailer(p,0x18);
    } else return -1;
    if(banks){
        unsigned dpu=0,rdma=0;
        for(unsigned j=0;j<p->count;j++){
            unsigned reg=p->words[j]&0xffff;
            if(reg==0x4004||reg==0x5004){
                unsigned stage=reg==0x4004?dpu++:rdma++;
                unsigned ptr=stage?0xf:0x3e;
                p->words[j]=(p->words[j]&0xffff00000000ffffull)|((uint64_t)ptr<<16);
            }
        }
    }
    return 0;
}
static void split_programs(co_program p[2],const co_program *a,const co_program *b) {
    memset(p,0,2*sizeof(*p));
    config(&p[0],a,1);config(&p[0],b,2);trailer(&p[0],0x1d);
    config(&p[1],a,2);trailer(&p[1],0x18);
}
static void serial_programs(co_program p[2],const co_program *a,const co_program *b) {
    split_programs(p,a,b);
    p[0].words[p[0].count-1]=NPUOP(OP_ENABLE,0x18,PC_OPERATION_ENABLE);
    p[1].words[p[1].count-1]=NPUOP(OP_ENABLE,0x1d,PC_OPERATION_ENABLE);
}
static void early_programs(co_program p[2],const co_program *a,const co_program *b,int overlap) {
    memset(p,0,2*sizeof(*p));
    config(&p[0],a,1);
    put(&p[0],NPUOP(OP_REG_CNA,overlap?1:0,0x1008));
    put(&p[0],NPUOP(OP_REG_CORE,overlap?1:0,0x3008));
    config(&p[0],b,2);trailer(&p[0],0x18);
    config(&p[1],a,2);trailer(&p[1],overlap?0x18:0x1d);
}
static int score(inst_t *in,int used,size_t *bad,double *peak,size_t *nan) {
    _Float16 *got=malloc(in->ny*2);if(!got)return -1;
    int rc=co_download(&in->y.v,got);*bad=*nan=0;*peak=0;
    in->last_hash=1469598103934665603ull;
    if(!rc)for(size_t j=0;j<in->ny;j++) {
        uint16_t bits;memcpy(&bits,&got[j],2);
        in->last_hash=(in->last_hash^bits)*1099511628211ull;
        if(isnan((double)got[j])){(*nan)++;continue;}
        double e=fabs((double)got[j]-(double)in->ref[j]);
        if(!isfinite(e)||e>inst_tol(in))(*bad)++;
        if(isfinite(e)&&e>*peak)*peak=e;
    }
    free(got);
    int guard=co_guard_check(&in->x.b)|co_guard_check(&in->y.b);
    if(in->kind==INST_MM)guard|=co_guard_check(&in->w.b);
    return rc||guard|| (used?(*bad||*nan):(*nan!=in->ny));
}
static void cache_tables(co_program *,unsigned);
static int host(void) {
    co_program a={0},b={0},p;
    put(&a,NPUOP(OP_REG_CNA,123,0x1070));put(&a,NPUOP(OP_REG_CORE,456,0x3010));
    put(&a,NPUOP(OP_REG_DPU,14,0x4004));put(&a,NPUOP(BLOCK_DPU_RDMA|1,14,0x5004));
    trailer(&a,0x1d);
    put(&b,NPUOP(OP_REG_DPU,14,0x4004));put(&b,NPUOP(BLOCK_DPU_RDMA|1,14,0x5004));
    trailer(&b,0x18);
    const char *arms[]={"direct-a","direct-b","raw-ab","raw-ba","split"};
    for(unsigned k=0;k<2;k++)for(unsigned i=0;i<5;i++) {
        if(compose(&p,arms[i],&a,&b,k)||p.count>CO_MAX_WORDS||(p.count&1))return 1;
        if(p.words[p.count-2]!=NPUOP(OP_40,0,0))return 1;
        unsigned launches=0;
        for(unsigned j=0;j<p.count-4;j++) {
            unsigned reg=p.words[j]&0xffff;
            if(reg==0x1004||reg==0x3004||reg==PC_BASE_ADDRESS||reg==PC_REGISTER_AMOUNTS)return 1;
            if(reg==0x1008||reg==0x3008||reg==0x4008||reg==0x5008){
                if(((p.words[j]>>16)&0xffffffff)!=1)return 1;
                if(((p.words[j]>>48)&0x80)!=(k?0x80:0))return 1;
                launches++;
            }
        }
        if(launches!=(i==1||i==3?2u:4u))return 1;
    }
    co_program parallel[2],serial[2];
    split_programs(parallel,&a,&b);serial_programs(serial,&a,&b);
    for(unsigned i=0;i<2;i++){
        if(parallel[i].count!=serial[i].count)return 1;
        if(memcmp(parallel[i].words,serial[i].words,(parallel[i].count-1)*8))return 1;
    }
    co_program *chain=calloc(CO_MAX_PROGRAMS,sizeof(*chain));
    if(!chain)return 1;
    size_t total=0,offsets[CO_MAX_PROGRAMS];
    for(unsigned i=0;i<CO_MAX_PROGRAMS/2;i++)split_programs(&chain[2*i],&a,&b);
    for(unsigned i=0;i<CO_MAX_PROGRAMS;i++){offsets[i]=total;total+=rkt_chain_words(chain[i].count);}
    rocket_bo bo={.ptr=calloc(total,8),.size=total*8,.dma_address=0x10000000};
    rocket_task_desc desc[CO_MAX_PROGRAMS];
    if(!bo.ptr){free(chain);return 1;}
    for(unsigned i=0;i<CO_MAX_PROGRAMS;i++){
        unsigned next=i+1<CO_MAX_PROGRAMS?i+1:i;
        if(rkt_chain_pack_at(&bo,desc,(int)i,offsets[i],offsets[next],chain[i].words,chain[i].count,chain[next].count))return 1;
    }
    if(rkt_chain_verify(&bo,desc,CO_MAX_PROGRAMS))return 1;
    if(((uint64_t *)bo.ptr)[offsets[CO_MAX_PROGRAMS-1]+chain[CO_MAX_PROGRAMS-1].count-4]!=0)return 1;
    free(bo.ptr);
    co_program *table=&chain[0];memset(table,0,sizeof(*table));
    put(table,NPUOP(OP_REG_DPU,14,DPU_S_POINTER));
    for(unsigned bank=0;bank<2;bank++){
        put(table,NPUOP(OP_REG_DPU,(1u<<17)|(bank<<16),DPU_LUT_ACCESS_CFG));
        for(unsigned j=0;j<513;j++)put(table,NPUOP(OP_REG_DPU,j,DPU_LUT_ACCESS_DATA));
    }
    trailer(table,0x18);chain[1]=*table;
    unsigned before=table->count;cache_tables(chain,2);
    if(chain[0].count!=before||chain[1].count!=before-1028)return 1;
    if(chain[1].words[chain[1].count-1]!=NPUOP(OP_ENABLE,0x18,PC_OPERATION_ENABLE))return 1;
    free(chain);
    puts("regcmd_overlap_host PASS");return 0;
}
/* LUT tables are global ports, not banked geometry. Load the same table once
 * per Submit, omit unused MM port accesses, and fully configure each DPU group. */
static void cache_tables(co_program *programs,unsigned count) {
    int loaded=0;
    for(unsigned i=0;i<count;i++){
        co_program *p=&programs[i];unsigned data=0;
        for(unsigned j=0;j<p->count-4;j++)if((p->words[j]&0xffff)==DPU_LUT_ACCESS_DATA)data++;
        int upload=data==1026,keep=upload&&!loaded;
        unsigned out=0;
        for(unsigned j=0;j<p->count;j++){
            unsigned reg=p->words[j]&0xffff;
            if(!keep&&(reg==DPU_LUT_ACCESS_CFG||reg==DPU_LUT_ACCESS_DATA))continue;
            p->words[out++]=p->words[j];
        }
        p->count=out;if(upload)loaded=1;
    }
}

/* Matched-workload benchmark: all arms stay in one Submit and use distinct tile
 * outputs. The causal serial/overlap pair differs ONLY in two enables per tile. */
static int matched_main(int argc,char **argv) {
    if(argc!=10){fprintf(stderr,"Usage: %s matched M K N B_ELEMENTS PAIRS REPS WARMUPS DUMP_PREFIX\n",argv[0]);return 2;}
    unsigned v[7];
    for(unsigned i=0;i<7;i++){
        char *end=NULL;unsigned long x=strtoul(argv[i+2],&end,10);
        if(!end||*end||x>UINT32_MAX)return 2;
        v[i]=(unsigned)x;
    }
    unsigned m=v[0],k=v[1],n=v[2],bn=v[3],pairs=v[4],reps=v[5],warmups=v[6];
    if(!m||m>256||!k||k>OV_K||k%32||!n||n>256||n%32||
       !bn||bn>65520||bn%8||!pairs||pairs>CO_MAX_PROGRAMS/2||!reps||reps>1024||warmups>32)return 2;
    setenv("RG_PC_PP","0",1);
    int cache_lut=getenv("RG_CACHE_LUT")?atoi(getenv("RG_CACHE_LUT")):0;
    if(cache_lut!=0&&cache_lut!=1)return 2;
    int early_core=getenv("RG_EARLY_CORE")?atoi(getenv("RG_EARLY_CORE")):0;
    if(early_core!=0&&early_core!=1)return 2;
    matched_common_weights=1; /* pairs are row tiles of one [pairs*M,K] x [N,K]^T. */
    int fd=rocket_open();if(fd<0)return 77;
    inst_t *a=calloc(pairs,sizeof(*a)),*b=calloc(pairs,sizeof(*b));
    co_program *pack=calloc(2*pairs,sizeof(*pack));
    int rc=0;
    if(!a||!b||!pack){rc=ROCKET_E_NOMEM;goto done;}
    for(unsigned i=0;i<pairs;i++){
        inst_def da={INST_MM,m,k,n,17+101*i},db={INST_LUT,1,0,bn,107+113*i};
        matched_row_offset=(size_t)i*m;matched_b_offset=(size_t)i*bn;
        if((rc=inst_init(fd,&a[i],&da))||(rc=inst_init(fd,&b[i],&db)))goto done;
    }
    static const char *names[]={"a","b","ba","serial","overlap","bb","ab","a_then_b","separate"};
    const unsigned arms=sizeof(names)/sizeof(names[0]);
    for(unsigned r=0;r<warmups+reps&&!rc;r++)for(unsigned order=0;order<arms&&!rc;order++){
        unsigned arm=(order+r)%arms,np=0;
        for(unsigned i=0;i<pairs;i++){
            if((rc=co_sentinel(&a[i].y.b))||(rc=co_sentinel(&b[i].y.b)))break;
            if(arm==0)pack[np++]=a[i].prog;
            else if(arm==1)pack[np++]=b[i].prog;
            else if(arm==2){pack[np++]=b[i].prog;pack[np++]=a[i].prog;}
            else if(arm==3){if(early_core)early_programs(&pack[np],&a[i].prog,&b[i].prog,0);else serial_programs(&pack[np],&a[i].prog,&b[i].prog);np+=2;}
            else if(arm==4){if(early_core)early_programs(&pack[np],&a[i].prog,&b[i].prog,1);else split_programs(&pack[np],&a[i].prog,&b[i].prog);np+=2;}
            else if(arm==5){pack[np++]=b[i].prog;pack[np++]=b[i].prog;}
            else if(arm==6){pack[np++]=a[i].prog;pack[np++]=b[i].prog;}
            else pack[np++]=a[i].prog;
        }
        if((arm==7||arm==8)&&!rc)for(unsigned i=0;i<pairs;i++)pack[np++]=b[i].prog;
        if(rc)break;
        if(cache_lut)cache_tables(pack,np);
        char path[4096];int len=snprintf(path,sizeof(path),"%s.%s.submitted",argv[9],names[arm]);
        if(len<0||(size_t)len>=sizeof(path)){rc=2;break;}setenv("RG_DUMP_PATH",path,1);
        co_schedule_profile pr,parts[2]={{0}};unsigned submits=arm==8?2u:1u;
        uint64_t i0[3]={0},i1[3]={0};int irq_ok=!irqs(i0);
        uint64_t ioctl_sum=0,submit_sum=0;memset(&pr,0,sizeof(pr));
        for(unsigned sub=0;sub<submits;sub++){
            if(submits==2){
                int written=snprintf(path,sizeof(path),"%s.%s.%s.submitted",argv[9],names[arm],sub?"b":"a");
                if(written<0||(size_t)written>=sizeof(path)){rc=2;break;}
                setenv("RG_DUMP_PATH",path,1);
            }
            unsigned count=submits==2?pairs:np;
            rc=raw_execute(fd,pack+(submits==2?sub*pairs:0),count,count,&parts[sub]);
            pr.driver_service_ns+=parts[sub].driver_service_ns;
            ioctl_sum+=parts[sub].ioctl_end_ns-parts[sub].ioctl_begin_ns;
            submit_sum+=parts[sub].host_done_ns-parts[sub].host_begin_ns;
            pr.int_status=parts[sub].int_status;pr.actual_mask=parts[sub].actual_mask;
            if(rc)break;
        }
        irq_ok=irq_ok&&!irqs(i1);
        int valid=!rc;
        unsigned words=0;for(unsigned j=0;j<np;j++)words+=pack[j].count;
        printf("{\"type\":\"matched\",\"arm_count\":9,\"arm\":\"%s\",\"phase\":\"%s\",\"rep\":%u,\"m\":%u,\"k\":%u,\"n\":%u,\"b_elements\":%u,\"pairs\":%u,\"requested_reps\":%u,\"warmups\":%u,\"global_inputs\":1,\"cache_lut\":%d,\"early_core\":%d,\"task_count\":%u,\"submits\":%u,\"regcmd_words\":%u,\"result\":%d,\"service_ns\":%lld,\"ioctl_ns\":%llu,\"submit_ns\":%llu,\"int_status\":%u,\"actual_mask\":%u,\"irq_ok\":%d,\"irq\":[%llu,%llu,%llu],\"checks\":[",
            names[arm],r<warmups?"warm":"main",r<warmups?r:r-warmups,m,k,n,bn,pairs,reps,warmups,cache_lut,early_core,np,submits,words,rc,
            (long long)pr.driver_service_ns,(unsigned long long)ioctl_sum,
            (unsigned long long)submit_sum,pr.int_status,pr.actual_mask,irq_ok,
            (unsigned long long)(i1[0]-i0[0]),(unsigned long long)(i1[1]-i0[1]),(unsigned long long)(i1[2]-i0[2]));
        for(unsigned i=0;i<pairs;i++)for(unsigned kind=0;kind<2;kind++){
            int used=kind?(arm!=0):(arm!=1&&arm!=5);
            inst_t *in=kind?&b[i]:&a[i];size_t bad=0,nan=0;double peak=0;
            int err=score(in,used,&bad,&peak,&nan);valid&=!err;
            printf("%s{\"tile\":%u,\"kind\":\"%s\",\"used\":%d,\"valid\":%d,\"elements\":%zu,\"nan\":%zu,\"bad\":%zu,\"max_abs\":%.9g,\"hash\":\"%016llx\"}",
                i||kind?",":"",i,kind?"lut":"mm",used,!err,in->ny,nan,bad,peak,(unsigned long long)in->last_hash);
        }
        printf("],\"submissions\":[");
        for(unsigned sub=0;sub<submits;sub++)printf("%s{\"tasks\":%u,\"service_ns\":%lld,\"result\":%d,\"actual_mask\":%u,\"int_status\":%u}",
            sub?",":"",parts[sub].task_count,(long long)parts[sub].driver_service_ns,parts[sub].result,parts[sub].actual_mask,parts[sub].int_status);
        printf("],\"valid\":%d}\n",valid);fflush(stdout);
        if(!valid){usleep(20000);rc=1;}
    }
done:
    if(a)for(unsigned i=0;i<pairs;i++)inst_free(&a[i]);
    if(b)for(unsigned i=0;i<pairs;i++)inst_free(&b[i]);
    free(a);free(b);free(pack);rocket_close(fd);return rc?1:0;
}

int main(int argc,char **argv) {
    if(argc>1&&!strcmp(argv[1],"matched"))return matched_main(argc,argv);
    if(argc==2&&!strcmp(argv[1],"host"))return host();
    if(argc<5){fprintf(stderr,"Usage: %s ARM REPS HW_TASKS TAGGED [DUMP]\n",argv[0]);return 2;}
    const char *arm=argv[1];unsigned reps=(unsigned)strtoul(argv[2],0,10);
    unsigned tasks=(unsigned)strtoul(argv[3],0,10);int tagged=atoi(argv[4]);
    if(!reps||reps>1024||tasks<1||tasks>2||(tagged!=0&&tagged!=1))return 2;
    int fd=rocket_open();if(fd<0)return 77;
    inst_t in[2]={0};
    inst_def defs[]={{INST_MM,256,256,256,17},{INST_LUT,240,0,256,107}};
    if(getenv("RG_LUT_ROWS"))defs[1].rows=(unsigned)strtoul(getenv("RG_LUT_ROWS"),NULL,10);
    if(!defs[1].rows||defs[1].rows>240||defs[1].rows%4){rocket_close(fd);return 2;}
    int rc=inst_init(fd,&in[0],&defs[0]);if(!rc)rc=inst_init(fd,&in[1],&defs[1]);
    if(rc)goto done;
    int paired=!strcmp(arm,"paired");
    if(paired)arm="a";
    int split_chain=!strcmp(arm,"split-chain")||!strcmp(arm,"split-serial");
    int control=split_chain||!strcmp(arm,"a")||!strcmp(arm,"b")||!strcmp(arm,"ab")||!strcmp(arm,"ba")||!strcmp(arm,"bb");
    co_program p[2];unsigned n=1;
    if(split_chain){
        n=2;split_programs(p,&in[0].prog,&in[1].prog);
        if(!strcmp(arm,"split-serial"))serial_programs(p,&in[0].prog,&in[1].prog);
    } else if(control) {
        p[0]=in[!strcmp(arm,"b")||!strcmp(arm,"ba")||!strcmp(arm,"bb")].prog;
        if(strlen(arm)==2){n=2;p[1]=in[!strcmp(arm,"ab")||!strcmp(arm,"bb")].prog;}
    } else if(compose(&p[0],arm,&in[0].prog,&in[1].prog,tagged)){rc=2;goto done;}
    if(argc>5){
        FILE *dump=fopen(argv[5],"w");if(!dump){rc=2;goto done;}
        for(unsigned i=0;i<n;i++)for(unsigned j=0;j<p[i].count;j++)
            fprintf(dump,"%u %04u %016llx\n",i,j,(unsigned long long)p[i].words[j]);
        fclose(dump);
    }
    int use_a=strcmp(arm,"b")&&strcmp(arm,"direct-b")&&strcmp(arm,"bb");
    int use_b=strcmp(arm,"a")&&strcmp(arm,"direct-a");
    int pair_pp=getenv("RG_PAIR_PP")?atoi(getenv("RG_PAIR_PP")):-1;
    if(pair_pp < -1 || pair_pp > 1){rc=2;goto done;}
    unsigned cycle=pair_pp<0?14u:7u;
    for(unsigned r=0;r<reps*(paired?cycle:1u);r++) {
        if(paired){
            static const char *arms[]={"a","b","ab","ba","bb","split-chain","split-serial"};
            unsigned index=(r%cycle+r/cycle)%cycle;
            arm=arms[pair_pp<0?index/2:index];
            setenv("RG_PC_PP",(pair_pp<0?(int)(index%2):pair_pp)?"1":"0",1);
            p[0]=in[!strcmp(arm,"b")||!strcmp(arm,"ba")||!strcmp(arm,"bb")].prog;
            n=strlen(arm)==2?2:1;
            if(n==2)p[1]=in[!strcmp(arm,"ab")||!strcmp(arm,"bb")].prog;
            if(!strcmp(arm,"split-chain")){n=2;split_programs(p,&in[0].prog,&in[1].prog);}
            if(!strcmp(arm,"split-serial")){n=2;serial_programs(p,&in[0].prog,&in[1].prog);}
            use_a=strcmp(arm,"b")&&strcmp(arm,"bb");use_b=!!strcmp(arm,"a");
        }
        if((rc=co_sentinel(&in[0].y.b))||(rc=co_sentinel(&in[1].y.b)))break;
        uint64_t i0[3]={0},i1[3]={0};int irq_ok=!irqs(i0);
        if(argc>5){char path[4096];int len=snprintf(path,sizeof(path),"%s.%s.pp%s.submitted",argv[5],arm,getenv("RG_PC_PP")?getenv("RG_PC_PP"):"0");
            if(len<0||(size_t)len>=sizeof(path)){rc=2;break;}setenv("RG_DUMP_PATH",path,1);}
        co_schedule_profile pr;
        rc=raw_execute(fd,p,control?n:tasks,control?n:1,&pr);
        irq_ok=irq_ok&&!irqs(i1);
        int valid=1;
        printf("{\"type\":\"sample\",\"arm\":\"%s\",\"rep\":%u,\"hw_tasks\":%u,\"tagged\":%d,\"pc_pp\":%d,\"lut_rows\":%u,\"words\":%u,\"result\":%d,\"service_ns\":%lld,\"ioctl_ns\":%llu,\"int_status\":%u,\"actual_mask\":%u,\"irq_ok\":%d,\"irq\":[%llu,%llu,%llu],\"verify\":[",
            arm,paired?r/cycle:r,control?n:tasks,tagged,getenv("RG_PC_PP")?atoi(getenv("RG_PC_PP")):0,defs[1].rows,p[0].count,rc,(long long)pr.driver_service_ns,
            (unsigned long long)(pr.ioctl_end_ns-pr.ioctl_begin_ns),pr.int_status,pr.actual_mask,irq_ok,
            (unsigned long long)(i1[0]-i0[0]),(unsigned long long)(i1[1]-i0[1]),(unsigned long long)(i1[2]-i0[2]));
        for(unsigned j=0;j<2;j++) {
            size_t bad=0,nan=0;double peak=0;int v=score(&in[j],j?use_b:use_a,&bad,&peak,&nan);valid&=!v;
            printf("%s{\"used\":%d,\"valid\":%d,\"bad\":%zu,\"nan\":%zu,\"max_abs\":%.9g}",j?",":"",j?use_b:use_a,!v,bad,nan,peak);
        }
        printf("],\"valid\":%d}\n",valid);fflush(stdout);
        if(rc||!valid){
            /* Diagnostic observation only: never rehabilitate an early completion. */
            usleep(20000);
            for(unsigned j=0;j<2;j++){size_t bad=0,nan=0;double peak=0;int v=score(&in[j],j?use_b:use_a,&bad,&peak,&nan);
                printf("{\"type\":\"delayed\",\"inst\":%u,\"valid\":%d,\"nan\":%zu,\"bad\":%zu}\n",j,!v,nan,bad);}
            rc=1;break;
        }
    }
done:
    /* Failed raw commands may leave an engine active; do not submit another job here. */
    inst_free(&in[0]);inst_free(&in[1]);rocket_close(fd);return rc?1:0;
}
