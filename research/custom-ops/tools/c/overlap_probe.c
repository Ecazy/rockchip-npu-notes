// SPDX-License-Identifier: GPL-3.0-or-later
/* Same-core overlap / concurrency probe for the RK3588 vendor submit path.
 *
 * Two suites share one harness:
 *   matmul-lut : A = 256^3 FP16 MatMul (CNA/CORE + DPU), B = DPU-only LUT
 *                (the original experiment; also carries the Setup-3
 *                PC_OPERATION_ENABLE ablation).
 *   gemm-gemv  : A = 256/256/256 GEMM, B = 1/256/256 GEMV (physical M padded
 *                to 12), amplified B8/B16, plus two-thread same-core and
 *                cross-core controls.
 *
 * Every submit is a blocking vendor ioctl through co_schedule_execute, pinned
 * to one core with mask 1 or 2. The driver service envelope (hw_elapse_time),
 * the last task's int_status, the per-core IRQ delta and an independent
 * numerical check are recorded per submit. The two-thread groups release both
 * workers on a barrier and record the pair window T_pair = max(done)-min(start).
 *
 * Module-level byte/event counters do not exist on this RK3588 path, so
 * overlap is inferred from the service differential, the same-kind controls
 * and the cross-core positive control, never from a counter. */
#define _GNU_SOURCE
#include "co_schedule.h"
#include "npu_hw.h"
#include "rocket_chain.h"
#include <ctype.h>
#include <execinfo.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define OV_MAX_INST  24
#define OV_MAX_PROGS 24
#define OV_GEMM_M    256u
#define OV_K         256u
#define OV_N         256u
#define OV_GEMV_M    1u
#define OV_LUT_ROWS  240u
#define OV_LUT_COLS  256u   /* physical 240 x 256 = 61440 < 8191*8 */
#define OV_MM_ENABLE       0x1Du
#define OV_MM_ENABLE_NODPU 0x15u
#define OV_LUT_ENABLE      0x18u

enum { INST_MM = 0, INST_LUT = 1 };

static FILE *records;
static unsigned checks, failures;
static uint64_t irq_delta[3];
static struct inst_t *insts;
static co_program core_prog;   /* DPU-disabled GEMM (matmul-lut explore) */

/* ---- shared helpers ---------------------------------------------------- */

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

static int patch_enable(co_program *p,uint32_t enable) {
    unsigned found=0;
    for(unsigned i=0;i<p->count;i++)
        if((unsigned)(p->words[i]&0xffffu)==PC_OPERATION_ENABLE) {
            p->words[i]=(p->words[i]&0xffff00000000ffffull)|((uint64_t)enable<<16);
            found++;
        }
    return found==1?0:ROCKET_E_SHAPE;
}
static uint32_t program_enable(const co_program *p) {
    for(unsigned i=0;i<p->count;i++)
        if((unsigned)(p->words[i]&0xffffu)==PC_OPERATION_ENABLE)
            return (uint32_t)((p->words[i]>>16)&0xffffffffu);
    return 0;
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
    co_program prog;
} inst_t;

typedef struct { int kind; unsigned rows,k,cols; uint32_t seed; } inst_def;

static int inst_init(int fd,inst_t *in,const inst_def *d) {
    memset(in,0,sizeof(*in));
    in->kind=d->kind; in->rows=d->rows; in->k=d->k; in->cols=d->cols;
    co_tensor xt,yt,wt; int rc;
    if(d->kind==INST_MM) {
        if(d->rows>OV_GEMM_M||!d->k||d->k>OV_K||!d->cols||d->cols>OV_N)return ROCKET_E_SHAPE;
        if((rc=co_tensor_matrix(&xt,d->rows,d->k,CO_F16)))return rc;
        if((rc=co_tensor_weight(&wt,d->cols,d->k)))return rc;
        if((rc=co_tensor_matrix(&yt,d->rows,d->cols,CO_F16)))return rc;
        in->physical_rows=xt.physical_rows;
        in->nx=(size_t)xt.physical_rows*(size_t)xt.physical_cols;
        in->nw=(size_t)d->cols*d->k;
        in->ny=(size_t)d->rows*d->cols;
        in->hx=malloc(in->nx*2); in->hw=malloc(in->nw*2); in->ref=malloc(in->ny*2);
        if(!in->hx||!in->hw||!in->ref)return ROCKET_E_NOMEM;
        for(size_t i=0;i<in->nx;i++)in->hx[i]=(_Float16)(((int)(mix32(d->seed+(uint32_t)i)%17)-8)/64.0);
        for(size_t i=0;i<in->nw;i++)in->hw[i]=(_Float16)(((int)(mix32(d->seed+31+(uint32_t)i)%9)-4)/128.0);
        for(unsigned r=0;r<d->rows;r++)for(unsigned c=0;c<d->cols;c++) {
            double a=0;
            for(unsigned q=0;q<d->k;q++)a+=(double)in->hx[(size_t)r*d->k+q]*(double)in->hw[(size_t)c*d->k+q];
            in->ref[(size_t)r*d->cols+c]=(_Float16)a;
        }
        if((rc=slot_init(fd,&xt,&in->x))||(rc=slot_init(fd,&wt,&in->w))||(rc=slot_init(fd,&yt,&in->y)))return rc;
        if((rc=co_upload_f16(&in->x.v,in->hx))||(rc=co_upload_f16(&in->w.v,in->hw)))return rc;
        rc=co_emit_mm(&in->prog,in->x.v,in->w.v,in->y.v);
    } else {
        unsigned r=d->rows,c=d->cols;
        in->k=0; in->physical_rows=r;
        if((rc=co_tensor_matrix(&xt,r,c,CO_F16)))return rc;
        if((rc=co_tensor_matrix(&yt,r,c,CO_F16)))return rc;
        in->nx=in->ny=(size_t)r*c;
        in->hx=malloc(in->nx*2); in->ref=malloc(in->ny*2);
        if(!in->hx||!in->ref)return ROCKET_E_NOMEM;
        for(size_t i=0;i<in->nx;i++) {
            in->hx[i]=(_Float16)(((int)(mix32(d->seed+7+(uint32_t)i)%33)-16)/4.0);
            in->ref[i]=hs((double)in->hx[i]);
        }
        if((rc=slot_init(fd,&xt,&in->x))||(rc=slot_init(fd,&yt,&in->y)))return rc;
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

/* ---- arms and suites --------------------------------------------------- */

typedef struct { int nprog; int inst[OV_MAX_PROGS]; int core[OV_MAX_PROGS]; } subprog;
typedef struct {
    const char *name;
    int pair;
    int nsub;
    subprog sub[2];
    int pair_inst[2];
    unsigned pair_mask[2];
} armspec;

static const armspec ML_ARMS[] = {
    {"A",   .nsub=1,.sub={{.nprog=1,.inst={0}}}},
    {"B",   .nsub=1,.sub={{.nprog=1,.inst={1}}}},
    {"S1",  .nsub=2,.sub={{.nprog=1,.inst={0}},{.nprog=1,.inst={1}}}},
    {"C_AB",.nsub=1,.sub={{.nprog=2,.inst={0,1}}}},
    {"C_BA",.nsub=1,.sub={{.nprog=2,.inst={1,0}}}},
    {"C_AA",.nsub=1,.sub={{.nprog=2,.inst={0,2}}}},
    {"C_BB",.nsub=1,.sub={{.nprog=2,.inst={1,3}}}},
};
static const armspec ML_EXPLORE[] = {
    {"S3_A_CORE",.nsub=1,.sub={{.nprog=1,.inst={0},.core={1}}}},
    {"S3_CORE_B",.nsub=1,.sub={{.nprog=2,.inst={0,1},.core={1,0}}}},
    {"S3_B_CORE",.nsub=1,.sub={{.nprog=2,.inst={1,0},.core={0,1}}}},
};

#define GEMV8  4,5,6,7,8,9,10,11
#define GEMV16 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
static const armspec GG_ARMS[] = {
    {"A",    .nsub=1,.sub={{.nprog=1,.inst={0}}}},
    {"B",    .nsub=1,.sub={{.nprog=1,.inst={1}}}},
    {"B8",   .nsub=1,.sub={{.nprog=8,.inst={GEMV8}}}},
    {"B16",  .nsub=1,.sub={{.nprog=16,.inst={GEMV16}}}},
    {"S_AB",  .nsub=2,.sub={{.nprog=1,.inst={0}},{.nprog=1,.inst={1}}}},
    {"S_AB8", .nsub=2,.sub={{.nprog=1,.inst={0}},{.nprog=8,.inst={GEMV8}}}},
    {"S_AB16",.nsub=2,.sub={{.nprog=1,.inst={0}},{.nprog=16,.inst={GEMV16}}}},
    {"C_AB",  .nsub=1,.sub={{.nprog=2,.inst={0,1}}}},
    {"C_BA",  .nsub=1,.sub={{.nprog=2,.inst={1,0}}}},
    {"C_AB8", .nsub=1,.sub={{.nprog=9,.inst={0,GEMV8}}}},
    {"C_B8A", .nsub=1,.sub={{.nprog=9,.inst={GEMV8,0}}}},
    {"C_AB16",.nsub=1,.sub={{.nprog=17,.inst={0,GEMV16}}}},
    {"C_B16A",.nsub=1,.sub={{.nprog=17,.inst={GEMV16,0}}}},
    {"C_AA",  .nsub=1,.sub={{.nprog=2,.inst={0,2}}}},
    {"C_BB",  .nsub=1,.sub={{.nprog=2,.inst={1,3}}}},
    {"P_same", .pair=1,.pair_inst={0,1},.pair_mask={1,1}},
    {"P_cross",.pair=1,.pair_inst={0,1},.pair_mask={1,2}},
};

static const inst_def ML_INSTS[] = {
    {INST_MM,OV_GEMM_M,OV_K,OV_N,1},
    {INST_LUT,OV_LUT_ROWS,0,OV_LUT_COLS,101},
    {INST_MM,OV_GEMM_M,OV_K,OV_N,201},
    {INST_LUT,OV_LUT_ROWS,0,OV_LUT_COLS,301},
};

static const inst_def GG_INSTS[] = {
    {INST_MM,OV_GEMM_M,OV_K,OV_N,1},
    {INST_MM,OV_GEMV_M,OV_K,OV_N,101},
    {INST_MM,OV_GEMM_M,OV_K,OV_N,201},
    {INST_MM,OV_GEMV_M,OV_K,OV_N,301},
    {INST_MM,OV_GEMV_M,OV_K,OV_N,401},{INST_MM,OV_GEMV_M,OV_K,OV_N,402},
    {INST_MM,OV_GEMV_M,OV_K,OV_N,403},{INST_MM,OV_GEMV_M,OV_K,OV_N,404},
    {INST_MM,OV_GEMV_M,OV_K,OV_N,405},{INST_MM,OV_GEMV_M,OV_K,OV_N,406},
    {INST_MM,OV_GEMV_M,OV_K,OV_N,407},{INST_MM,OV_GEMV_M,OV_K,OV_N,408},
    {INST_MM,OV_GEMV_M,OV_K,OV_N,409},{INST_MM,OV_GEMV_M,OV_K,OV_N,410},
    {INST_MM,OV_GEMV_M,OV_K,OV_N,411},{INST_MM,OV_GEMV_M,OV_K,OV_N,412},
    {INST_MM,OV_GEMV_M,OV_K,OV_N,413},{INST_MM,OV_GEMV_M,OV_K,OV_N,414},
    {INST_MM,OV_GEMV_M,OV_K,OV_N,415},{INST_MM,OV_GEMV_M,OV_K,OV_N,416},
};

typedef struct {
    const char *name;
    unsigned ninst;
    const inst_def *insts;
    unsigned narm;
    const armspec *arms;
    unsigned nexplore;
    const armspec *explore;
} suite_def;

static const suite_def SUITES[] = {
    {"matmul-lut",4,ML_INSTS,sizeof(ML_ARMS)/sizeof(ML_ARMS[0]),ML_ARMS,
                   sizeof(ML_EXPLORE)/sizeof(ML_EXPLORE[0]),ML_EXPLORE},
    {"gemm-gemv",20,GG_INSTS,sizeof(GG_ARMS)/sizeof(GG_ARMS[0]),GG_ARMS,0,NULL},
};

/* ---- submit / verify --------------------------------------------------- */

static void emit_submit(const char *suite,const char *arm,const char *phase,
                        unsigned rep,unsigned sub,unsigned tasks,unsigned nrw,
                        const co_schedule_profile *p) {
    fprintf(records,"{\"type\":\"submit\",\"suite\":\"%s\",\"arm\":\"%s\",\"phase\":\"%s\","
        "\"rep\":%u,\"sub_index\":%u,\"task_count\":%u,\"read_views\":%u,"
        "\"requested_mask\":%u,\"actual_mask\":%u,\"int_status\":%u,"
        "\"irq_delta\":[%llu,%llu,%llu],\"driver_service_ns\":%lld,"
        "\"ioctl_begin_ns\":%llu,\"ioctl_end_ns\":%llu,\"host_begin_ns\":%llu,"
        "\"host_done_ns\":%llu,\"result\":%d}\n",
        suite,arm,phase,rep,sub,tasks,nrw,p->requested_mask,p->actual_mask,p->int_status,
        (unsigned long long)irq_delta[0],(unsigned long long)irq_delta[1],
        (unsigned long long)irq_delta[2],
        (long long)p->driver_service_ns,(unsigned long long)p->ioctl_begin_ns,
        (unsigned long long)p->ioctl_end_ns,(unsigned long long)p->host_begin_ns,
        (unsigned long long)p->host_done_ns,p->result);
    fflush(records);
}

static void verify_arm(const char *suite,const armspec *a,const char *phase,unsigned rep) {
    unsigned used[OV_MAX_INST]={0};
    for(int s=0;s<a->nsub;s++)
        for(int j=0;j<a->sub[s].nprog;j++)used[a->sub[s].inst[j]]=1;
    for(unsigned i=0;i<OV_MAX_INST;i++) if(used[i]) {
        inst_t *in=&insts[i];
        _Float16 *got=malloc(in->ny*2);
        if(!got){failures++;continue;}
        int rc=co_download(&in->y.v,got);
        int guard=co_guard_check(&in->y.b);
        size_t nan=0,bad=0; double peak=0;
        for(size_t j=0;j<in->ny;j++) {
            if(isnan((double)got[j])){nan++;continue;}
            double e=fabs((double)got[j]-(double)in->ref[j]);
            if(!isfinite(e)||e>inst_tol(in))bad++;
            if(isfinite(e)&&e>peak)peak=e;
        }
        int unwritten=(nan==in->ny);
        int ok=rc==0&&bad==0&&guard==0;
        fprintf(records,"{\"type\":\"verify\",\"suite\":\"%s\",\"arm\":\"%s\",\"phase\":\"%s\","
            "\"rep\":%u,\"inst\":%u,\"kind\":\"%s\",\"physical_rows\":%u,\"used\":1,"
            "\"written\":%d,\"nan\":%zu,\"elements\":%zu,\"bad\":%zu,\"max_abs\":%.9g,"
            "\"guard_ok\":%d,\"download_result\":%d}\n",
            suite,a->name,phase,rep,i,in->kind==INST_LUT?"lut":"mm",in->physical_rows,
            !unwritten,nan,in->ny,bad,peak,guard==0,rc);
        fflush(records);
        if(!unwritten){checks++;if(!ok)failures++;}
        free(got);
    }
}

static int run_arm(int fd,const suite_def *suite,const armspec *a,const char *phase,
                   unsigned rep,uint64_t base) {
    static co_program pack[OV_MAX_PROGS];
    int rc=0;
    uint64_t w0=0,w1=0; long long svc=0;
    for(unsigned i=0;i<suite->ninst;i++) if((rc=co_sentinel(&insts[i].y.b)))return rc;
    w0=mono();
    for(int s=0;s<a->nsub;s++) {
        const subprog *sub=&a->sub[s];
        for(int j=0;j<sub->nprog;j++)
            pack[j]=sub->core[j]?core_prog:insts[sub->inst[j]].prog;
        unsigned reads=0;
        for(int j=0;j<sub->nprog;j++)reads+=pack[j].nreads;
        uint64_t i0[3],i1[3]; irqs(i0);
        co_schedule_profile pr;
        rc=co_schedule_execute(fd,pack,(unsigned)sub->nprog,1,-1,base+s,&pr);
        irqs(i1);
        for(int k=0;k<3;k++)irq_delta[k]=i1[k]-i0[k];
        svc+=pr.driver_service_ns;
        emit_submit(suite->name,a->name,phase,rep,(unsigned)s,(unsigned)sub->nprog,reads,&pr);
        if(rc)break;
    }
    verify_arm(suite->name,a,phase,rep);
    w1=mono();
    fprintf(records,"{\"type\":\"arm\",\"suite\":\"%s\",\"arm\":\"%s\",\"phase\":\"%s\","
        "\"rep\":%u,\"device_service_sum_ns\":%lld,\"wall_ns\":%llu,\"result\":%d}\n",
        suite->name,a->name,phase,rep,svc,(unsigned long long)(w1-w0),rc);
    fflush(records);
    return rc;
}

/* ---- concurrent pair groups -------------------------------------------- */
/* Barrier-synchronized processes, not threads. Two submits from one process
 * race the vendor driver's per-fd IOMMU domain switch during BO setup: the
 * single-threaded preflight writes correctly, but interleaved setup from two
 * fds leaves both jobs completing with no observable write (measured). Two
 * processes each holding one fd is the configuration the earlier multicore
 * campaign validated, and it still releases both submits on one barrier. */

#define OV_PAIR_MAX 1024

typedef struct {
    co_schedule_profile prof[2][OV_PAIR_MAX];
    unsigned char written[2][OV_PAIR_MAX], written2[2][OV_PAIR_MAX];
    size_t bad[2][OV_PAIR_MAX];
    double peak[2][OV_PAIR_MAX];
    int rc[2];
} pair_shared;

static int rd1(int fd){char c;return read(fd,&c,1)==1?0:-1;}
static int wr1(int fd){char c='1';return write(fd,&c,1)==1?0:-1;}

static void child_crash(int sig){
    void *bt[32]; int n=backtrace(bt,32);
    backtrace_symbols_fd(bt,n,2);
    _exit(128+sig);
}

static void pair_child(int t,const suite_def *suite,const armspec *a,pair_shared *sh,
                       unsigned total,int ready_w,int go_r,int done_w) {
    signal(SIGSEGV,child_crash); signal(SIGABRT,child_crash); signal(SIGBUS,child_crash);
    /* rknpu_gem.c marks BO mappings VM_DONTCOPY, so a forked child cannot touch
     * the parent's BO mappings. Each child opens its own fd and builds its own
     * instance; setup is outside every timed window. */
    int fd=rocket_open();
    inst_t in;
    if(fd<0||inst_init(fd,&in,&suite->insts[a->pair_inst[t]]))_exit(5);
    for(unsigned r=0;r<total;r++) {
        co_sentinel(&in.y.b);
        if(wr1(ready_w))_exit(2);
        if(rd1(go_r))_exit(3);
        sh->rc[t]=co_schedule_execute(fd,&in.prog,1,a->pair_mask[t],-1,r,&sh->prof[t][r]);
        size_t nan=0,bad=0; double peak=0;
        _Float16 *got=malloc(in.ny*2);
        int drc=got?co_download(&in.y.v,got):ROCKET_E_NOMEM;
        for(size_t j=0;got&&j<in.ny;j++) {
            if(isnan((double)got[j])){nan++;continue;}
            double e=fabs((double)got[j]-(double)in.ref[j]);
            if(!isfinite(e)||e>inst_tol(&in))bad++;
            if(isfinite(e)&&e>peak)peak=e;
        }
        sh->written2[t][r]=0;
        if(got&&nan==in.ny) {
            usleep(20000);
            if(co_download(&in.y.v,got)==0) {
                size_t n2=0;
                for(size_t j=0;j<in.ny;j++) if(isnan((double)got[j]))n2++;
                sh->written2[t][r]=(n2!=in.ny);
            }
        }
        sh->written[t][r]=(nan==in.ny)?0:1;
        sh->bad[t][r]=bad; sh->peak[t][r]=peak;
        free(got); (void)drc;
        if(wr1(done_w))_exit(4);
    }
    inst_free(&in); rocket_close(fd);
    _exit(0);
}

static int run_pair(const suite_def *suite,const armspec *a,unsigned samples,unsigned warmups) {
    unsigned total=warmups+samples;
    if(total>OV_PAIR_MAX)return ROCKET_E_SHAPE;
    pair_shared *sh=mmap(NULL,sizeof(*sh),PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0);
    if(sh==MAP_FAILED)return ROCKET_E_NOMEM;
    int ready[2][2],go[2][2],done[2][2];
    for(int t=0;t<2;t++)
        if(pipe(ready[t])||pipe(go[t])||pipe(done[t])){munmap(sh,sizeof(*sh));return ROCKET_E_NOMEM;}
    pid_t pids[2]={0,0};
    for(int t=0;t<2;t++) {
        fflush(NULL);
        pid_t p=fork();
        if(p==0)pair_child(t,suite,a,sh,total,ready[t][1],go[t][0],done[t][1]);
        if(p<0){ if(pids[0])kill(pids[0],SIGKILL); munmap(sh,sizeof(*sh)); return ROCKET_E_DEVICE; }
        pids[t]=p;
    }
    for(int t=0;t<2;t++){close(ready[t][1]);close(go[t][0]);close(done[t][1]);}
    for(unsigned r=0;r<total;r++) {
        uint64_t w0=mono();
        uint64_t i0[3],i1[3]; irqs(i0);
        int br=rd1(ready[0][0])||rd1(ready[1][0]);
        if(br){failures++;break;}
        int bw=wr1(go[0][1])||wr1(go[1][1]);
        int bd=bw?-1:((rd1(done[0][0])||rd1(done[1][0]))?1:0);
        irqs(i1);
        uint64_t w1=mono();
        if(bw||bd){failures++;break;}
        if(r<warmups)continue;
        const co_schedule_profile *pa=&sh->prof[0][r],*pb=&sh->prof[1][r];
        uint64_t begin=pa->ioctl_begin_ns<pb->ioctl_begin_ns?pa->ioctl_begin_ns:pb->ioctl_begin_ns;
        uint64_t end=pa->ioctl_end_ns>pb->ioctl_end_ns?pa->ioctl_end_ns:pb->ioctl_end_ns;
        long long sum=(long long)pa->driver_service_ns+(long long)pb->driver_service_ns;
        for(int t=0;t<2;t++) {
            const co_schedule_profile *p=&sh->prof[t][r];
            fprintf(records,"{\"type\":\"pair\",\"suite\":\"%s\",\"arm\":\"%s\",\"rep\":%u,"
                "\"thread\":%d,\"requested_mask\":%u,\"actual_mask\":%u,\"int_status\":%u,"
                "\"driver_service_ns\":%lld,\"ioctl_begin_ns\":%llu,\"ioctl_end_ns\":%llu,"
                "\"written\":%d,\"written2\":%d,\"bad\":%zu,\"max_abs\":%.9g,\"result\":%d}\n",
                suite->name,a->name,r,t,p->requested_mask,p->actual_mask,p->int_status,
                (long long)p->driver_service_ns,(unsigned long long)p->ioctl_begin_ns,
                (unsigned long long)p->ioctl_end_ns,sh->written[t][r],sh->written2[t][r],
                sh->bad[t][r],sh->peak[t][r],p->result);
        }
        fprintf(records,"{\"type\":\"pair_window\",\"suite\":\"%s\",\"arm\":\"%s\",\"rep\":%u,"
            "\"t_pair_ns\":%llu,\"sum_service_ns\":%lld,\"service_overlap_ns\":%lld,"
            "\"wall_ns\":%llu,\"irq_delta\":[%llu,%llu,%llu]}\n",
            suite->name,a->name,r,(unsigned long long)(end-begin),sum,sum-(long long)(end-begin),
            (unsigned long long)(w1-w0),
            (unsigned long long)(i1[0]-i0[0]),(unsigned long long)(i1[1]-i0[1]),
            (unsigned long long)(i1[2]-i0[2]));
        fflush(records);
        checks+=2;
        if(sh->rc[0]||sh->rc[1]||!sh->written[0][r]||!sh->written[1][r]||
           sh->bad[0][r]||sh->bad[1][r])failures++;
    }
    for(int t=0;t<2;t++){close(ready[t][0]);close(go[t][1]);close(done[t][0]);}
    for(int t=0;t<2;t++){int st=0;waitpid(pids[t],&st,0);
        if(!WIFEXITED(st)||WEXITSTATUS(st)) {
            failures++;
            fprintf(stderr,"pair child %d: exited=%d code=%d signal=%d\n",t,
                WIFEXITED(st),WIFEXITED(st)?WEXITSTATUS(st):-1,
                WIFSIGNALED(st)?WTERMSIG(st):0);
        }}
    munmap(sh,sizeof(*sh));
    return 0;
}

/* ---- host-only layout test -------------------------------------------- */

static int host(void) {
    co_tensor xt,wt,yt,lt,lo;
    int rc=co_tensor_matrix(&xt,OV_GEMM_M,OV_K,CO_F16);
    if(!rc)rc=co_tensor_weight(&wt,OV_N,OV_K);
    if(!rc)rc=co_tensor_matrix(&yt,OV_GEMM_M,OV_N,CO_F16);
    if(!rc)rc=co_tensor_matrix(&lt,OV_LUT_ROWS,OV_LUT_COLS,CO_F16);
    if(!rc)rc=co_tensor_matrix(&lo,OV_LUT_ROWS,OV_LUT_COLS,CO_F16);
    if(rc){fprintf(stderr,"host tensors rc=%d\n",rc);return 1;}
    co_buffer bx={.payload=xt.bytes,.bo=(rocket_bo){.handle=1,.dma_address=0x100000,.size=xt.bytes}};
    co_buffer bw={.payload=wt.bytes,.bo=(rocket_bo){.handle=2,.dma_address=0x200000,.size=wt.bytes}};
    co_buffer by={.payload=yt.bytes,.bo=(rocket_bo){.handle=3,.dma_address=0x300000,.size=yt.bytes}};
    co_buffer bl={.payload=lt.bytes,.bo=(rocket_bo){.handle=4,.dma_address=0x400000,.size=lt.bytes}};
    co_buffer bl2={.payload=lo.bytes,.bo=(rocket_bo){.handle=5,.dma_address=0x500000,.size=lo.bytes}};
    co_view vx,vw,vy,vl,vo;
    if((rc=co_view_init(&vx,&bx,0,&xt)))return 1;
    if((rc=co_view_init(&vw,&bw,0,&wt)))return 1;
    if((rc=co_view_init(&vy,&by,0,&yt)))return 1;
    if((rc=co_view_init(&vl,&bl,0,&lt)))return 1;
    if((rc=co_view_init(&vo,&bl2,0,&lo)))return 1;
    co_program pm,pl,patch;
    if((rc=co_emit_mm(&pm,vx,vw,vy))||(rc=co_emit_lut(&pl,vl,vo,CO_HARDSIGMOID))) {
        fprintf(stderr,"host emit rc=%d\n",rc);return 1;
    }
    if(program_enable(&pm)!=OV_MM_ENABLE||program_enable(&pl)!=OV_LUT_ENABLE) {
        fprintf(stderr,"HOST enable mismatch mm=%#x lut=%#x\n",program_enable(&pm),program_enable(&pl));return 1;
    }
    patch=pm;
    if(patch_enable(&patch,OV_MM_ENABLE_NODPU)||program_enable(&patch)!=OV_MM_ENABLE_NODPU)return 1;
    size_t off0=rkt_chain_words(pm.count),off1=rkt_chain_words(pl.count),words=off0+off1;
    uint64_t *buf=calloc(1,words*8); if(!buf)return 1;
    rocket_bo rcbo={.handle=9,.dma_address=0x900000,.ptr=buf,.size=words*8};
    rocket_task_desc desc[2];
    rc=rkt_chain_pack_at(&rcbo,desc,0,0,off0,pm.words,pm.count,pl.count)||
       rkt_chain_pack_at(&rcbo,desc,1,off0,off0,pl.words,pl.count,pl.count)||
       rkt_chain_verify(&rcbo,desc,2);
    printf("HOST_OVERLAP %s mm_words=%u lut_words=%u enable_mm=%#x enable_lut=%#x\n",
        rc?"FAIL":"PASS",pm.count,pl.count,OV_MM_ENABLE,OV_LUT_ENABLE);
    free(buf);
    return rc?1:0;
}

/* ---- main -------------------------------------------------------------- */

int main(int argc,char **argv) {
    if(argc==2&&!strcmp(argv[1],"host"))return host();
    const char *suitename=argc>1?argv[1]:"gemm-gemv";
    const suite_def *suite=NULL;
    for(unsigned i=0;i<sizeof(SUITES)/sizeof(SUITES[0]);i++)
        if(!strcmp(SUITES[i].name,suitename))suite=&SUITES[i];
    unsigned samples=argc>2?(unsigned)strtoul(argv[2],NULL,0):64;
    unsigned seed=argc>3?(unsigned)strtoul(argv[3],NULL,0):1u;
    unsigned warmups=argc>4?(unsigned)strtoul(argv[4],NULL,0):8u;
    const char *maskenv=getenv("RKNPU_CORE_MASK");
    unsigned mask=maskenv?(unsigned)strtoul(maskenv,NULL,0):1u;
    const char *path=getenv("OL_PROBE_JSON");
    const char *exenv=getenv("OL_EXPLORE");
    int explore=suite&&suite->nexplore&&exenv&&exenv[0]&&strcmp(exenv,"0")!=0;
    if(!suite||samples<1||samples>4096||warmups>4096||!path||mask!=1u) {
        fprintf(stderr,"usage: RKNPU_CORE_MASK=1 OL_PROBE_JSON=file [OL_EXPLORE=1] %s "
            "<matmul-lut|gemm-gemv> [samples>=1] [seed] [warmups]\n",argv[0]);
        return 2;
    }
    records=fopen(path,"w"); if(!records){perror(path);return 1;}
    signal(SIGPIPE,SIG_IGN);
    inst_t local[OV_MAX_INST]; insts=local;
    int fd=rocket_open(); if(fd<0){fprintf(stderr,"no NPU\n");fclose(records);return 77;}
    int rc=0;
    for(unsigned i=0;i<suite->ninst;i++)
        if((rc=inst_init(fd,&insts[i],&suite->insts[i])))goto done;
    if(suite->nexplore) {
        core_prog=insts[0].prog;
        if((rc=patch_enable(&core_prog,OV_MM_ENABLE_NODPU)))goto done;
    }
    printf("ENABLE suite=%s insts=%u mm=%#x",suite->name,suite->ninst,program_enable(&insts[0].prog));
    if(suite->ninst>1&&suite->insts[1].kind==INST_LUT)printf(" lut=%#x",program_enable(&insts[1].prog));
    printf(" physical_rows(A)=%u physical_rows(B)=%u\n",insts[0].physical_rows,
        suite->ninst>1?insts[1].physical_rows:0);

    /* Chain/single arms: warm up, then sample, arm order rotated per round. */
    uint64_t base=((uint64_t)seed<<32)|(uint64_t)getpid();
    for(unsigned round=0;round<warmups+samples&&!rc;round++) {
        const char *phase=round<warmups?"warm":"main";
        for(unsigned i=0;i<suite->narm&&!rc;i++) {
            const armspec *a=&suite->arms[(i+round)%suite->narm];
            if(a->pair)continue;
            rc=run_arm(fd,suite,a,phase,round,base+((uint64_t)round<<16)+i);
        }
    }
    /* Concurrent pair groups run once; each child builds its own fd/instance. */
    for(unsigned i=0;i<suite->narm&&!rc;i++) {
        const armspec *a=&suite->arms[i];
        if(!a->pair)continue;
        rc=run_pair(suite,a,samples,warmups);
    }
    if(explore&&!rc) {
        for(unsigned i=0;i<suite->nexplore;i++) {
            int erc=run_arm(fd,suite,&suite->explore[i],"explore",0,base+0x800000+i);
            if(erc)printf("EXPLORE %s result=%d (recorded)\n",suite->explore[i].name,erc);
        }
    }
done:
    for(unsigned i=0;i<suite->ninst;i++)inst_free(&insts[i]);
    rocket_close(fd);
    fprintf(records,"{\"type\":\"summary\",\"suite\":\"%s\",\"samples\":%u,\"warmups\":%u,"
        "\"seed\":%u,\"checks\":%u,\"failures\":%u,\"explore\":%s,\"status\":\"%s\"}\n",
        suite->name,samples,warmups,seed,checks,failures+(rc!=0),
        explore?"true":"false",rc?"FAIL":"PASS");
    fclose(records);
    printf("OVERLAP_SUMMARY suite=%s samples=%u checks=%u status=%s\n",
        suite->name,samples,checks,rc?"FAIL":"PASS");
    return rc?1:0;
}
