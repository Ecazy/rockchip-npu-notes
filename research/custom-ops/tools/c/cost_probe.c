// SPDX-License-Identifier: GPL-3.0-or-later
/* Controlled closed-loop jobs. Each worker owns its fd and every writable BO.
 * One JSON submit record is one blocking ioctl, operation records group them.
 * Numerical checking, input refresh, sentinel writes and logging are outside
 * the measured operation window; host_io includes upload/download explicitly. */
#define _GNU_SOURCE
#include "co_schedule.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MAX_BUFFERS (CO_MAX_PROGRAMS*2+8)
typedef struct {
    co_buffer buffers[MAX_BUFFERS]; unsigned nb;
    co_view inputs[64], outputs[64], weights, bias, identity, acc, affine;
    co_program programs[64]; unsigned np, ni, no;
    unsigned m,k,n,pc; int fd,kind,tracefd;
    _Float16 *x[2], *ref[2], *w, *b, *got;
    uint64_t logical_macs[64], physical_macs[64], bytes[64];
} load;
typedef struct {load l; unsigned index; int rc; uint64_t end_ns;} worker;
typedef struct {co_buffer buffers[3];co_view views[3];co_program *p;co_schedule_profile profile;uint64_t begin,end;int rc;} slot;
static pthread_mutex_t gate=PTHREAD_MUTEX_INITIALIZER, log_lock=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond=PTHREAD_COND_INITIALIZER;
static int started;
static unsigned worker_count,operations,warmups,m,k,n,pc,seed,base_mask,allowed;
static const char *workload,*strategy,*mode,*path;
static uint64_t start_ns;
static uint64_t request_prefix;
static int burst,affinity_base;
static pthread_mutex_t round_lock=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t round_cond=PTHREAD_COND_INITIALIZER;
static unsigned arrived,generation;static int aborted;
static int rendezvous(void){
    pthread_mutex_lock(&round_lock);unsigned g=generation;
    if(++arrived==worker_count){arrived=0;generation++;pthread_cond_broadcast(&round_cond);}
    while(g==generation&&!aborted)pthread_cond_wait(&round_cond,&round_lock);
    int ok=!aborted;pthread_mutex_unlock(&round_lock);return ok;
}
static void abort_round(void){pthread_mutex_lock(&round_lock);aborted=1;pthread_cond_broadcast(&round_cond);pthread_mutex_unlock(&round_lock);}
static uint64_t now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000ull+t.tv_nsec;}
static uint32_t mix(uint32_t x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
static int buffer(load *l,co_tensor t,co_view *v){
    if(l->nb>=MAX_BUFFERS)return ROCKET_E_SHAPE;
    co_buffer *b=&l->buffers[l->nb++];int rc=co_buffer_alloc(l->fd,t.bytes,b);
    return rc?rc:co_view_init(v,b,0,&t);
}
static int init(load *l,unsigned wi){
    l->fd=rocket_open();if(l->fd<0)return 77;
    const char *marker=getenv("COST_TRACE_MARKER");l->tracefd=marker?open(marker,O_WRONLY|O_CLOEXEC):-1;
    if(marker&&l->tracefd<0)return ROCKET_E_DEVICE;
    l->m=m;l->k=k;l->n=n;l->pc=pc;
    const char *selected=workload;char first[32];const char *plus=strchr(workload,'+');
    if(plus){size_t len=(size_t)(plus-workload);if(len>=sizeof(first))return ROCKET_E_SHAPE;
        memcpy(first,workload,len);first[len]=0;selected=wi%2?plus+1:first;}
    l->kind=!strcmp(selected,"lut")?0:!strcmp(selected,"matmul")?1:2;
    if(plus){l->pc=l->kind?1:pc;}
    unsigned m=l->m,k=l->k,n=l->n,pc=l->pc;
    co_tensor xt,yt,wt,it;
    int rc=co_tensor_matrix(&xt,m,l->kind?k:n,CO_F16);
    if(!rc)rc=co_tensor_matrix(&yt,m,n,CO_F16);
    if(rc)return rc;
    l->x[0]=malloc((size_t)m*(l->kind?k:n)*2);l->x[1]=malloc((size_t)m*(l->kind?k:n)*2);
    l->ref[0]=malloc((size_t)m*n*2);l->ref[1]=malloc((size_t)m*n*2);l->got=malloc((size_t)m*n*2);
    if(!l->x[0]||!l->x[1]||!l->ref[0]||!l->ref[1]||!l->got)return ROCKET_E_NOMEM;
    size_t nx=(size_t)m*(l->kind?k:n);
    for(size_t i=0;i<nx;i++){
        l->x[0][i]=(_Float16)(((int)(mix(seed+wi*977+(uint32_t)i)%17)-8)/(l->kind?32.0:2.0));
        l->x[1][i]=(_Float16)(-l->x[0][i]);
    }
    if(l->kind){
        rc=co_tensor_weight(&wt,n,k);if(rc)return rc;
        l->w=malloc((size_t)n*k*2);l->b=malloc(n*2);if(!l->w||!l->b)return ROCKET_E_NOMEM;
        for(size_t i=0;i<(size_t)n*k;i++)l->w[i]=(_Float16)(((int)(mix(seed+31+(uint32_t)i)%9)-4)/64.0);
        for(unsigned i=0;i<n;i++)l->b[i]=(_Float16)(((int)(i%7)-3)/32.0);
        if((rc=buffer(l,wt,&l->weights))||(rc=co_upload_f16(&l->weights,l->w)))return rc;
    }
    for(unsigned v=0;v<2;v++)for(unsigned r=0;r<m;r++)for(unsigned c=0;c<n;c++){
        double a=l->kind?0:l->x[v][(size_t)r*n+c];
        if(l->kind){a=0;for(unsigned q=0;q<k;q++)a+=(double)l->x[v][(size_t)r*k+q]*(double)l->w[(size_t)c*k+q];a=(_Float16)a;}
        if(l->kind==2)a=(_Float16)(a+(double)l->b[c]);
        if(l->kind!=1)a=fmin(1,fmax(0,a/6+.5));
        l->ref[v][(size_t)r*n+c]=(_Float16)a;
    }
    l->ni=l->no=l->kind==2?1:pc;
    for(unsigned i=0;i<l->ni;i++)if((rc=buffer(l,xt,&l->inputs[i]))||(rc=buffer(l,yt,&l->outputs[i])))return rc;
    if(l->kind==2){
        if((rc=co_tensor_weight(&it,yt.physical_cols,yt.physical_cols)))return rc;
        if((rc=buffer(l,yt,&l->acc))||(rc=buffer(l,yt,&l->affine))||
           (rc=buffer(l,yt,&l->bias))||(rc=buffer(l,it,&l->identity))||
           (rc=co_bias_upload(&l->bias,l->b))||(rc=co_identity_upload(&l->identity)))return rc;
        if((rc=co_emit_mm(&l->programs[0],l->inputs[0],l->weights,l->acc))||
           (rc=co_emit_ew(&l->programs[1],l->acc,l->bias,l->identity,l->affine,0)))return rc;
        l->logical_macs[0]=(uint64_t)m*k*n;l->physical_macs[0]=(uint64_t)xt.physical_rows*xt.physical_cols*yt.physical_cols;
        l->logical_macs[1]=0; /* semantic bias has no MAC; physical identity conv does */
        l->physical_macs[1]=(uint64_t)yt.physical_rows*yt.physical_cols*yt.physical_cols;
        l->np=2;
        size_t elements=yt.bytes/2;
        for(size_t off=0;off<elements;off+=32768){
            size_t count=elements-off;if(count>32768)count=32768;
            co_view src=l->affine,dst=l->outputs[0];
            if(elements>32768){co_tensor lt;if((rc=co_tensor_linear(&lt,(unsigned)count))||
                (rc=co_view_init(&src,l->affine.buffer,off*2,&lt))||
                (rc=co_view_init(&dst,l->outputs[0].buffer,off*2,&lt)))return rc;}
            if((rc=co_emit_lut(&l->programs[l->np++],src,dst,CO_HARDSIGMOID)))return rc;
        }
    }else{
        l->np=pc;
        for(unsigned i=0;i<pc;i++){
            if(l->kind){rc=co_emit_mm(&l->programs[i],l->inputs[i],l->weights,l->outputs[i]);
                l->logical_macs[i]=(uint64_t)m*k*n;l->physical_macs[i]=(uint64_t)xt.physical_rows*xt.physical_cols*yt.physical_cols;
            }else rc=co_emit_lut(&l->programs[i],l->inputs[i],l->outputs[i],CO_HARDSIGMOID);
            if(rc)return rc;
        }
    }
    for(unsigned i=0;i<l->np;i++){
        co_program *p=&l->programs[i];l->bytes[i]=p->output.tensor.bytes;
        for(unsigned j=0;j<p->nreads;j++)l->bytes[i]+=p->reads[j].tensor.bytes;
    }
    return 0;
}
static void destroy(load *l){
    for(unsigned i=0;i<l->nb;i++)co_buffer_free(&l->buffers[i]);
    for(unsigned i=0;i<2;i++){free(l->x[i]);free(l->ref[i]);}free(l->got);free(l->w);free(l->b);
    if(l->fd>=0)rocket_close(l->fd);
    if(l->tracefd>=0)close(l->tracefd);
}
static int refresh(load *l,unsigned variant){
    for(unsigned i=0;i<l->ni;i++){int rc=co_upload_f16(&l->inputs[i],l->x[variant]);if(rc)return rc;}
    return 0;
}
static int stamp(load *l){
    for(unsigned i=0;i<l->no;i++){int rc=co_sentinel(l->outputs[i].buffer);if(rc)return rc;}
    if(l->kind==2){int rc=co_sentinel(l->acc.buffer);if(!rc)rc=co_sentinel(l->affine.buffer);return rc;}return 0;
}
static int check(load *l,unsigned variant,size_t *bad,double *error,int downloaded){
    unsigned m=l->m,n=l->n;
    int rc=0;*bad=0;*error=0;
    for(unsigned i=0;i<l->no;i++){
        if(!downloaded||i){if((rc=co_download(&l->outputs[i],l->got)))return rc;}
        for(size_t j=0;j<(size_t)m*n;j++){
            double e=fabs((double)l->got[j]-(double)l->ref[variant][j]);
            double tol=l->kind==1?.01:.002;
            if(!isfinite(e)||e>tol)(*bad)++;
            if(isfinite(e)&&e>*error)*error=e;
        }
        co_view *v=&l->outputs[i];
        if((rc=rocket_bo_prep(l->fd,&v->buffer->bo,0,2000000000ull)))return rc;
        _Float16 *physical=(_Float16 *)((char *)v->buffer->bo.ptr+v->offset);
        for(unsigned r=0;r<v->tensor.physical_rows;r++)for(unsigned c=0;c<v->tensor.physical_cols;c++)if(r>=m||c>=n){
            double e=fabs((double)physical[co_index(&v->tensor,r,c)]-(l->kind==1?0:.5));
            if(!isfinite(e)||e>.002)(*bad)++;
        }
        if((rc=rocket_bo_fini(l->fd,&v->buffer->bo)))return rc;
    }
    for(unsigned i=0;i<l->nb;i++)if((rc=co_guard_check(&l->buffers[i])))return rc;
    return *bad?ROCKET_E_DEVICE:0;
}
static unsigned mask_for(unsigned wi,unsigned r){
    if(!strcmp(strategy,"auto"))return 0;
    if(!strcmp(strategy,"split")){unsigned out=0;co_schedule_pick(allowed,wi,&out);return out;}
    if(!strcmp(strategy,"poll")){unsigned out=0;co_schedule_pick(allowed,wi+r,&out);return out;}
    return base_mask;
}
static void deferred(worker *w){
    load *l=&w->l;unsigned budget=operations/worker_count+(w->index<operations%worker_count),total=warmups+budget;
    if(pc!=1||strcmp(mode,"chain")||strcmp(path,"device_ready")||strchr(workload,'+')){w->rc=1;abort_round();return;}
    uint64_t estimate=(uint64_t)total*((l->kind==2?3u:1u)*(l->outputs[0].tensor.bytes+CO_SLACK+CO_GUARD+4096)+l->np*sizeof(co_program));
    if(estimate>1073741824ull/worker_count){w->rc=ROCKET_E_SHAPE;abort_round();return;}
    slot *slots=calloc(total,sizeof(*slots));co_view alternate;int rc=slots?0:ROCKET_E_NOMEM;
    if(!rc)rc=buffer(l,l->inputs[0].tensor,&alternate);
    if(!rc)rc=co_upload_f16(&l->inputs[0],l->x[0]);
    if(!rc)rc=co_upload_f16(&alternate,l->x[1]);
    size_t extra_payload=0,extra_allocated=0;
    for(unsigned r=0;!rc&&r<total;r++){
        slot *s=&slots[r];s->p=calloc(l->np,sizeof(*s->p));if(!s->p){rc=ROCKET_E_NOMEM;break;}
        unsigned buffers=l->kind==2?3:1;
        for(unsigned i=0;!rc&&i<buffers;i++){
            rc=co_buffer_alloc(l->fd,l->outputs[0].tensor.bytes,&s->buffers[i]);
            if(!rc)rc=co_view_init(&s->views[i],&s->buffers[i],0,&l->outputs[0].tensor);
            if(!rc)rc=co_sentinel(&s->buffers[i]);
            extra_payload+=s->buffers[i].payload;extra_allocated+=s->buffers[i].bo.size;
        }
        co_view input=r%2?alternate:l->inputs[0];
        if(!rc&&l->kind==0)rc=co_emit_lut(&s->p[0],input,s->views[0],CO_HARDSIGMOID);
        if(!rc&&l->kind==1)rc=co_emit_mm(&s->p[0],input,l->weights,s->views[0]);
        if(!rc&&l->kind==2){
            rc=co_emit_mm(&s->p[0],input,l->weights,s->views[1]);
            if(!rc)rc=co_emit_ew(&s->p[1],s->views[1],l->bias,l->identity,s->views[2],0);
            size_t elements=l->outputs[0].tensor.bytes/2;unsigned index=2;
            for(size_t off=0;!rc&&off<elements;off+=32768){
                size_t count=elements-off;if(count>32768)count=32768;
                co_view src=s->views[2],dst=s->views[0];
                if(elements>32768){co_tensor t;rc=co_tensor_linear(&t,(unsigned)count);
                    if(!rc)rc=co_view_init(&src,&s->buffers[2],off*2,&t);
                    if(!rc)rc=co_view_init(&dst,&s->buffers[0],off*2,&t);}
                if(!rc)rc=co_emit_lut(&s->p[index++],src,dst,CO_HARDSIGMOID);
            }
        }
    }
    if(rc){w->rc=rc;abort_round();goto cleanup;}
    if(!rendezvous()){w->rc=1;goto cleanup;}
    for(unsigned r=0;r<total;r++){
        if(r==warmups&&!rendezvous()){w->rc=1;break;}
        if(burst&&!rendezvous()){w->rc=1;break;}
        slot *s=&slots[r];s->begin=now();
        uint64_t id=request_prefix|((uint64_t)w->index<<40)|((uint64_t)r<<8);
        s->rc=co_schedule_execute(l->fd,s->p,l->np,mask_for(w->index,r),l->tracefd,id,&s->profile);
        s->end=now();w->end_ns=s->end;
        if(s->rc){w->rc=s->rc;abort_round();break;}
    }
    if(!w->rc&&!rendezvous())w->rc=1;
    /* Every output is preserved until validation. This is not sparse sampling:
     * all logical/padding elements and all slot guards are checked after timing. */
    co_view original=l->outputs[0];
    for(unsigned r=0;r<total&&slots[r].end;r++){
        slot *s=&slots[r];co_schedule_profile *p=&s->profile;l->outputs[0]=s->views[0];
        size_t bad=0;double error=0;int checkrc=s->rc?s->rc:check(l,r%2,&bad,&error,0);
        for(unsigned i=0;!checkrc&&i<(l->kind==2?3u:1u);i++)checkrc=co_guard_check(&s->buffers[i]);
        if(checkrc)w->rc=checkrc;
        uint64_t lm=0,pm=0,bytes=0,commands=0;
        for(unsigned i=0;i<l->np;i++){lm+=l->logical_macs[i];pm+=l->physical_macs[i];bytes+=l->bytes[i];commands+=(uint64_t)s->p[i].count*8;}
        size_t payload=extra_payload,allocated=extra_allocated;
        for(unsigned i=0;i<l->nb;i++){payload+=l->buffers[i].payload;allocated+=l->buffers[i].bo.size;}
        uint64_t oid=request_prefix|((uint64_t)w->index<<32)|r;
        pthread_mutex_lock(&log_lock);
        printf("{\"type\":\"submit\",\"worker\":%u,\"round\":%u,\"warmup\":%s,\"operation_id\":%llu,\"request_id\":%llu,\"submit_index\":0,\"submit_count\":1,\"task_count\":%u,\"requested_mask\":%u,\"actual_mask\":%u,\"int_status\":%u,\"tid\":%u,\"host_begin_ns\":%llu,\"ioctl_begin_ns\":%llu,\"ioctl_end_ns\":%llu,\"host_done_ns\":%llu,\"driver_service_ns\":%lld,\"logical_macs\":%llu,\"physical_macs\":%llu,\"estimated_bytes\":%llu,\"command_bytes\":%llu,\"result\":%d,\"bad_count\":%zu,\"max_abs_error\":%.9g,\"guard_ok\":%s}\n",
            w->index,r,r<warmups?"true":"false",(unsigned long long)oid,(unsigned long long)p->request_id,l->np,p->requested_mask,p->actual_mask,p->int_status,p->tid,
            (unsigned long long)p->host_begin_ns,(unsigned long long)p->ioctl_begin_ns,(unsigned long long)p->ioctl_end_ns,(unsigned long long)p->host_done_ns,(long long)p->driver_service_ns,
            (unsigned long long)lm,(unsigned long long)pm,(unsigned long long)bytes,(unsigned long long)commands,checkrc,bad,error,checkrc?"false":"true");
        printf("{\"type\":\"operation\",\"worker\":%u,\"round\":%u,\"warmup\":%s,\"operation_id\":%llu,\"begin_ns\":%llu,\"upload_end_ns\":%llu,\"submit_end_ns\":%llu,\"end_ns\":%llu,\"program_count\":%u,\"submit_count\":1,\"completed_submits\":1,\"result\":%d,\"bad_count\":%zu,\"max_abs_error\":%.9g,\"payload_bytes\":%zu,\"allocated_tensor_bo_bytes\":%zu}\n",
            w->index,r,r<warmups?"true":"false",(unsigned long long)oid,(unsigned long long)s->begin,(unsigned long long)s->begin,(unsigned long long)s->end,(unsigned long long)s->end,l->np,checkrc,bad,error,payload,allocated);
        pthread_mutex_unlock(&log_lock);
    }
    l->outputs[0]=original;
cleanup:
    if(slots){for(unsigned r=0;r<total;r++){for(unsigned i=0;i<3;i++)co_buffer_free(&slots[r].buffers[i]);free(slots[r].p);}free(slots);}
}
static void *run(void *arg){
    worker *w=arg;load *l=&w->l;
    if(affinity_base>=0){cpu_set_t set;CPU_ZERO(&set);CPU_SET((affinity_base+(int)w->index)%8,&set);
        if(pthread_setaffinity_np(pthread_self(),sizeof(set),&set))w->rc=1;}
    pthread_mutex_lock(&gate);while(!started)pthread_cond_wait(&cond,&gate);int go=started>0;pthread_mutex_unlock(&gate);
    if(!go||w->rc){abort_round();return NULL;}
    const char *capture=getenv("COST_CAPTURE_MODE");if(capture&&!strcmp(capture,"deferred")){deferred(w);return NULL;}
    unsigned budget=operations/worker_count+(w->index<operations%worker_count);
    unsigned measured_round=0;
    for(unsigned r=0;r<warmups+budget;r++){
        if(r==warmups&&!rendezvous()){w->rc=1;break;}
        int iswarm=r<warmups;unsigned variant=r%2;
        int rc=stamp(l);if(!rc&&!strcmp(path,"device_ready"))rc=refresh(l,variant);
        if(rc){w->rc=rc;abort_round();break;}
        if(burst&&!rendezvous()){w->rc=1;break;}
        uint64_t opid=request_prefix|((uint64_t)w->index<<32)|r;
        co_schedule_profile profiles[64];memset(profiles,0,sizeof(profiles));
        unsigned submits=!strcmp(mode,"chain")?1:l->np,done=0;
        uint64_t ob=now(),upload_end=ob;
        if(!strcmp(path,"host_io")){rc=refresh(l,variant);upload_end=now();}
        for(unsigned s=0;!rc&&s<submits;s++){
            unsigned off=submits==1?0:s,count=submits==1?l->np:1;
            uint64_t id=request_prefix|((uint64_t)w->index<<40)|((uint64_t)r<<8)|s;
            rc=co_schedule_execute(l->fd,&l->programs[off],count,mask_for(w->index,r),l->tracefd,id,&profiles[s]);done++;
        }
        uint64_t submit_end=now();
        if(!rc&&!strcmp(path,"host_io"))for(unsigned i=0;i<l->no;i++)if((rc=co_download(&l->outputs[i],l->got)))break;
        uint64_t oe=now();w->end_ns=oe;
        size_t bad=0;double err=0;
        if(!rc)rc=check(l,variant,&bad,&err,!strcmp(path,"host_io")&&l->no==1);
        pthread_mutex_lock(&log_lock);
        for(unsigned s=0;s<done;s++){
            co_schedule_profile *p=&profiles[s];uint64_t lm=0,pm=0,bytes=0,commands=0;
            unsigned off=submits==1?0:s,count=submits==1?l->np:1;
            for(unsigned j=off;j<off+count;j++){lm+=l->logical_macs[j];pm+=l->physical_macs[j];bytes+=l->bytes[j];commands+=(uint64_t)l->programs[j].count*8;}
            printf("{\"type\":\"submit\",\"worker\":%u,\"round\":%u,\"warmup\":%s,\"operation_id\":%llu,\"request_id\":%llu,\"submit_index\":%u,\"submit_count\":%u,\"task_count\":%u,\"requested_mask\":%u,\"actual_mask\":%u,\"int_status\":%u,\"tid\":%u,\"host_begin_ns\":%llu,\"ioctl_begin_ns\":%llu,\"ioctl_end_ns\":%llu,\"host_done_ns\":%llu,\"driver_service_ns\":%lld,\"logical_macs\":%llu,\"physical_macs\":%llu,\"estimated_bytes\":%llu,\"command_bytes\":%llu,\"result\":%d,\"bad_count\":%zu,\"max_abs_error\":%.9g,\"guard_ok\":%s}\n",
                w->index,r,iswarm?"true":"false",(unsigned long long)opid,(unsigned long long)p->request_id,s,submits,count,p->requested_mask,p->actual_mask,p->int_status,p->tid,
                (unsigned long long)p->host_begin_ns,(unsigned long long)p->ioctl_begin_ns,(unsigned long long)p->ioctl_end_ns,(unsigned long long)p->host_done_ns,(long long)p->driver_service_ns,
                (unsigned long long)lm,(unsigned long long)pm,(unsigned long long)bytes,(unsigned long long)commands,rc,bad,err,rc?"false":"true");
        }
        size_t payload=0,allocated=0;for(unsigned i=0;i<l->nb;i++){payload+=l->buffers[i].payload;allocated+=l->buffers[i].bo.size;}
        printf("{\"type\":\"operation\",\"worker\":%u,\"round\":%u,\"warmup\":%s,\"operation_id\":%llu,\"begin_ns\":%llu,\"upload_end_ns\":%llu,\"submit_end_ns\":%llu,\"end_ns\":%llu,\"program_count\":%u,\"submit_count\":%u,\"completed_submits\":%u,\"result\":%d,\"bad_count\":%zu,\"max_abs_error\":%.9g,\"payload_bytes\":%zu,\"allocated_tensor_bo_bytes\":%zu}\n",
            w->index,r,iswarm?"true":"false",(unsigned long long)opid,(unsigned long long)ob,(unsigned long long)upload_end,(unsigned long long)submit_end,(unsigned long long)oe,l->np,submits,done,rc,bad,err,payload,allocated);
        fflush(stdout);pthread_mutex_unlock(&log_lock);
        if(rc){w->rc=rc;abort_round();break;}
        if(!iswarm)measured_round++;
    }
    if(measured_round!=budget&&!w->rc)w->rc=1;
    return NULL;
}
static int number(const char *s,unsigned *v,unsigned max){char *end;errno=0;unsigned long x=strtoul(s,&end,10);if(errno||!*s||*end||x>max)return 1;*v=(unsigned)x;return 0;}
int main(int argc,char **argv){
    if(argc==2&&!strcmp(argv[1],"--help")){puts("cost_probe workload m k n programs workers operations warmup seed chain|separate device_ready|host_io fixed|split|poll|auto mask allowed continuous|burst affinity_base 2");return 0;}
    if(argc!=18)return 2;
    const char *prefix=getenv("COST_REQUEST_PREFIX");
    if(prefix){unsigned v;if(number(prefix,&v,1048575))return 2;request_prefix=(uint64_t)v<<44;}
    workload=argv[1];mode=argv[10];path=argv[11];strategy=argv[12];burst=!strcmp(argv[15],"burst");
    /* argv16 is CPU base, argv17 reserved format version to reject old callers. */
    if(strcmp(argv[17],"2")||number(argv[2],&m,256)||number(argv[3],&k,256)||number(argv[4],&n,256)||number(argv[5],&pc,64)||
       number(argv[6],&worker_count,8)||number(argv[7],&operations,100000)||number(argv[8],&warmups,10000)||number(argv[9],&seed,0xffffffffu)||
       number(argv[13],&base_mask,4)||number(argv[14],&allowed,7)||!m||!k||!n||!pc||!worker_count||operations<worker_count)return 2;
    if(strcmp(workload,"lut")&&strcmp(workload,"matmul")&&strcmp(workload,"mm_bias_hsig")&&strcmp(workload,"lut+matmul")&&strcmp(workload,"matmul+lut"))return 2;
    if(strcmp(mode,"chain")&&strcmp(mode,"separate"))return 2;
    if(strcmp(path,"device_ready")&&strcmp(path,"host_io"))return 2;
    if(strcmp(strategy,"fixed")&&strcmp(strategy,"split")&&strcmp(strategy,"poll")&&strcmp(strategy,"auto"))return 2;
    if(!co_schedule_mask_valid(base_mask)||(!strcmp(strategy,"fixed")&&!base_mask)||!allowed)return 2;
    if(strcmp(argv[15],"continuous")&&strcmp(argv[15],"burst"))return 2;
    affinity_base=atoi(argv[16]);if(affinity_base < -1||affinity_base>7)return 2;
    if(burst&&operations%worker_count){fprintf(stderr,"burst requires equal worker budgets\n");return 2;}
    worker *ws=calloc(worker_count,sizeof(*ws));pthread_t *threads=calloc(worker_count,sizeof(*threads));if(!ws||!threads)return 1;
    for(unsigned i=0;i<worker_count;i++){ws[i].l.fd=-1;ws[i].l.tracefd=-1;}
    int rc=0;unsigned created=0;
    for(unsigned i=0;i<worker_count;i++){ws[i].index=i;if((rc=init(&ws[i].l,i)))break;}
    if(!rc)for(unsigned i=0;i<worker_count;i++){if(pthread_create(&threads[i],NULL,run,&ws[i])){rc=1;break;}created++;}
    start_ns=now();pthread_mutex_lock(&gate);started=rc?-1:1;pthread_cond_broadcast(&cond);pthread_mutex_unlock(&gate);
    for(unsigned i=0;i<created;i++){pthread_join(threads[i],NULL);if(ws[i].rc&&!rc)rc=ws[i].rc;}
    uint64_t end=0;for(unsigned i=0;i<worker_count;i++)if(ws[i].end_ns>end)end=ws[i].end_ns;
    printf("{\"type\":\"campaign\",\"pid\":%d,\"start_ns\":%llu,\"end_ns\":%llu,\"operations\":%u,\"workers\":%u,\"warmup_per_worker\":%u,\"result\":%d}\n",getpid(),(unsigned long long)start_ns,(unsigned long long)end,operations,worker_count,warmups,rc);
    for(unsigned i=0;i<worker_count;i++)destroy(&ws[i].l);
    free(ws);free(threads);return rc?rc==77?77:1:0;
}
