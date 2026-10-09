// SPDX-License-Identifier: GPL-3.0-or-later
#define _GNU_SOURCE
#include "co_schedule.h"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

static pthread_barrier_t barrier;
static pthread_mutex_t log_lock=PTHREAD_MUTEX_INITIALIZER;
static FILE *json;
static pthread_mutex_t start_lock=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t start_cond=PTHREAD_COND_INITIALIZER;
static int start_state;
static unsigned workers,rounds,tasks;
static const char *scenario;
typedef struct{unsigned index;int rc;co_buffer input,output;co_view *in,*out;co_program *p;int fd,tracefd;_Float16 *data,*got;} worker;
static uint32_t mix(uint32_t x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
static _Float16 input_value(unsigned worker_id,unsigned task,size_t i){
    return (_Float16)(((int)(mix(worker_id*3337+task*131+(uint32_t)i)%257)-128)/16.0);
}
static _Float16 expected(_Float16 x){return (_Float16)fmin(1,fmax(0,(double)x/6+.5));}
static int prepare(worker *w){
    unsigned index=w->index;co_tensor t;
    int fd=rocket_open();w->fd=fd;if(fd<0){w->rc=77;return w->rc;}
    if(co_tensor_matrix(&t,64,64,CO_F16)){w->rc=1;return w->rc;}
    size_t slot=(t.bytes+128+15)&~(size_t)15;
    w->in=calloc(tasks,sizeof(*w->in));w->out=calloc(tasks,sizeof(*w->out));w->p=calloc(tasks,sizeof(*w->p));
    _Float16 *data=w->data=malloc(4096*2),*got=w->got=malloc(4096*2);
    if(!w->in||!w->out||!w->p||!data||!got){w->rc=1;return w->rc;}
    if(co_buffer_alloc(fd,slot*tasks,&w->input)||co_buffer_alloc(fd,slot*tasks,&w->output)){w->rc=1;return w->rc;}
    for(unsigned i=0;i<tasks;i++){
        if(co_view_init(&w->in[i],&w->input,slot*i+64,&t)||co_view_init(&w->out[i],&w->output,slot*i+64,&t)){w->rc=1;return w->rc;}
        for(size_t j=0;j<4096;j++)data[j]=input_value(index,i,j);
        if(co_upload_f16(&w->in[i],data)||co_emit_lut(&w->p[i],w->in[i],w->out[i],CO_HARDSIGMOID)){w->rc=1;return w->rc;}
    }
    const char *marker_path=getenv("RS_TRACE_MARKER");int tracefd=w->tracefd=marker_path?open(marker_path,O_WRONLY|O_CLOEXEC):-1;
    if(tracefd<0){w->rc=1;return w->rc;}
    return 0;
}
static void release(worker *w){
    if(w->tracefd>=0)close(w->tracefd);
    free(w->data);free(w->got);
    co_buffer_free(&w->input);co_buffer_free(&w->output);
    free(w->in);free(w->out);free(w->p);
    if(w->fd>=0)rocket_close(w->fd);
}
static void *run(void *arg){
    worker *w=arg;unsigned index=w->index;int fd=w->fd,tracefd=w->tracefd;
    _Float16 *got=w->got;
    pthread_mutex_lock(&start_lock);
    while(!start_state)pthread_cond_wait(&start_cond,&start_lock);
    int proceed=start_state>0;
    pthread_mutex_unlock(&start_lock);
    if(!proceed)return NULL;
    for(unsigned r=0;r<rounds;r++){
        unsigned mask=0,allowed=7;
        if(!strcmp(scenario,"serial"))mask=1u<<(r%3);
        else if(!strcmp(scenario,"same"))mask=1;
        else if(!strcmp(scenario,"split"))mask=1u<<(index%3);
        else if(!strcmp(scenario,"candidates")){allowed=3;co_schedule_pick(allowed,index+r,&mask);}
        else if(strcmp(scenario,"auto"))w->rc=1;
        if(co_sentinel(&w->output))w->rc=1;
        pthread_barrier_wait(&barrier);
        if(w->rc){pthread_barrier_wait(&barrier);continue;}
        uint64_t id=(uint64_t)getpid()*1000000ull+index*10000+r;
        co_schedule_profile p;int rc=co_schedule_execute(fd,w->p,tasks,mask,tracefd,id,&p);
        double error=0;size_t bad=0;
        if(!rc)for(unsigned i=0;i<tasks;i++){
            if(co_download(&w->out[i],got)){rc=1;break;}
            for(size_t j=0;j<4096;j++){
                double e=fabs((double)got[j]-expected(input_value(index,i,j)));
                if(!isfinite(e)||e>.002)bad++;
                if(isfinite(e)&&e>error)error=e;
            }
        }
        if(!rc)rc=co_guard_check(&w->input)||co_guard_check(&w->output);
        pthread_mutex_lock(&log_lock);
        fprintf(json,"{\"id\":%llu,\"pid\":%d,\"tid\":%u,\"worker\":%u,\"round\":%u,\"scenario\":\"%s\",\"allowed_mask\":%u,\"requested_mask\":%u,\"actual_mask\":%u,\"task_count\":%u,\"int_status\":%u,\"host_begin_ns\":%llu,\"ioctl_begin_ns\":%llu,\"ioctl_end_ns\":%llu,\"host_done_ns\":%llu,\"driver_service_ns\":%lld,\"result\":%d,\"bad\":%zu,\"max_abs\":%.9g}\n",
            (unsigned long long)id,getpid(),p.tid,index,r,scenario,allowed,mask,p.actual_mask,tasks,p.int_status,
            (unsigned long long)p.host_begin_ns,(unsigned long long)p.ioctl_begin_ns,(unsigned long long)p.ioctl_end_ns,
            (unsigned long long)p.host_done_ns,(long long)p.driver_service_ns,rc,bad,error);fflush(json);
        pthread_mutex_unlock(&log_lock);
        if(rc||bad)w->rc=1;
        pthread_barrier_wait(&barrier);
    }
    return NULL;
}
static int host(void){
    unsigned out=0;
    if(!co_schedule_mask_valid(0)||!co_schedule_mask_valid(1)||!co_schedule_mask_valid(2)||!co_schedule_mask_valid(4))return 1;
    for(unsigned m=3;m<8;m++)if(m!=4&&co_schedule_mask_valid(m))return 1;
    if(co_schedule_pick(3,0,&out)||out!=1||co_schedule_pick(3,1,&out)||out!=2||co_schedule_pick(3,2,&out)||out!=1)return 1;
    if(!co_schedule_pick(0,0,&out)||!co_schedule_pick(8,0,&out))return 1;
    puts("SCHEDULE_HOST PASS");return 0;
}
int main(int argc,char **argv){
    if(argc==2&&!strcmp(argv[1],"host"))return host();
    if(argc!=5){fprintf(stderr,"usage: schedule_probe serial|same|split|auto|candidates workers rounds tasks\n");return 1;}
    scenario=argv[1];workers=(unsigned)strtoul(argv[2],NULL,10);rounds=(unsigned)strtoul(argv[3],NULL,10);tasks=(unsigned)strtoul(argv[4],NULL,10);
    if(!workers||workers>8||!rounds||!tasks||tasks>64)return 1;
    if(strcmp(scenario,"serial")&&strcmp(scenario,"same")&&strcmp(scenario,"split")&&strcmp(scenario,"auto")&&strcmp(scenario,"candidates"))return 1;
    const char *file=getenv("RS_PROFILE_JSON");if(!file)return 1;json=fopen(file,"w");if(!json)return 1;
    worker *w=calloc(workers,sizeof(*w));pthread_t *threads=calloc(workers,sizeof(*threads));
    if(!w||!threads)return 1;
    int rc=0;unsigned created=0;
    for(unsigned i=0;i<workers;i++){w[i].fd=-1;w[i].tracefd=-1;}
    for(unsigned i=0;i<workers;i++){w[i].index=i;if(prepare(&w[i])){rc=1;break;}}
    pthread_barrier_init(&barrier,NULL,workers);
    if(!rc)for(unsigned i=0;i<workers;i++){
        if(pthread_create(&threads[i],NULL,run,&w[i])){rc=1;break;}created++;
    }
    pthread_mutex_lock(&start_lock);start_state=rc?-1:1;
    pthread_cond_broadcast(&start_cond);pthread_mutex_unlock(&start_lock);
    for(unsigned i=0;i<created;i++){pthread_join(threads[i],NULL);if(w[i].rc)rc=1;}
    for(unsigned i=0;i<workers;i++)release(&w[i]);
    fclose(json);pthread_barrier_destroy(&barrier);free(w);free(threads);
    printf("SCHEDULE scenario=%s workers=%u rounds=%u tasks=%u %s\n",scenario,workers,rounds,tasks,rc?"FAIL":"PASS");return rc;
}
