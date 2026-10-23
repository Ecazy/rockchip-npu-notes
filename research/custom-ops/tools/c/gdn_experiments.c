// SPDX-License-Identifier: GPL-3.0-or-later
#define _POSIX_C_SOURCE 200809L
#include "co_gdn.h"
#include "co_gdn_cpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#define TRY(x) do{int z=(x);if(z){fprintf(stderr,"line %d rc=%d\n",__LINE__,z);rc=z;goto done;}}while(0)
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"line %d %s\n",__LINE__,#x);rc=1;goto done;}}while(0)
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec/1e3;}
static void fill(_Float16 *x,size_t n,unsigned seed,float scale){for(size_t i=0;i<n;i++){seed=1664525u*seed+1013904223u;x[i]=(_Float16)(((int)(seed>>16)-32768)/32768.f*scale);}}
static unsigned read_khz(const char *path,int hz){unsigned f=0;FILE *fp=fopen(path,"r");if(fp){if(fscanf(fp,"%u",&f)!=1)f=0;fclose(fp);}if(hz)f/=1000;return f;}
static unsigned cpu_freq(void){return read_khz("/sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq",0);}
static unsigned npu_freq(void){return read_khz("/sys/class/devfreq/fdab0000.npu/cur_freq",1);}
static int save(const char *root,const char *name,const void *p,size_t bytes){char path[1024];if(snprintf(path,sizeof(path),"%s/%s",root,name)>=(int)sizeof(path))return 1;FILE *f=fopen(path,"wb");if(!f)return 1;int rc=fwrite(p,1,bytes,f)!=bytes;if(fclose(f))rc=1;return rc;}
typedef struct {co_buffer b;co_view v;} tensor;
static int make(int fd,tensor *t,unsigned r,unsigned c,int weight){co_tensor d;int rc=weight?co_tensor_weight(&d,r,c):co_tensor_matrix(&d,r,c,CO_F16);if(!rc)rc=co_buffer_alloc(fd,d.bytes,&t->b);if(!rc)rc=co_view_init(&t->v,&t->b,0,&d);return rc;}
static int native_programs(co_program *p,unsigned *n,co_view x,co_view y,_Float16 a){
    *n=0;size_t left=x.tensor.bytes/2,offset=0;
    while(left){unsigned count=left>32768?32768:(unsigned)left;co_tensor d;co_view xv,yv;int rc=co_tensor_linear(&d,count);if(rc)return rc;
        if((rc=co_view_init(&xv,x.buffer,offset,&d))||(rc=co_view_init(&yv,y.buffer,offset,&d))||
           (rc=co_emit_scale_native(p+(*n)++,xv,yv,a)))return rc;
        left-=count;offset+=count*2;
    }return 0;
}
/* Scalar A/B/C/D: both state matrix and beta vector geometry. x resident in its
 * own engine; dynamic coefficient materialization/upload included, output
 * download/validation outside timing for every NPU arm. All guards timed. */
static int scalars(int fd,unsigned n,unsigned vector,int smoke){
    int rc=0;unsigned cols=vector?1:n;size_t count=(size_t)n*cols;tensor t[5]={0};co_program p[2];
    _Float16 *x=calloc(count,2),*y=calloc(count,2),*coef=calloc((size_t)n*n,2),*cpu=calloc(count,2);
    REQUIRE(x&&y&&coef&&cpu);fill(x,count,1101,2);
    TRY(make(fd,&t[0],n,cols,0));TRY(make(fd,&t[1],n,cols,0));TRY(make(fd,&t[2],cols,cols,1));
    TRY(make(fd,&t[3],n,cols,0));unsigned pc=t[0].v.tensor.physical_cols;TRY(make(fd,&t[4],pc,pc,1));TRY(co_identity_upload(&t[4].v));
    TRY(co_upload_f16(&t[0].v,x));
    unsigned reps=smoke?8:32;
    const _Float16 scales[]={0,1,-1,.9375,.6,2,.00006103515625,-.5};
    for(unsigned rep=0;rep<reps;rep++)for(unsigned order=0;order<(smoke?1u:4u);order++){
        unsigned arm=smoke?2:(rep+order)%4;_Float16 a=scales[rep%8];unsigned tasks=0;double prep=0,upload=0,emit=0,execute=0,guard=0,st,total=now();
        if(arm==0){st=now();memset(coef,0,(size_t)cols*cols*2);for(unsigned j=0;j<cols;j++)coef[(size_t)j*cols+j]=a;prep=now()-st;st=now();TRY(co_upload_f16(&t[2].v,coef));upload=now()-st;st=now();TRY(co_emit_mm(p,t[0].v,t[2].v,t[1].v));tasks=1;emit=now()-st;}
        else if(arm==1){st=now();for(size_t i=0;i<count;i++)coef[i]=a;prep=now()-st;st=now();TRY(co_upload_f16(&t[3].v,coef));upload=now()-st;st=now();TRY(co_emit_ew(p,t[0].v,t[3].v,t[4].v,t[1].v,1));tasks=1;emit=now()-st;}
        else if(arm==2){st=now();TRY(native_programs(p,&tasks,t[0].v,t[1].v,a));emit=now()-st;}
        if(arm!=3){st=now();TRY(co_execute(fd,p,tasks,CO_CHAIN));execute=now()-st;st=now();for(unsigned i=0;i<5;i++)TRY(co_guard_check(&t[i].b));guard=now()-st;total=now()-total;TRY(co_download(&t[1].v,y));}
        else {st=now();for(unsigned batch=0;batch<64;batch++){co_gdn_cpu_scale_f16(x,cpu,count,a);__asm__ volatile("" : : "r"(cpu) : "memory");}total=(now()-st)/64;memcpy(y,cpu,count*2);}
        double peak=0;for(size_t i=0;i<count;i++){double ref=(_Float16)((float)x[i]*(float)a),err=fabs((double)y[i]-ref);if(err>peak)peak=err;REQUIRE(isfinite((float)y[i])&&err<=.00001+.001*fabs(ref));}
        printf("{\"kind\":\"scalar\",\"n\":%u,\"vector\":%u,\"rep\":%u,\"arm\":%u,\"total_us\":%.3f,\"prepare_us\":%.3f,\"upload_us\":%.3f,\"emit_us\":%.3f,\"execute_api_us\":%.3f,\"guard_us\":%.3f,\"tasks\":%u,\"max_abs\":%.9g,\"freq_khz\":%u,\"npu_freq_khz\":%u}\n",n,vector,rep,arm,total,prep,upload,emit,execute,guard,tasks,peak,cpu_freq(),npu_freq());
    }
done:for(unsigned i=0;i<5;i++)co_buffer_free(&t[i].b);free(x);free(y);free(coef);free(cpu);return rc;
}
static const char *names[]={"cpu16","cpu32","npu_alloc","npu_reuse","beta_fold","merged","native_alpha"};
enum {ARMS=7,STEPS=32,BLOCKS=5};
static void profile_json(const co_gdn_profile *p){
    printf(",\"profile\":{\"total_us\":%.3f,\"allocation_us\":%.3f,\"prepare_us\":%.3f,\"pack_k_us\":%.3f,\"pack_q_us\":%.3f,\"pack_v_us\":%.3f,\"pack_coeff_us\":%.3f,\"bo_sync_copy_us\":%.3f,\"emit_us\":%.3f,\"execute_api_us\":%.3f,\"command_allocation_us\":%.3f,\"command_us\":%.3f,\"submit_wait_us\":%.3f,\"output_sync_us\":%.3f,\"guard_us\":%.3f,\"download_us\":%.3f,\"host_allocations\":%u,\"tasks\":%u}",p->total_us,p->host_allocation_us,p->prepare_us,p->pack_k_us,p->pack_q_us,p->pack_v_us,p->pack_coeff_us,p->bo_sync_copy_us,p->emit_us,p->execute_api_us,p->executor.allocation_us,p->executor.command_us,p->executor.submit_wait_us,p->executor.output_sync_us,p->guard_us,p->download_us,p->host_allocations,p->tasks);
}
static int recurrence(int fd,const char *root,unsigned n){
    int rc=0;size_t ns=(size_t)n*n;co_gdn *gpu[ARMS]={0};co_gdn_cpu *cpu[2]={0};
    _Float16 *initial=calloc(ns,2),*k=calloc(STEPS*n,2),*q=calloc(STEPS*n,2),*v=calloc(STEPS*n,2),*hs=calloc(ns,2);
    float *outputs=calloc((size_t)ARMS*STEPS*n,4),*states=calloc((size_t)ARMS*ns,4);_Float16 gates[STEPS*2],ho[256];float fo[256];char name[128];
    REQUIRE(initial&&k&&q&&v&&hs&&outputs&&states);
    TRY(co_gdn_cpu_create(n,n,CO_GDN_CPU_F16,&cpu[0]));TRY(co_gdn_cpu_create(n,n,CO_GDN_CPU_F32,&cpu[1]));
    for(unsigned arm=2;arm<ARMS;arm++){
        co_gdn_config cfg={.reuse=arm!=2,.graph=arm==4?CO_GDN_BETA_FOLD:arm==5?CO_GDN_MERGED:CO_GDN_SIX,.alpha=arm==6?CO_GDN_NATIVE:CO_GDN_DIAGONAL};
        TRY(co_gdn_create_ex(fd,n,n,CO_CHAIN,&cfg,&gpu[arm]));
    }
    /* warm all contexts/code paths before measurement, including native alpha. */
    fill(initial,ns,77,.1f);fill(k,STEPS*n,79,.05f);fill(q,STEPS*n,83,.15f);fill(v,STEPS*n,89,.3f);
    for(unsigned arm=0;arm<ARMS;arm++){
        if(arm<2){TRY(co_gdn_cpu_reset(cpu[arm],initial));TRY(co_gdn_cpu_step(cpu[arm],k,q,v,.95,.6,arm? (void *)fo:(void *)ho));}
        else {TRY(co_gdn_reset(gpu[arm],initial));TRY(co_gdn_step(gpu[arm],k,q,v,.95,.6,ho));}
    }
    for(unsigned instrumented=0;instrumented<2;instrumented++)for(unsigned block=0;block<BLOCKS;block++){
        fill(initial,ns,301+block,.1f);fill(k,STEPS*n,307+block,.7f/sqrtf(n));fill(q,STEPS*n,311+block,.15f);fill(v,STEPS*n,313+block,.3f);
        for(unsigned t=0;t<STEPS;t++){gates[2*t]=(_Float16)(block==0&&t==0?0:block==0&&t==1?1:.97f);gates[2*t+1]=(_Float16)(block==0&&t==0?0:block==0&&t==1?1:.6f);}
        if(!instrumented){
#define SAVE_INPUT(label,data,bytes) do{snprintf(name,sizeof(name),"n%u-b%u-" label ".bin",n,block);TRY(save(root,name,data,bytes));}while(0)
            SAVE_INPUT("initial",initial,ns*2);SAVE_INPUT("k",k,STEPS*n*2);SAVE_INPUT("q",q,STEPS*n*2);SAVE_INPUT("v",v,STEPS*n*2);SAVE_INPUT("gates",gates,sizeof(gates));
#undef SAVE_INPUT
        }
        for(unsigned arm=0;arm<ARMS;arm++){if(arm<2)TRY(co_gdn_cpu_reset(cpu[arm],initial));else TRY(co_gdn_reset(gpu[arm],initial));}
        for(unsigned t=0;t<STEPS;t++)for(unsigned order=0;order<ARMS;order++){
            unsigned arm=(order+block+t)%ARMS;unsigned before=cpu_freq(),npu_before=npu_freq();co_gdn_profile pr;rocket_submit_counters_reset();double start=now();
            if(arm<2)TRY(co_gdn_cpu_step(cpu[arm],k+t*n,q+t*n,v+t*n,gates[2*t],gates[2*t+1],arm?(void *)fo:(void *)ho));
            else if(instrumented)TRY(co_gdn_step_profiled(gpu[arm],k+t*n,q+t*n,v+t*n,gates[2*t],gates[2*t+1],ho,&pr));
            else TRY(co_gdn_step(gpu[arm],k+t*n,q+t*n,v+t*n,gates[2*t],gates[2*t+1],ho));
            double elapsed=now()-start;unsigned after=cpu_freq(),npu_after=npu_freq();
            for(unsigned r=0;r<n;r++)outputs[((size_t)arm*STEPS+t)*n+r]=arm==1?fo[r]:(float)ho[r];
            unsigned tasks=arm<2?0:arm==4?5:arm==5?7:arm==6&&n==256?7:6;
            REQUIRE(rocket_submit_task_count()==tasks&&rocket_submit_ioctl_count()==(arm<2?0u:1u));
            printf("{\"kind\":\"step\",\"n\":%u,\"block\":%u,\"token\":%u,\"arm\":\"%s\",\"instrumented\":%u,\"total_us\":%.3f,\"freq_before_khz\":%u,\"freq_after_khz\":%u,\"npu_freq_before_khz\":%u,\"npu_freq_after_khz\":%u,\"tasks\":%u,\"submits\":%u",n,block,t,names[arm],instrumented,elapsed,before,after,npu_before,npu_after,tasks,arm<2?0:1);
            if(instrumented&&arm>=2)profile_json(&pr);
            puts("}");
        }
        for(unsigned arm=0;arm<ARMS;arm++){
            if(arm==1)TRY(co_gdn_cpu_export(cpu[1],states+arm*ns));
            else {if(!arm)TRY(co_gdn_cpu_export(cpu[0],hs));else TRY(co_gdn_export(gpu[arm],hs));for(size_t i=0;i<ns;i++)states[arm*ns+i]=hs[i];}
        }
        snprintf(name,sizeof(name),"n%u-b%u-p%u-out.bin",n,block,instrumented);TRY(save(root,name,outputs,(size_t)ARMS*STEPS*n*4));
        snprintf(name,sizeof(name),"n%u-b%u-p%u-state.bin",n,block,instrumented);TRY(save(root,name,states,(size_t)ARMS*ns*4));
    }
    /* Per-task separate-submit envelopes are a diagnostic only, not attributed
     * hardware times within a chain. Different scheduling/sync overhead. */
    {co_gdn_config cfg={1,CO_GDN_SIX,CO_GDN_DIAGONAL};co_gdn *diag=NULL;TRY(co_gdn_create_ex(fd,n,n,CO_SEPARATE,&cfg,&diag));
        rc=co_gdn_reset(diag,initial);
        for(unsigned t=0;t<16&&!rc;t++){co_gdn_profile pr;rc=co_gdn_step_profiled(diag,k+t*n,q+t*n,v+t*n,.97,.6,ho,&pr);
            if(!rc){printf("{\"kind\":\"task_diagnostic\",\"n\":%u,\"rep\":%u,\"submit_wait_us\":[",n,t);for(unsigned i=0;i<pr.tasks;i++)printf("%s%.3f",i?",":"",pr.task_submit_wait_us[i]);puts("]}");}}
        co_gdn_destroy(diag);if(rc)goto done;
    }
done:for(unsigned a=2;a<ARMS;a++)co_gdn_destroy(gpu[a]);for(unsigned a=0;a<2;a++)co_gdn_cpu_destroy(cpu[a]);
    free(initial);free(k);free(q);free(v);free(hs);free(outputs);free(states);return rc;
}
int main(int argc,char **argv){
    if(argc!=3){fprintf(stderr,"usage: %s native-smoke|run NEW_DIR\n",argv[0]);return 2;}
    int smoke=!strcmp(argv[1],"native-smoke");if(!smoke&&strcmp(argv[1],"run"))return 2;
    if(mkdir(argv[2],0755))return 2;
    int fd=rocket_open();if(fd<0)return 77;int rc=0;
    printf("{\"kind\":\"backend\",\"cpu\":\"%s\"}\n",co_gdn_cpu_backend());
    const unsigned sizes[]={128,64,256};
    for(unsigned i=0;i<3&&!rc;i++){
        rc=scalars(fd,sizes[i],0,smoke);if(!rc)rc=scalars(fd,sizes[i],1,smoke);
        if(!rc&&!smoke)rc=recurrence(fd,argv[2],sizes[i]);
    }
    rocket_close(fd);if(!rc)puts("{\"kind\":\"result\",\"status\":\"PASS\"}");return rc?1:0;
}
