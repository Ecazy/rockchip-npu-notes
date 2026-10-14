// SPDX-License-Identifier: GPL-3.0-or-later
#include "co_decoder.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
static unsigned long long uploads,downloads;static int fail_after_post;
int __real_co_upload_f16(co_view *,const _Float16 *);
int __wrap_co_upload_f16(co_view *v,const _Float16 *x){uploads++;return __real_co_upload_f16(v,x);}
int __real_co_download(co_view *,void *);
int __wrap_co_download(co_view *v,void *x){downloads++;return __real_co_download(v,x);}
int __real_co_graph_run(co_graph *,const co_graph_input *,unsigned,_Float16 *);
int __wrap_co_graph_run(co_graph *g,const co_graph_input *x,unsigned n,_Float16 *y){
    int rc=__real_co_graph_run(g,x,n,y);
    if(!rc&&fail_after_post){fail_after_post=0;return ROCKET_E_DEVICE;}return rc;
}
#define TRY(e) do{rc=(e);if(rc){fprintf(stderr,"line %d status=%d\n",__LINE__,rc);goto done;}}while(0)
#define REQUIRE(e) do{if(!(e)){fprintf(stderr,"line %d failed: %s\n",__LINE__,#e);rc=ROCKET_E_DEVICE;goto done;}}while(0)
struct witness {unsigned n,cap,mq,dm,ff,hq,hv,d,rd,window,base;enum co_rope_style style;unsigned steps[5],nsteps;};
static const struct witness cases[]={
    {7,7,3,17,33,4,2,6,4,3,17,CO_ROPE_SPLIT_HALF,{3,1,3},3},
    {131,259,129,33,65,4,2,8,8,0,0,CO_ROPE_SPLIT_HALF,{129,1,1},3},
    {17,17,13,304,513,16,4,19,18,5,2048,CO_ROPE_INTERLEAVED,{13,1,1,1,1},5},
};
static const char *names[]={"norm1.bin","norm2.bin","wq.bin","wk.bin","wv.bin","wo.bin","wgate.bin","wup.bin","wdown.bin"};
static void fill(_Float16 *x,size_t n,uint32_t seed,double scale){
    for(size_t i=0;i<n;i++){seed=1664525u*seed+1013904223u;x[i]=(_Float16)((((seed>>8)&65535)/65535.0*2-1)*scale);}
}
static int save(const char *root,unsigned id,const char *name,const void *data,size_t bytes){
    char path[1024];if(snprintf(path,sizeof(path),"%s/case-%u-%s",root,id,name)>=(int)sizeof(path))return -1;
    FILE *f=fopen(path,"wb");if(!f)return -1;
    int ok=fwrite(data,1,bytes,f)==bytes;if(fclose(f))ok=0;return ok?0:-1;
}
static uint64_t now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000ull+t.tv_nsec;}
static int run_case(int fd,const char *root,unsigned id,enum co_mode mode){
    struct witness w=cases[id];int rc=0;co_decoder *c[2]={0};co_graph *full=NULL;FILE *report=NULL;
    unsigned qw=w.hq*w.d,kw=w.hv*w.d;size_t xs=(size_t)w.n*w.dm,ks=(size_t)w.n*kw;
    _Float16 *weights[9]={0};size_t sizes[]={w.dm,w.dm,(size_t)qw*w.dm,(size_t)kw*w.dm,(size_t)kw*w.dm,
        (size_t)w.dm*qw,(size_t)w.ff*w.dm,(size_t)w.ff*w.dm,(size_t)w.dm*w.ff};
    _Float16 *x=malloc(xs*2),*mask=malloc((size_t)w.n*w.n*2),*smask=malloc((size_t)w.mq*w.n*2),
        *y[2]={malloc(xs*2),malloc(xs*2)},*first[2]={malloc(xs*2),malloc(xs*2)},*fy=malloc(xs*2),*f0=malloc(xs*2),
        *k=malloc(ks*2),*v=malloc(ks*2),*prevk=malloc(ks*2),*prevv=malloc(ks*2);
    REQUIRE(x&&mask&&smask&&y[0]&&y[1]&&first[0]&&first[1]&&fy&&f0&&k&&v&&prevk&&prevv);
    fill(x,xs,991+id,0.7);
    for(unsigned t=0;t<w.n;t++)for(unsigned j=0;j<w.n;j++)mask[(size_t)t*w.n+j]=t==2?-INFINITY:(j%7?0:-0.125f);
    REQUIRE(!save(root,id,"x.bin",x,xs*2)&&!save(root,id,"mask.bin",mask,(size_t)w.n*w.n*2));
    double scale[]={0.2,0.2,0.7/sqrt(w.dm),0.7/sqrt(w.dm),0.8/sqrt(w.dm),0.6/sqrt(qw),1.0/sqrt(w.dm),1.0/sqrt(w.dm),0.8/sqrt(w.ff)};
    for(unsigned i=0;i<9;i++){
        weights[i]=malloc(sizes[i]*2);REQUIRE(weights[i]);fill(weights[i],sizes[i],1231+37*i+id,scale[i]);
        if(i<2)for(size_t j=0;j<sizes[i];j++)weights[i][j]=(_Float16)(1+weights[i][j]);
        REQUIRE(!save(root,id,names[i],weights[i],sizes[i]*2));
    }
    co_decoder_weights cw={weights[0],weights[1],weights[2],weights[3],weights[4],weights[5],weights[6],weights[7],weights[8]};
    co_decoder_config cfg={.model_dim=w.dm,.hidden_dim=w.ff,.epsilon=1e-5f,
        .attention={.max_query=w.mq,.capacity=w.cap,.heads=w.hq,.kv_heads=w.hv,.head_dim=w.d,.rotary_dim=w.rd,
         .theta=10000,.style=w.style,.causal=1,.window=w.window}};
    co_decoder_plan plan[2];
    for(unsigned path=0;path<2;path++){
        cfg.disable_one_row_decode=path==0;TRY(co_decoder_plan_init(&cfg,&plan[path]));TRY(co_decoder_create(fd,&cfg,&cw,mode,&c[path]));
    }
    co_graph_node nodes[]={
        {.op=CO_GRAPH_INPUT,.rows=w.n,.cols=w.dm},
        {.op=CO_GRAPH_RMSNORM,.a=0,.gamma=cw.norm1,.epsilon=1e-5f},
        {.op=CO_GRAPH_LINEAR,.a=1,.cols=qw,.weights=cw.wq},{.op=CO_GRAPH_LINEAR,.a=1,.cols=kw,.weights=cw.wk},
        {.op=CO_GRAPH_LINEAR,.a=1,.cols=kw,.weights=cw.wv},
        {.op=CO_GRAPH_INPUT,.rows=w.n,.cols=w.n,.flags=CO_GRAPH_ALLOW_NEG_INF},
        {.op=CO_GRAPH_ATTENTION,.a=2,.b=3,.c=4,.mask_node=5,.has_mask=1,.heads=w.hq,.kv_heads=w.hv,.head_dim=w.d,
         .rotary_dim=w.rd,.theta=10000,.rope_style=w.style,.position_base=w.base,.causal=1,.window=w.window},
        {.op=CO_GRAPH_LINEAR,.a=6,.cols=w.dm,.weights=cw.wo},{.op=CO_GRAPH_ADD,.a=7,.b=0},
        {.op=CO_GRAPH_RMSNORM,.a=8,.gamma=cw.norm2,.epsilon=1e-5f},
        {.op=CO_GRAPH_LINEAR,.a=9,.cols=w.ff,.weights=cw.wgate},{.op=CO_GRAPH_LINEAR,.a=9,.cols=w.ff,.weights=cw.wup},
        {.op=CO_GRAPH_SILU,.a=10},{.op=CO_GRAPH_MUL,.a=12,.b=11},
        {.op=CO_GRAPH_LINEAR,.a=13,.cols=w.dm,.weights=cw.wdown},{.op=CO_GRAPH_ADD,.a=14,.b=8}};
    co_graph_desc desc={nodes,16,15,0};co_graph_plan gp;TRY(co_graph_plan_init(&desc,&gp));TRY(co_graph_compile(fd,&desc,mode,&full));
    /* All three compiled paths must own their matrix/gamma copies. */
    for(unsigned i=0;i<9;i++){memset(weights[i],0,sizes[i]*2);free(weights[i]);weights[i]=NULL;}
    char file[1024];REQUIRE(snprintf(file,sizeof(file),"%s/case-%u.json",root,id)<(int)sizeof(file));report=fopen(file,"w");REQUIRE(report);
    fprintf(report,"{\"case\":%u,\"tokens\":%u,\"capacity\":%u,\"max_query\":%u,\"model_dim\":%u,\"hidden_dim\":%u,\"heads\":%u,\"kv_heads\":%u,\"head_dim\":%u,\"rotary_dim\":%u,\"style\":%u,\"window\":%u,\"position_base\":%u,\"epsilon\":1e-5,\"theta\":10000,\"baseline_device_bytes\":%zu,\"optimized_device_bytes\":%zu,\"steps\":[",
        id,w.n,w.cap,w.mq,w.dm,w.ff,w.hq,w.hv,w.d,w.rd,w.style,w.window,w.base,plan[0].device_bytes,plan[1].device_bytes);
    for(unsigned i=0;i<w.nsteps;i++)fprintf(report,"%s%u",i?",":"",w.steps[i]);
    fprintf(report,"],\"runs\":[");unsigned runs=0;
    int rollback=0;
    for(unsigned rep=0;rep<3;rep++){
        for(unsigned p=0;p<2;p++)TRY(co_decoder_reset(c[p],w.base));
        unsigned old=0;
        for(unsigned step=0;step<w.nsteps;step++){
            unsigned tok=w.steps[step],length=old+tok;
            for(unsigned t=0;t<tok;t++)for(unsigned j=0;j<length;j++)smask[(size_t)t*length+j]=mask[(size_t)(old+t)*w.n+j];
            for(unsigned order=0;order<2;order++){
                unsigned path=(order+rep)%2;
                if(rep==0&&step==1&&path==1){
                    TRY(co_decoder_cache_export(c[path],prevk,prevv,w.n));size_t bytes=(size_t)old*kw*2;
                    REQUIRE(!save(root,id,"before-fail-k.bin",prevk,bytes)&&!save(root,id,"before-fail-v.bin",prevv,bytes));
                    fail_after_post=1;memset(y[path]+(size_t)old*w.dm,0x23,(size_t)tok*w.dm*2);
                    REQUIRE(co_decoder_step(c[path],tok,x+(size_t)old*w.dm,smask,y[path]+(size_t)old*w.dm)==ROCKET_E_DEVICE);
                    REQUIRE(!fail_after_post&&co_decoder_cache_length(c[path])==old);TRY(co_decoder_cache_export(c[path],k,v,w.n));
                    REQUIRE(!memcmp(k,prevk,bytes)&&!memcmp(v,prevv,bytes));
                    REQUIRE(!save(root,id,"after-fail-k.bin",k,bytes)&&!save(root,id,"after-fail-v.bin",v,bytes));
                    for(size_t i=0;i<(size_t)tok*w.dm*2;i++)REQUIRE(((unsigned char *)(y[path]+(size_t)old*w.dm))[i]==0x23);
                    REQUIRE(!save(root,id,"fail-y.bin",y[path]+(size_t)old*w.dm,(size_t)tok*w.dm*2));rollback=1;
                }
                unsigned geo=path==1&&tok==1?1:0;uploads=downloads=0;rocket_submit_counters_reset();uint64_t start=now();
                TRY(co_decoder_step(c[path],tok,x+(size_t)old*w.dm,smask,y[path]+(size_t)old*w.dm));uint64_t elapsed=now()-start;
                unsigned long long submits=rocket_submit_ioctl_count(),tasks=rocket_submit_task_count();
                REQUIRE(submits==(mode==CO_CHAIN?plan[path].chain_submits[geo]:plan[path].tasks[geo])&&tasks==plan[path].tasks[geo]&&
                    uploads==plan[path].upload_calls[geo]&&downloads==plan[path].download_calls[geo]&&co_decoder_cache_length(c[path])==length&&
                    co_decoder_last_query_rows(c[path])==plan[path].rows[geo]);
                fprintf(report,"%s{\"repeat\":%u,\"step\":%u,\"path\":\"%s\",\"old\":%u,\"tokens\":%u,\"length\":%u,\"query_rows\":%u,\"score_elements\":%zu,\"submits\":%llu,\"tasks\":%llu,\"uploads\":%llu,\"downloads\":%llu,\"api_ns\":%llu,\"cache_switches\":%u}",
                    runs++?",":"",rep,step,path?"optimized":"baseline",old,tok,length,plan[path].rows[geo],plan[path].score_elements[geo],
                    submits,tasks,uploads,downloads,(unsigned long long)elapsed,co_decoder_cache_switches(c[path]));
            }
            old=length;
        }
        for(unsigned path=0;path<2;path++){
            if(!rep)memcpy(first[path],y[path],xs*2);else REQUIRE(!memcmp(first[path],y[path],xs*2));
        }
    }
    for(unsigned path=0;path<2;path++){
        TRY(co_decoder_cache_export(c[path],k,v,w.n));
        REQUIRE(!save(root,id,path?"optimized-y.bin":"baseline-y.bin",y[path],xs*2)&&
            !save(root,id,path?"optimized-k.bin":"baseline-k.bin",k,ks*2)&&!save(root,id,path?"optimized-v.bin":"baseline-v.bin",v,ks*2));
    }
    fprintf(report,"],\"full_runs\":[");co_graph_input bindings[]={{0,x},{5,mask}};
    for(unsigned rep=0;rep<3;rep++){
        uploads=downloads=0;rocket_submit_counters_reset();TRY(co_graph_run(full,bindings,2,fy));
        unsigned long long submits=rocket_submit_ioctl_count(),tasks=rocket_submit_task_count();
        REQUIRE(submits==(mode==CO_CHAIN?gp.chain_submits:gp.tasks)&&tasks==gp.tasks&&
            uploads==gp.tensors[0].tiles+gp.tensors[5].tiles+gp.host_upload_calls&&
            downloads==gp.tensors[15].tiles+gp.statistic_download_calls+gp.activation_download_calls);
        if(!rep)memcpy(f0,fy,xs*2);else REQUIRE(!memcmp(f0,fy,xs*2));
        fprintf(report,"%s{\"repeat\":%u,\"submits\":%llu,\"tasks\":%llu,\"uploads\":%llu,\"downloads\":%llu}",rep?",":"",rep,submits,tasks,uploads,downloads);
    }
    REQUIRE(!save(root,id,"full-y.bin",fy,xs*2));fprintf(report,"],\"post_ffn_rollback\":%s,\"status\":\"PASS\"}\n",rollback?"true":"false");
    printf("DECODER case=%u tokens=%u model=%u hidden=%u max_query=%u mode=%s post_ffn_rollback=PASS PASS\n",id,w.n,w.dm,w.ff,w.mq,mode==CO_CHAIN?"chain":"separate");
done:
    if(report)fclose(report);
    co_graph_destroy(full);for(unsigned i=0;i<2;i++)co_decoder_destroy(c[i]);for(unsigned i=0;i<9;i++)free(weights[i]);
    free(x);free(mask);free(smask);for(unsigned i=0;i<2;i++){free(y[i]);free(first[i]);}free(fy);free(f0);free(k);free(v);free(prevk);free(prevv);return rc;
}
int main(int argc,char **argv){
    if(argc!=3){fprintf(stderr,"usage: %s RESULTS chain|separate\n",argv[0]);return 1;}
    if(strcmp(argv[2],"chain")&&strcmp(argv[2],"separate"))return 1;
    enum co_mode mode=!strcmp(argv[2],"chain")?CO_CHAIN:CO_SEPARATE;
    if(mkdir(argv[1],0755)){perror(argv[1]);return 1;}
    int fd=rocket_open();if(fd<0)return 77;int rc=0;
    for(unsigned i=0;i<sizeof(cases)/sizeof(cases[0])&&!rc;i++)rc=run_case(fd,argv[1],i,mode);
    rocket_close(fd);return rc?1:0;
}
