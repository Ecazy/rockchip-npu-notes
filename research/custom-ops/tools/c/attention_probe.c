// SPDX-License-Identifier: GPL-3.0-or-later
#include "co_graph.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static unsigned long long uploads,downloads,execs,fail_exec;
int __real_co_upload_f16(co_view *,const _Float16 *);
int __wrap_co_upload_f16(co_view *v,const _Float16 *x){uploads++;return __real_co_upload_f16(v,x);}
int __real_co_download(co_view *,void *);
int __wrap_co_download(co_view *v,void *x){downloads++;return __real_co_download(v,x);}
int __real_co_execute(int,const co_program *,unsigned,enum co_mode);
int __wrap_co_execute(int fd,const co_program *p,unsigned n,enum co_mode mode){
    if(++execs==fail_exec)return ROCKET_E_DEVICE;
    return __real_co_execute(fd,p,n,mode);
}
#define TRY(e) do{rc=(e);if(rc){fprintf(stderr,"line %d status=%d\n",__LINE__,rc);goto done;}}while(0)
#define REQUIRE(e) do{if(!(e)){fprintf(stderr,"line %d failed: %s\n",__LINE__,#e);rc=ROCKET_E_DEVICE;goto done;}}while(0)
struct witness {unsigned n,cap,mq,hq,hv,d,rd,window,base;enum co_rope_style style;int causal;unsigned steps[4],nsteps;};
static const struct witness cases[]={
    {7,7,3,4,2,6,4,3,17,CO_ROPE_INTERLEAVED,1,{3,3,1},3},
    {259,259,129,2,1,8,8,0,0,CO_ROPE_SPLIT_HALF,1,{129,129,1},3},
    {257,257,257,1,1,19,18,17,1000,CO_ROPE_INTERLEAVED,1,{257},1},
    {17,17,17,2,2,5,0,0,0,CO_ROPE_INTERLEAVED,0,{17},1},
    {8,16,8,4,1,32,16,4,4096,CO_ROPE_SPLIT_HALF,1,{8},1},
    {13,13,13,32,16,19,18,5,2048,CO_ROPE_INTERLEAVED,1,{13},1},
};
static void fill(_Float16 *x,size_t n,uint32_t seed,double scale){
    for(size_t i=0;i<n;i++){seed=1664525u*seed+1013904223u;x[i]=(_Float16)((((seed>>8)&65535)/65535.0*2-1)*scale);}
}
static int save(const char *root,unsigned id,const char *name,const void *data,size_t bytes){
    char path[1024];if(snprintf(path,sizeof(path),"%s/case-%u-%s",root,id,name)>=(int)sizeof(path))return -1;
    FILE *f=fopen(path,"wb");if(!f)return -1;
    int ok=fwrite(data,1,bytes,f)==bytes;if(fclose(f))ok=0;return ok?0:-1;
}
static int run_case(int fd,const char *root,unsigned id,enum co_mode mode){
    const struct witness w=cases[id];int rc=0;co_attention *c=NULL;co_graph *g=NULL;
    size_t qs=(size_t)w.n*w.hq*w.d,ks=(size_t)w.n*w.hv*w.d;
    _Float16 *q=malloc(qs*2),*k=malloc(ks*2),*v=malloc(ks*2),*mask=malloc((size_t)w.n*w.n*2),
        *stepmask=malloc((size_t)w.mq*w.n*2),*y=malloc(qs*2),*first=malloc(qs*2),*gy=malloc(qs*2),
        *ck=malloc(ks*2),*cv=malloc(ks*2),*prevk=malloc(ks*2),*prevv=malloc(ks*2);
    FILE *report=NULL;char path[1024];
    REQUIRE(q&&k&&v&&mask&&stepmask&&y&&first&&gy&&ck&&cv&&prevk&&prevv);
    fill(q,qs,123+id,0.7);fill(k,ks,551+id,0.6);fill(v,ks,771+id,0.8);
    /* Make the final key dominant: dropping a ragged K/PV tail must cause a
     * large numerical mismatch, rather than hide beneath a 1/N probability. */
    for(unsigned h=0;h<w.hq;h++)for(unsigned i=0;i<w.d;i++)q[((size_t)(w.n-1)*w.hq+h)*w.d+i]=1;
    for(unsigned h=0;h<w.hv;h++)for(unsigned i=0;i<w.d;i++)k[((size_t)(w.n-1)*w.hv+h)*w.d+i]=1;
    for(unsigned t=0;t<w.n;t++)for(unsigned j=0;j<w.n;j++)
        mask[(size_t)t*w.n+j]=t==2?-INFINITY:(t==w.n-1&&j==w.n-1?4:(j%7?0:-0.25f));
    REQUIRE(!save(root,id,"q.bin",q,qs*2)&&!save(root,id,"k.bin",k,ks*2)&&!save(root,id,"v.bin",v,ks*2)&&
        !save(root,id,"mask.bin",mask,(size_t)w.n*w.n*2));
    co_attention_config cfg={.max_query=w.mq,.capacity=w.cap,.heads=w.hq,.kv_heads=w.hv,.head_dim=w.d,
        .rotary_dim=w.rd,.theta=10000,.style=w.style,.causal=w.causal,.window=w.window};
    co_attention_plan plan;TRY(co_attention_plan_init(&cfg,&plan));TRY(co_attention_create(fd,&cfg,mode,&c));
    REQUIRE(snprintf(path,sizeof(path),"%s/case-%u.json",root,id)<(int)sizeof(path));report=fopen(path,"w");REQUIRE(report);
    fprintf(report,"{\"case\":%u,\"tokens\":%u,\"capacity\":%u,\"max_query\":%u,\"heads\":%u,\"kv_heads\":%u,\"head_dim\":%u,\"rotary_dim\":%u,\"style\":%u,\"causal\":%d,\"window\":%u,\"position_base\":%u,\"theta\":10000,\"device_bytes\":%zu,\"steps\":[",
        id,w.n,w.cap,w.mq,w.hq,w.hv,w.d,w.rd,w.style,w.causal,w.window,w.base,plan.device_bytes);
    for(unsigned i=0;i<w.nsteps;i++)fprintf(report,"%s%u",i?",":"",w.steps[i]);
    fprintf(report,"],\"runs\":[");unsigned run=0;int rollback=0;
    for(unsigned rep=0;rep<3;rep++){
        TRY(co_attention_reset(c,w.base));unsigned old=0;
        for(unsigned step=0;step<w.nsteps;step++){
            unsigned tok=w.steps[step],len=old+tok;
            for(unsigned t=0;t<tok;t++)for(unsigned j=0;j<len;j++)stepmask[(size_t)t*len+j]=mask[(size_t)(old+t)*w.n+j];
            if(rep==0&&step==(w.nsteps>1?1u:0u)){
                TRY(co_attention_cache_export(c,prevk,prevv,w.n));
                size_t bytes=(size_t)old*w.hv*w.d*2;
                REQUIRE(!save(root,id,"before-fail-k.bin",prevk,bytes)&&!save(root,id,"before-fail-v.bin",prevv,bytes));
                unsigned mt=(w.mq+255)/256,nt=(w.cap+255)/256;
                fail_exec=execs+(w.rd?w.hv*mt+mt:0)+3*mt*nt+1;
                memset(y+(size_t)old*w.hq*w.d,0x23,(size_t)tok*w.hq*w.d*2);
                REQUIRE(co_attention_step(c,tok,q+(size_t)old*w.hq*w.d,k+(size_t)old*w.hv*w.d,v+(size_t)old*w.hv*w.d,
                                           stepmask,y+(size_t)old*w.hq*w.d)==ROCKET_E_DEVICE);
                fail_exec=0;REQUIRE(co_attention_cache_length(c)==old);TRY(co_attention_cache_export(c,ck,cv,w.n));
                REQUIRE(!memcmp(ck,prevk,bytes)&&!memcmp(cv,prevv,bytes));
                REQUIRE(!save(root,id,"after-fail-k.bin",ck,bytes)&&!save(root,id,"after-fail-v.bin",cv,bytes));
                for(size_t i=0;i<(size_t)tok*w.hq*w.d*2;i++)REQUIRE(((unsigned char *)(y+(size_t)old*w.hq*w.d))[i]==0x23);
                REQUIRE(!save(root,id,"fail-y.bin",y+(size_t)old*w.hq*w.d,(size_t)tok*w.hq*w.d*2));rollback=1;
            }
            uploads=downloads=0;rocket_submit_counters_reset();
            TRY(co_attention_step(c,tok,q+(size_t)old*w.hq*w.d,k+(size_t)old*w.hv*w.d,v+(size_t)old*w.hv*w.d,stepmask,y+(size_t)old*w.hq*w.d));
            unsigned long long submits=rocket_submit_ioctl_count(),tasks=rocket_submit_task_count();
            REQUIRE(submits==(mode==CO_CHAIN?plan.chain_submits_per_step:plan.tasks_per_step)&&tasks==plan.tasks_per_step&&
                    uploads==plan.upload_calls&&downloads==plan.download_calls&&co_attention_cache_length(c)==len);
            fprintf(report,"%s{\"repeat\":%u,\"old\":%u,\"tokens\":%u,\"length\":%u,\"submits\":%llu,\"tasks\":%llu,\"uploads\":%llu,\"downloads\":%llu}",
                run++?",":"",rep,old,tok,len,submits,tasks,uploads,downloads);old=len;
        }
        if(!rep)memcpy(first,y,qs*2);else REQUIRE(!memcmp(first,y,qs*2));
    }
    TRY(co_attention_cache_export(c,ck,cv,w.n));REQUIRE(!memcmp(cv,v,ks*2));
    REQUIRE(!save(root,id,"stream-y.bin",y,qs*2)&&!save(root,id,"cache-k.bin",ck,ks*2)&&!save(root,id,"cache-v.bin",cv,ks*2));
    unsigned committed=co_attention_cache_length(c);REQUIRE(co_attention_reset(c,UINT32_MAX)==ROCKET_E_SHAPE&&co_attention_cache_length(c)==committed);
    TRY(co_attention_reset(c,w.base));_Float16 oldq=q[0];q[0]=NAN;unsigned long long calls=execs;
    REQUIRE(co_attention_step(c,1,q,k,v,NULL,y)==ROCKET_E_SHAPE&&execs==calls&&!co_attention_cache_length(c));q[0]=oldq;
    co_graph_node nodes[]={{.op=CO_GRAPH_INPUT,.rows=w.n,.cols=w.hq*w.d},
        {.op=CO_GRAPH_INPUT,.rows=w.n,.cols=w.hv*w.d},{.op=CO_GRAPH_INPUT,.rows=w.n,.cols=w.hv*w.d},
        {.op=CO_GRAPH_INPUT,.rows=w.n,.cols=w.n,.flags=CO_GRAPH_ALLOW_NEG_INF},
        {.op=CO_GRAPH_ATTENTION,.a=0,.b=1,.c=2,.mask_node=3,.has_mask=1,.heads=w.hq,.kv_heads=w.hv,.head_dim=w.d,
         .rotary_dim=w.rd,.theta=10000,.rope_style=w.style,.position_base=w.base,.causal=w.causal,.window=w.window}};
    co_graph_desc desc={nodes,5,4,0};co_graph_plan gp;TRY(co_graph_plan_init(&desc,&gp));TRY(co_graph_compile(fd,&desc,mode,&g));
    co_graph_input inputs[]={{0,q},{1,k},{2,v},{3,mask}};
    fprintf(report,"],\"graph_runs\":[");
    for(unsigned rep=0;rep<3;rep++){
        uploads=downloads=0;rocket_submit_counters_reset();TRY(co_graph_run(g,inputs,4,gy));
        unsigned long long submits=rocket_submit_ioctl_count(),tasks=rocket_submit_task_count(),in_tiles=0;
        for(unsigned i=0;i<4;i++)in_tiles+=gp.tensors[i].tiles;
        REQUIRE(submits==(mode==CO_CHAIN?gp.chain_submits:gp.tasks)&&tasks==gp.tasks&&
            uploads==in_tiles+gp.host_upload_calls&&downloads==gp.tensors[4].tiles+gp.statistic_download_calls+gp.activation_download_calls);
        if(!rep)memcpy(y,gy,qs*2);else REQUIRE(!memcmp(y,gy,qs*2));
        fprintf(report,"%s{\"repeat\":%u,\"submits\":%llu,\"tasks\":%llu,\"uploads\":%llu,\"downloads\":%llu}",
            rep?",":"",rep,submits,tasks,uploads,downloads);
    }
    REQUIRE(!save(root,id,"graph-y.bin",gy,qs*2));fprintf(report,"],\"rollback\":%s,\"status\":\"PASS\"}\n",rollback?"true":"false");
    printf("ATTENTION case=%u tokens=%u cap=%u max_query=%u heads=%u kv_heads=%u d=%u rope=%u style=%u mode=%s rollback=PASS PASS\n",
        id,w.n,w.cap,w.mq,w.hq,w.hv,w.d,w.rd,w.style,mode==CO_CHAIN?"chain":"separate");
done:
    if(report)fclose(report);
    co_graph_destroy(g);co_attention_destroy(c);
    free(q);free(k);free(v);free(mask);free(stepmask);free(y);free(first);free(gy);free(ck);free(cv);free(prevk);free(prevv);return rc;
}
static int rope_witness(int fd,const char *root,enum co_mode mode,unsigned id){
    unsigned m=id==99?13:257,h=id==99?4:16,d=id==99?100:19,rd=id==99?96:18;size_t count=(size_t)m*h*d;int rc=0;co_graph *g=NULL;
    _Float16 *x=malloc(count*2),*y=malloc(count*2);REQUIRE(x&&y);fill(x,count,719,0.7);
    co_graph_node nodes[]={{.op=CO_GRAPH_INPUT,.rows=m,.cols=h*d},
        {.op=CO_GRAPH_ROPE,.a=0,.heads=h,.head_dim=d,.rotary_dim=rd,.theta=10000,.rope_style=id==99?CO_ROPE_SPLIT_HALF:CO_ROPE_INTERLEAVED,.position_base=id==99?17:1000}};
    co_graph_desc desc={nodes,2,1,0};co_graph_plan p;TRY(co_graph_plan_init(&desc,&p));TRY(co_graph_compile(fd,&desc,mode,&g));
    co_graph_input input={0,x};uploads=downloads=0;rocket_submit_counters_reset();TRY(co_graph_run(g,&input,1,y));
    REQUIRE(rocket_submit_ioctl_count()==(mode==CO_CHAIN?p.chain_submits:p.tasks)&&rocket_submit_task_count()==p.tasks&&
        uploads==p.tensors[0].tiles+p.host_upload_calls&&downloads==p.tensors[1].tiles);
    REQUIRE(!save(root,id,"rope-x.bin",x,count*2)&&!save(root,id,"rope-y.bin",y,count*2));
    printf("ROPE case=%u m=%u heads=%u d=%u PASS\n",id,m,h,d);
done:co_graph_destroy(g);free(x);free(y);return rc;
}
int main(int argc,char **argv){
    if(argc!=3){fprintf(stderr,"usage: %s RESULTS chain|separate\n",argv[0]);return 1;}
    enum co_mode mode=!strcmp(argv[2],"chain")?CO_CHAIN:CO_SEPARATE;
    if(strcmp(argv[2],"chain")&&strcmp(argv[2],"separate"))return 1;
    if(mkdir(argv[1],0755)){perror(argv[1]);return 1;}
    int fd=rocket_open();if(fd<0)return 77;int rc=rope_witness(fd,argv[1],mode,99);
    if(!rc)rc=rope_witness(fd,argv[1],mode,98);
    for(unsigned i=0;i<sizeof(cases)/sizeof(cases[0])&&!rc;i++)rc=run_case(fd,argv[1],i,mode);
    rocket_close(fd);return rc?1:0;
}
