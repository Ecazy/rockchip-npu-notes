// SPDX-License-Identifier: GPL-3.0-or-later
#include "co_graph.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static unsigned long long uploads,downloads;
int __real_co_upload_f16(co_view *,const _Float16 *);
int __wrap_co_upload_f16(co_view *v,const _Float16 *x) {uploads++;return __real_co_upload_f16(v,x);}
int __real_co_download(co_view *,void *);
int __wrap_co_download(co_view *v,void *y) {downloads++;return __real_co_download(v,y);}
static void usage(const char *name) {
    printf("usage: %s rms|softmax|hsig-rms|hsig-softmax M N X.bin gamma.bin|- Y.bin [chain|separate] [repeat]\n",name);
}
static int number(const char *s,unsigned max,unsigned *out) {
    if(!s||!*s||*s=='-'||*s=='+')return -1;
    errno=0;char *end=NULL;unsigned long v=strtoul(s,&end,10);
    if(errno||*end||!v||v>max)return -1;
    *out=(unsigned)v;return 0;
}
static int read_half(const char *path,_Float16 *x,size_t n,int mask) {
    FILE *f=fopen(path,"rb");if(!f){perror(path);return -1;}
    int ok=fread(x,2,n,f)==n;
    if(ok)ok=fgetc(f)==EOF&&!ferror(f);
    if(fclose(f))ok=0;
    for(size_t i=0;ok&&i<n;i++)if(!isfinite((double)x[i])&&!(mask&&x[i]==-INFINITY))ok=0;
    if(!ok)fprintf(stderr,"%s: invalid FP16 tensor (expected %zu elements)\n",path,n);
    return ok?0:-1;
}
static int write_half(const char *path,const _Float16 *y,size_t n) {
    char *temp=malloc(strlen(path)+32);if(!temp)return -1;
    sprintf(temp,"%s.tmp.XXXXXX",path);int fd=mkstemp(temp);
    if(fd<0){free(temp);return -1;}
    FILE *f=fdopen(fd,"wb");if(!f){close(fd);unlink(temp);free(temp);return -1;}
    int ok=fwrite(y,2,n,f)==n;if(fclose(f))ok=0;
    if(ok&&rename(temp,path))ok=0;
    if(!ok)unlink(temp);
    free(temp);return ok?0:-1;
}
int main(int argc,char **argv) {
    if(argc==2&&!strcmp(argv[1],"--help")){usage(argv[0]);return 0;}
    if(argc<7||argc>9){usage(argv[0]);return 1;}
    uint16_t endian=1;if(*(unsigned char *)&endian!=1||sizeof(_Float16)!=2)return 1;
    int prefix=!strncmp(argv[1],"hsig-",5);const char *kind=argv[1]+(prefix?5:0);
    enum co_graph_op op;
    if(!strcmp(kind,"rms"))op=CO_GRAPH_RMSNORM;
    else if(!strcmp(kind,"softmax"))op=CO_GRAPH_SOFTMAX;
    else return 1;
    if((op==CO_GRAPH_RMSNORM&&!strcmp(argv[5],"-"))||
       (op==CO_GRAPH_SOFTMAX&&strcmp(argv[5],"-")))return 1;
    unsigned m,n,repeat=1;
    if(number(argv[2],4096,&m)||number(argv[3],4096,&n)||
       (argc==9&&number(argv[8],1000,&repeat)))return 1;
    enum co_mode mode=CO_CHAIN;
    if(argc>=8){if(!strcmp(argv[7],"separate"))mode=CO_SEPARATE;else if(strcmp(argv[7],"chain"))return 1;}
    if(!strcmp(argv[4],argv[6])||!strcmp(argv[5],argv[6]))return 1;
    int rc=1,fd=-1;co_graph *g=NULL;
    _Float16 *x=malloc((size_t)m*n*2),*y=malloc((size_t)m*n*2),*first=malloc((size_t)m*n*2),
        *gamma=op==CO_GRAPH_RMSNORM?malloc(n*2):NULL;
    if(!x||!y||!first||(op==CO_GRAPH_RMSNORM&&!gamma))goto done;
    if(read_half(argv[4],x,(size_t)m*n,op==CO_GRAPH_SOFTMAX&&!prefix)||
       (gamma&&read_half(argv[5],gamma,n,0)))goto done;
    co_graph_node nodes[]={
        {.op=CO_GRAPH_INPUT,.rows=m,.cols=n,.flags=op==CO_GRAPH_SOFTMAX&&!prefix?CO_GRAPH_ALLOW_NEG_INF:0},
        {.op=CO_GRAPH_HSIG,.a=0},
        {.op=op,.a=prefix?1:0,.gamma=gamma,.epsilon=1e-5f}};
    co_graph_desc desc={nodes,3,2,0};co_graph_plan plan;
    int status=co_graph_plan_init(&desc,&plan);if(status){fprintf(stderr,"plan: %d\n",status);goto done;}
    fd=rocket_open();if(fd<0){rc=77;goto done;}
    status=co_graph_compile(fd,&desc,mode,&g);if(status){fprintf(stderr,"compile: %d\n",status);goto done;}
    free(gamma);gamma=NULL;
    uploads=downloads=0;rocket_submit_counters_reset();co_graph_input input={0,x};
    for(unsigned rep=0;rep<repeat;rep++) {
        status=co_graph_run(g,&input,1,y);if(status){fprintf(stderr,"run: %d\n",status);goto done;}
        if(!rep)memcpy(first,y,(size_t)m*n*2);
        else if(memcmp(first,y,(size_t)m*n*2)){fprintf(stderr,"repeat mismatch\n");goto done;}
    }
    if(uploads!=((unsigned long long)plan.tensors[0].tiles+plan.host_upload_calls)*repeat||
       downloads!=((unsigned long long)plan.tensors[2].tiles+plan.statistic_download_calls)*repeat) {
        fprintf(stderr,"hybrid transfer mismatch uploads=%llu downloads=%llu\n",uploads,downloads);goto done;
    }
    if(write_half(argv[6],y,(size_t)m*n))goto done;
    printf("ROW kind=%s m=%u n=%u mode=%s slots=%u device_bytes=%zu repeat=%u submits=%llu tasks=%llu uploads=%llu downloads=%llu planned_activation_reads=%zu statistic_elements=%zu host_upload_elements=%zu PASS\n",
        argv[1],m,n,mode==CO_CHAIN?"chain":"separate",plan.slots,plan.device_bytes,repeat,
        (unsigned long long)rocket_submit_ioctl_count(),(unsigned long long)rocket_submit_task_count(),uploads,downloads,
        plan.host_activation_reads,plan.statistic_downloads,plan.host_uploads);
    rc=0;
done:
    co_graph_destroy(g);if(fd>=0)rocket_close(fd);free(x);free(y);free(first);free(gamma);return rc;
}
