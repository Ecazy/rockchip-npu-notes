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
    printf("usage: %s M D H X.bin Wgate.bin Wup.bin Wdown.bin b_gate.bin|- Y.bin [chain|separate] [repeat] [rmsnorm gamma.bin]\n",name);
    printf("FP16 row-major; Y=X+Linear(SiLU(Linear(X,Wgate,b_gate))*Linear(X,Wup),Wdown)\n");
}
static int number(const char *s,unsigned max,unsigned *out) {
    if(!s||!*s||*s=='-'||*s=='+')return -1;
    errno=0;char *end=NULL;unsigned long v=strtoul(s,&end,10);
    if(errno||*end||!v||v>max)return -1;
    *out=(unsigned)v;return 0;
}
static int read_half(const char *path,_Float16 *x,size_t n) {
    FILE *f=fopen(path,"rb");if(!f){perror(path);return -1;}
    int ok=fread(x,2,n,f)==n;
    if(ok)ok=fgetc(f)==EOF&&!ferror(f);
    if(fclose(f))ok=0;
    for(size_t i=0;ok&&i<n;i++)if(!isfinite((double)x[i]))ok=0;
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
    if(argc<10||argc>14||argc==13||(argc==14&&strcmp(argv[12],"rmsnorm"))){usage(argv[0]);return 1;}
    uint16_t endian=1;
    if(*(unsigned char *)&endian!=1||sizeof(_Float16)!=2)return 1;
    unsigned m,d,h,repeat=1;
    if(number(argv[1],4096,&m)||number(argv[2],4096,&d)||number(argv[3],4096,&h)||
       (argc>=12&&number(argv[11],1000,&repeat))){usage(argv[0]);return 1;}
    enum co_mode mode=CO_CHAIN;
    if(argc>=11){if(!strcmp(argv[10],"separate"))mode=CO_SEPARATE;else if(strcmp(argv[10],"chain"))return 1;}
    for(unsigned i=4;i<9;i++)if(!strcmp(argv[i],argv[9]))return 1;
    if(argc==14&&!strcmp(argv[13],argv[9]))return 1;
    int rc=1,fd=-1;co_graph *g=NULL;
    _Float16 *x=malloc((size_t)m*d*2),*wg=malloc((size_t)h*d*2),*wu=malloc((size_t)h*d*2),
        *wd=malloc((size_t)d*h*2),*b=strcmp(argv[8],"-")?malloc(h*2):NULL,
        *y=malloc((size_t)m*d*2),*first=malloc((size_t)m*d*2),*gamma=argc==14?malloc(d*2):NULL;
    if(!x||!wg||!wu||!wd||!y||!first||(argc==14&&!gamma)||(strcmp(argv[8],"-")&&!b))goto done;
    if(read_half(argv[4],x,(size_t)m*d)||read_half(argv[5],wg,(size_t)h*d)||
       read_half(argv[6],wu,(size_t)h*d)||read_half(argv[7],wd,(size_t)d*h)||
       (b&&read_half(argv[8],b,h))||(gamma&&read_half(argv[13],gamma,d)))goto done;
    co_graph_node nodes[]={
        {.op=CO_GRAPH_INPUT,.rows=m,.cols=d},
        {.op=CO_GRAPH_LINEAR,.a=0,.cols=h,.weights=wg,.bias=b},
        {.op=CO_GRAPH_LINEAR,.a=0,.cols=h,.weights=wu},
        {.op=CO_GRAPH_SILU,.a=1},{.op=CO_GRAPH_MUL,.a=3,.b=2},
        {.op=CO_GRAPH_LINEAR,.a=4,.cols=d,.weights=wd},{.op=CO_GRAPH_ADD,.a=5,.b=0},
        {.op=CO_GRAPH_RMSNORM,.a=0,.gamma=gamma,.epsilon=1e-5f}};
    if(gamma){nodes[1].a=7;nodes[2].a=7;}
    co_graph_desc desc={nodes,gamma?8:7,6,0};co_graph_plan plan;
    int status=co_graph_plan_init(&desc,&plan);if(status){fprintf(stderr,"plan: %d\n",status);goto done;}
    fd=rocket_open();if(fd<0){rc=77;goto done;}
    status=co_graph_compile(fd,&desc,mode,&g);if(status){fprintf(stderr,"compile: %d\n",status);goto done;}
    free(wg);wg=NULL;free(wu);wu=NULL;free(wd);wd=NULL;free(b);b=NULL;free(gamma);gamma=NULL;
    uploads=downloads=0;rocket_submit_counters_reset();co_graph_input input={0,x};
    for(unsigned rep=0;rep<repeat;rep++) {
        status=co_graph_run(g,&input,1,y);if(status){fprintf(stderr,"run: %d\n",status);goto done;}
        if(!rep)memcpy(first,y,(size_t)m*d*2);
        else if(memcmp(first,y,(size_t)m*d*2)){fprintf(stderr,"repeat mismatch\n");goto done;}
    }
    if(uploads!=((unsigned long long)plan.tensors[0].tiles+plan.host_upload_calls)*repeat||
       downloads!=((unsigned long long)plan.tensors[6].tiles+plan.statistic_download_calls)*repeat){fprintf(stderr,"boundary transfer mismatch\n");goto done;}
    if(write_half(argv[9],y,(size_t)m*d))goto done;
    printf("GRAPH m=%u d=%u h=%u mode=%s slots=%u device_bytes=%zu repeat=%u submits=%llu tasks=%llu uploads=%llu downloads=%llu PASS\n",
        m,d,h,mode==CO_CHAIN?"chain":"separate",plan.slots,plan.device_bytes,repeat,
        (unsigned long long)rocket_submit_ioctl_count(),(unsigned long long)rocket_submit_task_count(),uploads,downloads);
    rc=0;
done:
    co_graph_destroy(g);if(fd>=0)rocket_close(fd);
    free(x);free(wg);free(wu);free(wd);free(b);free(y);free(first);free(gamma);return rc;
}
