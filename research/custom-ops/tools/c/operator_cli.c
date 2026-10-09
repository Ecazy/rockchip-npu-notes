// SPDX-License-Identifier: GPL-3.0-or-later
#include "custom_ops.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void usage(const char *name) {
    printf("usage: %s M K N X.bin W.bin b.bin Y.bin [chain|separate] [repeat]\n",name);
    printf("row-major little-endian IEEE FP16: X[M,K], W[N,K], b[N], Y[M,N]\n");
    printf("M/N: 1..256; K: 1..%u; repeat: 1..1000; default: chain, 1 run; no CPU fallback\n",
           (unsigned)CO_MAX_K);
}

static int number(const char *s, unsigned max, unsigned *out) {
    if (!s || !*s || *s=='-' || *s=='+') return -1;
    errno=0;char *end=NULL;unsigned long v=strtoul(s,&end,10);
    if (errno || *end || !v || v>max) return -1;
    *out=(unsigned)v;return 0;
}

static int read_tensor(const char *path,_Float16 *v,size_t n) {
    FILE *f=fopen(path,"rb");
    if (!f) {perror(path);return -1;}
    int ok=fread(v,sizeof(*v),n,f)==n;
    if (ok) ok=fgetc(f)==EOF && !ferror(f);
    if (fclose(f)) ok=0;
    if (!ok) {fprintf(stderr,"%s: expected exactly %zu FP16 elements\n",path,n);return -1;}
    for (size_t i=0;i<n;i++) if(!isfinite((double)v[i])) {
        fprintf(stderr,"%s: nonfinite input at %zu\n",path,i);return -1;
    }
    return 0;
}

static int write_tensor(const char *path,const _Float16 *v,size_t n) {
    size_t len=strlen(path)+32;char *temp=malloc(len);
    if (!temp) return -1;
    snprintf(temp,len,"%s.tmp.XXXXXX",path);
    int fd=mkstemp(temp);
    if (fd<0) {perror(path);free(temp);return -1;}
    FILE *f=fdopen(fd,"wb");
    if (!f) {close(fd);unlink(temp);free(temp);return -1;}
    int ok=fwrite(v,sizeof(*v),n,f)==n;
    if (fclose(f)) ok=0;
    if(ok && rename(temp,path)) {perror(path);ok=0;}
    if(!ok)unlink(temp);
    free(temp);return ok?0:-1;
}

int main(int argc,char **argv) {
    if(argc==2 && !strcmp(argv[1],"--help")) {usage(argv[0]);return 0;}
    if(argc<8 || argc>10) {usage(argv[0]);return 1;}
    uint16_t endian=1;
    if(*(unsigned char *)&endian!=1 || sizeof(_Float16)!=2) {
        fprintf(stderr,"requires a little-endian FP16 host\n");return 1;
    }
    unsigned m,k,n,repeats=1;
    if(number(argv[1],256,&m)||number(argv[2],CO_MAX_K,&k)||number(argv[3],256,&n)||
       (argc==10 && number(argv[9],1000,&repeats))) {usage(argv[0]);return 1;}
    enum co_mode mode=CO_CHAIN;
    if(argc>=9) {
        if(!strcmp(argv[8],"separate")) mode=CO_SEPARATE;
        else if(strcmp(argv[8],"chain")) {usage(argv[0]);return 1;}
    }
    for(unsigned i=4;i<7;i++) if(!strcmp(argv[i],argv[7])) {
        fprintf(stderr,"output must differ from input paths\n");return 1;
    }
    _Float16 *x=malloc((size_t)m*k*2),*w=malloc((size_t)n*k*2),*b=malloc(n*2),
        *y=malloc((size_t)m*n*2),*first=malloc((size_t)m*n*2);
    int rc=1,fd=-1;co_linear_hsig *ctx=NULL;
    if(!x||!w||!b||!y||!first)goto cleanup;
    if(read_tensor(argv[4],x,(size_t)m*k)||read_tensor(argv[5],w,(size_t)n*k)||read_tensor(argv[6],b,n))goto cleanup;
    fd=rocket_open();
    if(fd<0) {fprintf(stderr,"no NPU: %d\n",fd);rc=77;goto cleanup;}
    int status=co_linear_hsig_create(fd,m,k,n,w,b,mode,&ctx);
    if(status) {fprintf(stderr,"operator create: %d\n",status);goto cleanup;}
    /* Prove W and b were copied at creation, not borrowed for later host compute. */
    free(w);w=NULL;free(b);b=NULL;
    rocket_submit_counters_reset();
    for(unsigned i=0;i<repeats;i++) {
        status=co_linear_hsig_run(ctx,x,y);
        if(status) {fprintf(stderr,"operator run %u: %d\n",i,status);goto cleanup;}
        if(!i)memcpy(first,y,(size_t)m*n*2);
        else if(memcmp(first,y,(size_t)m*n*2)) {fprintf(stderr,"repeated output differs\n");goto cleanup;}
    }
    if(write_tensor(argv[7],y,(size_t)m*n))goto cleanup;
    printf("OPERATOR HardSigmoid(XW^T+b) m=%u k=%u n=%u mode=%s repeat=%u submits=%llu tasks=%llu PASS\n",
        m,k,n,mode==CO_CHAIN?"chain":"separate",repeats,
        (unsigned long long)rocket_submit_ioctl_count(),(unsigned long long)rocket_submit_task_count());
    rc=0;
cleanup:co_linear_hsig_destroy(ctx);if(fd>=0)rocket_close(fd);free(x);free(w);free(b);free(y);free(first);return rc;
}
