// SPDX-License-Identifier: GPL-3.0-or-later
/* Candidate probe: can the DPU int16 TP writer act as a raw 16-bit reorder?
 *
 * It follows the pinned upstream int16 TP test path: raw rocket BOs,
 * rocket_submit_matmul, identity weights, and the public TP output layout.
 * It is deliberately narrow: tests this documented hardware writer, not all
 * possible reorder mechanisms.
 */
#include "custom_ops.h"
#include "npu_matmul.h"
#include "rocket_npu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(call) do { int s_=(call); if(s_){fprintf(stderr,"FAIL %s:%d %s=%d\n",__FILE__,__LINE__,#call,s_);return 1;} } while(0)

static size_t tp_idx(unsigned M, unsigned N, unsigned m, unsigned n) {
    (void)N;
    unsigned na = n / 4u;
    return (size_t)4u*m + (na % 4u) + (size_t)(na/4u)*(4u*M) +
           (size_t)(n%4u)*(16u*M);
}
static size_t weight_idx(unsigned n, unsigned k) {
    return (((size_t)(n/16u)*1u + k/32u)*16u + n%16u)*32u + k%32u;
}
static int run_case(int fd, unsigned M, unsigned N, unsigned K, int labels) {
    rocket_bo regcmd={0}, input={0}, weights={0}, output={0};
    uint64_t regs[256]={0};
    size_t in_bytes=(size_t)M*K*sizeof(int16_t);
    size_t wt_bytes=(size_t)N*K*sizeof(int16_t);
    size_t out_bytes=(size_t)M*N*sizeof(int32_t);
    size_t tp_span=(tp_idx(M,N,M-1,N-1)+1)*sizeof(int16_t);
    if (tp_span>out_bytes) out_bytes=tp_span;
    int ret=0;
    if ((ret|=rocket_bo_alloc(fd,4096,&regcmd)) ||
        (ret|=rocket_bo_alloc(fd,in_bytes,&input)) ||
        (ret|=rocket_bo_alloc(fd,wt_bytes,&weights)) ||
        (ret|=rocket_bo_alloc(fd,out_bytes,&output))) {
        fprintf(stderr,"bo alloc failed %d M=%u N=%u\n",ret,M,N); return 2;
    }
    int16_t *A=calloc((size_t)M*K,sizeof(int16_t));
    int16_t *B=calloc((size_t)N*K,sizeof(int16_t));
    if(!A||!B) return 2;
    /* labels mode: unique 16-bit values (address-permutation oracle).
     * pattern mode: raw FP16-like special bit patterns (bit-preservation
     * oracle, with duplicate values possible). */
    static const uint16_t patterns[] = {
        0x0000, 0x8000, 0x7c00, 0xfc00, 0x0001, 0x7fff, 0xffff,
        0x3c00, 0xc000, 0x7bff, 0x0400, 0x83ff, 0x007f, 0x807f
    };
    for(unsigned m=0;m<M;m++) for(unsigned k=0;k<K;k++) {
        size_t i=(size_t)m*K+k;
        A[i] = labels ? (int16_t)(i + 1)
                      : (int16_t)patterns[i % (sizeof(patterns)/sizeof(patterns[0]))];
    }
    for(unsigned n=0;n<N;n++) for(unsigned k=0;k<K;k++)
        B[(size_t)n*K+k]=(int16_t)(n==k?1:0);
    setenv("ROCKET_INT16_TP_ORG","1",1);
    setenv("ROCKET_INT16_TP_PREC","1",1);
    setenv("ROCKET_INT16_SIZE_E","7",1);
    setenv("ROCKET_INT16_SURF_MULT","8",1);
    matmul_params_t p={0};
    p.m=M; p.k=K; p.n=N; p.input_dma=(uint32_t)input.dma_address;
    p.weights_dma=(uint32_t)weights.dma_address;
    p.output_dma=(uint32_t)output.dma_address; p.tasks=regs;
    if((ret=gen_matmul_int16(&p))!=0){fprintf(stderr,"gen rc=%d M=%u N=%u\n",ret,M,N);return 3;}
    CK(rocket_bo_prep(fd,&regcmd,1,0)); CK(rocket_bo_prep(fd,&input,1,0));
    CK(rocket_bo_prep(fd,&weights,1,0));
    memcpy(regcmd.ptr,regs,(size_t)p.task_count*sizeof(uint64_t));
    memset(input.ptr,0,input.size); memset(weights.ptr,0,weights.size);
    int16_t *idst=input.ptr,*wdst=weights.ptr;
    for(unsigned n=1;n<=N;n++) for(unsigned k=1;k<=K;k++)
        wdst[weight_int16(K,n,k)] = B[(size_t)(n-1)*K+(k-1)];
    for(unsigned m=1;m<=M;m++) for(unsigned k=1;k<=K;k++)
        idst[feature_data(K,M,1,8,k,m,1)] = A[(size_t)(m-1)*K+(k-1)];
    CK(rocket_bo_fini(fd,&regcmd)); CK(rocket_bo_fini(fd,&input));
    CK(rocket_bo_fini(fd,&weights));
    CK(rocket_bo_prep(fd,&output,1,0)); memset(output.ptr,0xAA,output.size);
    CK(rocket_bo_fini(fd,&output));
    uint32_t inh[3]={input.handle,weights.handle,regcmd.handle};
    uint32_t outh[1]={output.handle};
    ret=rocket_submit_matmul(fd,&regcmd,p.task_count,inh,3,outh,1,6000);
    if(ret){fprintf(stderr,"submit rc=%d M=%u N=%u\n",ret,M,N);return 4;}
    if(rocket_bo_prep(fd,&output,0,2000000000LL)){fprintf(stderr,"prep output failed\n");return 5;}
    const int16_t *out=output.ptr;
    unsigned tp_ok=0,w_ok=0,tp_n=0,w_n=0;
    for(unsigned m=0;m<M;m++) for(unsigned n=0;n<N;n++) {
        int16_t want=A[(size_t)m*K+n];
        tp_n++; if(out[tp_idx(M,N,m,n)]==want) tp_ok++;
        w_n++; if(out[weight_idx(n,m)]==want) w_ok++;
    }
    printf("CASE M=%u N=%u K=%u kind=%s task=%u tp_match=%u/%u weight_match=%u/%u\n",
           M,N,K,labels?"label":"pattern",p.task_count,tp_ok,tp_n,w_ok,w_n);
    for(unsigned m=0;m<M&&m<4;m++) for(unsigned n=0;n<N&&n<4;n++)
        printf("  tp[%u,%u] idx=%zu got=%d want=%d  w[%u,%u] idx=%zu got=%d\n",
               m,n,tp_idx(M,N,m,n),(unsigned)(uint16_t)out[tp_idx(M,N,m,n)],
               (unsigned)(uint16_t)A[(size_t)m*K+n],
               n,m,weight_idx(n,m),(unsigned)(uint16_t)out[weight_idx(n,m)]);
    rocket_bo_free(fd,&regcmd); rocket_bo_free(fd,&input);
    rocket_bo_free(fd,&weights); rocket_bo_free(fd,&output);
    free(A); free(B);
    return (tp_ok==tp_n)?0:6;
}
int main(void) {
    int fd=rocket_open();
    if(fd<0){fprintf(stderr,"SKIP no NPU: %d\n",fd);return 77;}
    int a=run_case(fd,16,32,32,1);
    int b=run_case(fd,16,32,32,0);
    rocket_close(fd);
    printf("RESULT label_tp=%s pattern_tp=%s\n",
           a==0?"PASS":"FAIL", b==0?"PASS":"FAIL");
    return (a==0 && b==0) ? 0 : 1;
}
