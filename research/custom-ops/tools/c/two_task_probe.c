// SPDX-License-Identifier: GPL-3.0-or-later
#define _GNU_SOURCE
#include "custom_ops.h"
#include "rocket_chain.h"
#include <ctype.h>
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static unsigned checks,failed;
static void (*stage)(unsigned,const char *);
static FILE *records;
static unsigned mask,sptr;
static void witness(unsigned round,const char *label,const void *data,size_t bytes) {
    const char *dir=getenv("TT_TRACE_DIR");if(!dir||round>=3)return;
    char path[1024];snprintf(path,sizeof(path),"%s/round-%04u-%s.bin",dir,round,label);
    FILE *f=fopen(path,"wb");if(!f||fwrite(data,1,bytes,f)!=bytes||fclose(f)){perror(path);exit(1);}
}
static uint32_t hash32(uint32_t x) { x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16); }
static _Float16 hs(double x) { return (_Float16)fmin(1,fmax(0,x/6+0.5)); }

static int score(const char *name,const _Float16 *got,const _Float16 *ref,size_t n,int exact,double *max) {
    size_t bad=0;double peak=0;
    for(size_t i=0;i<n;i++) {
        double d=fabs((double)got[i]-(double)ref[i]);
        if(!isfinite(d)||(exact?memcmp(got+i,ref+i,2)!=0:d>0.002))bad++;
        if(isfinite(d)&&d>peak)peak=d;
    }
    if(peak>*max)*max=peak;
    printf("CHECK %s n=%zu max_abs=%.9g bad=%zu %s\n",name,n,peak,bad,bad?"FAIL":"PASS");
    checks++;if(bad)failed++;return bad?1:0;
}

static int irqs(uint64_t out[3]) {
    FILE *f=fopen("/proc/interrupts","r");if(!f)return -1;
    char line[2048];unsigned found=0;memset(out,0,3*sizeof(*out));
    while(fgets(line,sizeof(line),f))if(strstr(line,"fdab0000.npu")) {
        if(found>=3){fclose(f);return -1;}
        char *p=strchr(line,':');if(!p){fclose(f);return -1;}p++;
        for(;;) {while(isspace((unsigned char)*p))p++;if(!isdigit((unsigned char)*p))break;
            char *end;out[found]+=strtoull(p,&end,10);p=end;}
        found++;
    }
    fclose(f);return found==3?0:-1;
}

static int stamp(co_view v) {
    int rc=rocket_bo_prep(v.buffer->fd,&v.buffer->bo,1,0);if(rc)return rc;
    memset(v.buffer->bo.ptr,0xa5,v.buffer->payload);
    memset((char *)v.buffer->bo.ptr+v.offset,0xff,v.tensor.bytes);
    return rocket_bo_fini(v.buffer->fd,&v.buffer->bo);
}

static int boundary(co_view v,int output,const void *snapshot) {
    int rc=rocket_bo_prep(v.buffer->fd,&v.buffer->bo,0,2000000000ull);if(rc)return rc;
    unsigned char *p=v.buffer->bo.ptr;
    if(output) {
        for(size_t i=0;i<v.offset;i++)if(p[i]!=0xa5)rc=1;
        for(size_t i=v.offset+v.tensor.bytes;i<v.buffer->payload;i++)if(p[i]!=0xa5)rc=1;
    }else if(memcmp(p,snapshot,v.buffer->payload))rc=1;
    int end=rocket_bo_fini(v.buffer->fd,&v.buffer->bo);if(!rc)rc=end;
    if(!rc)rc=co_guard_check(v.buffer);
    return rc;
}

static int padding(co_view v,int second_activation) {
    int rc=rocket_bo_prep(v.buffer->fd,&v.buffer->bo,0,2000000000ull);if(rc)return rc;
    _Float16 *data=(_Float16 *)((char *)v.buffer->bo.ptr+v.offset);
    double expected=second_activation?(double)hs(0.5):0.5;
    for(unsigned r=0;r<v.tensor.physical_rows;r++)for(unsigned c=0;c<v.tensor.physical_cols;c++)
        if(r>=v.tensor.rows||c>=v.tensor.cols) {
            double got=data[co_index(&v.tensor,r,c)];
            if(!isfinite(got)||fabs(got-expected)>0.002)rc=1;
        }
    int end=rocket_bo_fini(v.buffer->fd,&v.buffer->bo);return rc?rc:end;
}

static int set_sptr(co_program *p) {
    unsigned found=0;
    for(unsigned i=0;i<p->count;i++) {
        unsigned reg=(unsigned)(p->words[i]&65535);
        if(reg==0x4004||reg==0x5004) {
            p->words[i]=(p->words[i]&0xffff00000000ffffull)|((uint64_t)sptr<<16);found++;
        }
    }
    return found==2?0:1;
}

#define TRY(x) do {int e_=(x);if(e_){fprintf(stderr,"FAIL line=%d %s result=%d\n",__LINE__,#x,e_);rc=1;goto cleanup;}}while(0)
static int round_run(int fd,unsigned round) {
    const unsigned shapes[3][2]={{16,32},{29,45},{64,64}};
    unsigned m=shapes[round%3][0],n=shapes[round%3][1];size_t count=(size_t)m*n;
    int rc=0;co_tensor t;co_buffer b[6]={0};co_view v[6];co_program p[2];
    _Float16 *xa=malloc(count*2),*xb=malloc(count*2),*ra=malloc(count*2),*rb=malloc(count*2),
        *rd=malloc(count*2),*got=malloc(count*2),*baseline=malloc(count*2);
    unsigned char *snap[2]={0};unsigned before_checks=checks;
    uint64_t delta[4][3]={{0}};double error[4]={0};unsigned armchecks[4]={0};
    if(!xa||!xb||!ra||!rb||!rd||!got||!baseline){rc=1;goto cleanup;}
    TRY(co_tensor_matrix(&t,m,n,CO_F16));
    for(unsigned i=0;i<6;i++) {TRY(co_buffer_alloc(fd,t.bytes+128,&b[i]));TRY(co_view_init(&v[i],&b[i],64,&t));}
    for(size_t i=0;i<count;i++) {
        double a=((int)(hash32((uint32_t)i+419)%257)-128)/16.0;
        if(round%2)a=-a+0.25;
        xa[i]=(_Float16)(round%13==0?0:a);
        xb[i]=(_Float16)(-a*0.75+(round%2?1.25:-1.5));
        ra[i]=hs((double)xa[i]);rb[i]=hs((double)xb[i]);rd[i]=hs((double)ra[i]);
    }
    TRY(co_upload_f16(&v[0],xa));TRY(co_upload_f16(&v[1],xb));
    witness(round,"input-a",xa,count*2);witness(round,"input-b",xb,count*2);
    for(unsigned i=0;i<2;i++) {
        snap[i]=malloc(b[i].payload);if(!snap[i]){rc=1;goto cleanup;}
        TRY(rocket_bo_prep(fd,&b[i].bo,0,0));memcpy(snap[i],b[i].bo.ptr,b[i].payload);TRY(rocket_bo_fini(fd,&b[i].bo));
    }
    for(unsigned arm=0;arm<4;arm++) {
        const char *names[] = {"A","B","C","D"};
        if(stage)stage(round,names[arm]);
        /* Stamp all outputs so writes to a non-participating BO are also detectable. */
        for(unsigned i=2;i<6;i++)TRY(stamp(v[i]));
        co_view o0=arm==3?v[4]:v[2],o1=arm==3?v[5]:v[3];
        TRY(co_emit_lut(&p[0],v[0],o0,CO_HARDSIGMOID));TRY(set_sptr(&p[0]));
        if(arm) {TRY(co_emit_lut(&p[1],arm==3?v[4]:v[1],o1,CO_HARDSIGMOID));TRY(set_sptr(&p[1]));}
        uint64_t irq0[3],irq1[3];TRY(irqs(irq0));rocket_submit_counters_reset();
        TRY(co_execute(fd,p,arm?2:1,arm>=2?CO_CHAIN:CO_SEPARATE));TRY(irqs(irq1));
        unsigned expected_submit=arm==1?2:1,expected_task=arm?2:1;
        if(rocket_submit_ioctl_count()!=expected_submit||rocket_submit_task_count()!=expected_task){rc=1;goto cleanup;}
        for(unsigned i=0;i<3;i++) {
            delta[arm][i]=irq1[i]-irq0[i];
            unsigned expected=mask==(1u<<i)?expected_submit:0;
            if(delta[arm][i]!=expected){fprintf(stderr,"IRQ mismatch round=%u arm=%s core=%u delta=%llu expected=%u\n",round,names[arm],i,(unsigned long long)delta[arm][i],expected);rc=1;goto cleanup;}
        }
        unsigned start=checks;
        TRY(co_download(&o0,got));TRY(score("task0-oracle",got,ra,count,0,&error[arm]));
        char label[64];snprintf(label,sizeof(label),"arm-%s-task-0",names[arm]);witness(round,label,got,count*2);
        if(!arm)memcpy(baseline,got,count*2);
        else TRY(score("task0-vs-A",got,baseline,count,1,&error[arm]));
        if(arm) {TRY(co_download(&o1,got));TRY(score(arm==3?"dependent-task1":"independent-task1",got,arm==3?rd:rb,count,0,&error[arm]));
            snprintf(label,sizeof(label),"arm-%s-task-1",names[arm]);witness(round,label,got,count*2);
        }
        armchecks[arm]=checks-start;
        TRY(padding(o0,0));if(arm)TRY(padding(o1,arm==3));
        for(unsigned i=0;i<6;i++) {
            TRY(boundary(v[i],i>=2,i<2?snap[i]:NULL));
            if(i>=2 && &b[i]!=o0.buffer && (arm==0||&b[i]!=o1.buffer)) {
                TRY(rocket_bo_prep(fd,&b[i].bo,0,0));
                unsigned char *bytes=(unsigned char *)b[i].bo.ptr+v[i].offset;
                for(size_t j=0;j<t.bytes;j++)if(bytes[j]!=0xff){rc=1;goto cleanup;}
                TRY(rocket_bo_fini(fd,&b[i].bo));
            }
        }
    }
    fprintf(records,"{\"event\":\"round\",\"round\":%u,\"rows\":%u,\"cols\":%u,\"pattern\":%u,\"pass\":true,\"checks\":%u,\"arms\":[",round,m,n,round%2,checks-before_checks);
    for(unsigned arm=0;arm<4;arm++)fprintf(records,"%s{\"arm\":\"%c\",\"submits\":%u,\"tasks\":%u,\"irq_delta\":[%llu,%llu,%llu],\"checks\":%u,\"max_abs\":%.9g,\"padding_guard\":true,\"pass\":true}",
        arm?",":"",'A'+arm,arm==1?2:1,arm?2:1,(unsigned long long)delta[arm][0],(unsigned long long)delta[arm][1],(unsigned long long)delta[arm][2],armchecks[arm],error[arm]);
    fprintf(records,"]}\n");fflush(records);
cleanup:
    for(unsigned i=0;i<6;i++)co_buffer_free(&b[i]);
    for(unsigned i=0;i<2;i++)free(snap[i]);
    free(xa);free(xb);free(ra);free(rb);free(rd);free(got);free(baseline);return rc;
}

static int host(void) {
    co_tensor t;co_buffer b[3]={0};co_view v[3];co_program p[2];
    int rc=co_tensor_matrix(&t,29,45,CO_F16);if(rc)return 1;
    for(unsigned i=0;i<3;i++) {b[i].payload=t.bytes;b[i].bo=(rocket_bo){.handle=i+1,.dma_address=0x100000+i*0x10000,.size=t.bytes};if(co_view_init(&v[i],&b[i],0,&t))return 1;}
    sptr=14;
    if(co_emit_lut(&p[0],v[0],v[1],CO_HARDSIGMOID)||co_emit_lut(&p[1],v[1],v[2],CO_HARDSIGMOID))return 1;
    size_t stride=rkt_chain_words(p[0].count),size=(stride+rkt_chain_words(p[1].count))*8;
    uint64_t *words=calloc(1,size);if(!words)return 1;
    rocket_bo bo={.handle=4,.dma_address=0x400000,.ptr=words,.size=size};rocket_task_desc tasks[2];
    rc=rkt_chain_pack_at(&bo,tasks,0,0,stride,p[0].words,p[0].count,p[1].count)||
        rkt_chain_pack_at(&bo,tasks,1,stride,stride,p[1].words,p[1].count,p[1].count)||rkt_chain_verify(&bo,tasks,2);
    if(!rc) {tasks[1].regcmd+=16;if(rkt_chain_verify(&bo,tasks,2)==0)rc=1;}
    if(!rc) {bo.size=size-8;if(rkt_chain_verify(&bo,tasks,2)==0)rc=1;}
    free(words);printf("HOST_TWO_TASK %s\n",rc?"FAIL":"PASS");return rc?1:0;
}

int main(int argc,char **argv) {
    if(argc==2&&!strcmp(argv[1],"host"))return host();
    unsigned rounds=argc>1?(unsigned)strtoul(argv[1],NULL,0):128;
    sptr=argc>2?(unsigned)strtoul(argv[2],NULL,0):14;
    const char *m=getenv("RKNPU_CORE_MASK");mask=m?(unsigned)strtoul(m,NULL,0):0;
    const char *path=getenv("TT_PROBE_JSON");
    if(argc>3||rounds<2||rounds>4096||(sptr!=14&&sptr!=48)||(mask!=1&&mask!=2&&mask!=4)||!path){fprintf(stderr,"usage: RKNPU_CORE_MASK=1 TT_PROBE_JSON=file %s [rounds>=2] [14|48]\n",argv[0]);return 1;}
    records=fopen(path,"w");if(!records){perror(path);return 1;}
    void *symbol=dlsym(RTLD_DEFAULT,"tt_stage");memcpy(&stage,&symbol,sizeof(stage));
    if(!stage){fprintf(stderr,"Two-task lifecycle trace observer must be loaded\n");fclose(records);return 1;}
    int fd=rocket_open();if(fd<0){fclose(records);return 77;}
    unsigned completed=0;int rc=0;
    for(;completed<rounds;completed++)if((rc=round_run(fd,completed)))break;
    rocket_close(fd);
    fprintf(records,"{\"event\":\"summary\",\"rounds\":%u,\"checks\":%u,\"failures\":%u,\"core_mask\":%u,\"sptr\":%u,\"status\":\"%s\"}\n",completed,checks,failed+(rc!=0),mask,sptr,rc?"FAIL":"PASS");
    fclose(records);printf("TWO_TASK_SUMMARY rounds=%u checks=%u status=%s\n",completed,checks,rc?"FAIL":"PASS");return rc?1:0;
}
