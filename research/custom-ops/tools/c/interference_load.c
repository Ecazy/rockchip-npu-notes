// SPDX-License-Identifier: GPL-3.0-or-later
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
static volatile sig_atomic_t stop;
static int streaming;static unsigned threads,seconds,total_mb;
static uint64_t deadline;
typedef struct{unsigned id;uint64_t bytes,loops,checksum;int rc;float *a,*b;} lane;
static uint64_t now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000ull+t.tv_nsec;}
static void halt(int s){(void)s;stop=1;}
static void *work(void *arg){
    lane *p=arg;cpu_set_t cpus;CPU_ZERO(&cpus);CPU_SET(4+p->id, &cpus);
    if(pthread_setaffinity_np(pthread_self(),sizeof(cpus),&cpus)){p->rc=1;return NULL;}
    size_t count=(size_t)total_mb*1024*1024/threads/2/sizeof(float);
    double x=.71;uint64_t sum=0;
    while(!stop&&now()<deadline){
        if(streaming){for(size_t i=0;i<count;i++)p->b[i]=p->a[i]*1.000001f+.00001f;
            float *temp=p->a;p->a=p->b;p->b=temp;p->bytes+=count*sizeof(float)*2;
            sum+=(uint64_t)(p->a[p->loops%count]*1000);
        }else{for(unsigned i=0;i<100000;i++)x=x*.999999+1e-7;sum+=(uint64_t)(x*1e9);}
        p->loops++;
    }
    p->checksum=sum;return NULL;
}
int main(int argc,char **argv){
    if(argc!=5)return 2;
    streaming=!strcmp(argv[1],"stream");if(!streaming&&strcmp(argv[1],"compute"))return 2;
    threads=(unsigned)atoi(argv[2]);seconds=(unsigned)atoi(argv[3]);total_mb=(unsigned)atoi(argv[4]);
    if(!threads||threads>4||!seconds||seconds>3600||total_mb<16||total_mb>256)return 2;
    signal(SIGINT,halt);signal(SIGTERM,halt);
    lane l[4]={0};pthread_t t[4];size_t bytes=(size_t)total_mb*1024*1024/threads/2;
    for(unsigned i=0;i<threads;i++){
        l[i].id=i;l[i].a=malloc(bytes);l[i].b=malloc(bytes);
        if(!l[i].a||!l[i].b)return 1;
        for(size_t j=0;j<bytes/sizeof(float);j++)l[i].a[j]=l[i].b[j]=.001f;
    }
    uint64_t begin=now();deadline=begin+(uint64_t)seconds*1000000000ull;
    unsigned created=0;int rc=0;
    for(unsigned i=0;i<threads;i++){if(pthread_create(&t[i],NULL,work,&l[i])){rc=1;stop=1;break;}created++;}
    /* READY means buffers are allocated and pre-touched, not calibrated load. */
    printf("{\"type\":\"ready\",\"mode\":\"%s\",\"threads\":%u,\"total_buffer_mb\":%u,\"begin_ns\":%llu}\n",argv[1],threads,total_mb,(unsigned long long)begin);fflush(stdout);
    uint64_t traffic=0,loops=0,checksum=0;
    for(unsigned i=0;i<created;i++){pthread_join(t[i],NULL);traffic+=l[i].bytes;loops+=l[i].loops;checksum+=l[i].checksum;if(l[i].rc)rc=1;}
    uint64_t end=now();
    printf("{\"type\":\"done\",\"end_ns\":%llu,\"software_stream_bytes\":%llu,\"loops\":%llu,\"checksum\":%llu,\"software_gb_per_s\":%.6f,\"result\":%d}\n",(unsigned long long)end,(unsigned long long)traffic,(unsigned long long)loops,(unsigned long long)checksum,(double)traffic/(double)(end-begin),rc);
    for(unsigned i=0;i<threads;i++){free(l[i].a);free(l[i].b);}return rc;
}
