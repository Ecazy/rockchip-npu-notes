// SPDX-License-Identifier: GPL-3.0-or-later
#include "co_gdn_cpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"line %d %s\n",__LINE__,#x);return 1;}}while(0)
static float h(float x){return (float)(_Float16)x;}
int main(void){
    const unsigned n=19,m=7;_Float16 initial[133],k[19],q[19],v[7],y[7],got[133];
    float ref[133],fs[133],fy[7];co_gdn_cpu *a=NULL,*b=NULL;int differs=0;
    for(unsigned i=0;i<m*n;i++){initial[i]=(_Float16)(sin(i*1.7)*.2);ref[i]=initial[i];}
    for(unsigned j=0;j<n;j++){k[j]=(_Float16)(cos(j*.8)*.21);q[j]=(_Float16)(sin(j*.9)*.19);}
    for(unsigned r=0;r<m;r++)v[r]=(_Float16)(cos(r*.5)*.3);
    REQUIRE(!co_gdn_cpu_create(m,n,CO_GDN_CPU_F16,&a));REQUIRE(!co_gdn_cpu_create(m,n,CO_GDN_CPU_F32,&b));
    REQUIRE(!co_gdn_cpu_reset(a,initial)&&!co_gdn_cpu_reset(b,initial));
    for(unsigned t=0;t<32;t++){
        _Float16 alpha=(_Float16).973,beta=(_Float16).613;
        for(unsigned r=0;r<m;r++){
            float pred=0;for(unsigned j=0;j<n;j++){ref[r*n+j]=h((float)alpha*ref[r*n+j]);pred=fmaf(ref[r*n+j],(float)k[j],pred);}
            float e=h((float)beta*h((float)v[r]-h(pred))),out=0;
            for(unsigned j=0;j<n;j++){ref[r*n+j]=h(fmaf(e,(float)k[j],ref[r*n+j]));out=fmaf(ref[r*n+j],(float)q[j],out);}
            fy[r]=h(out);
        }
        REQUIRE(!co_gdn_cpu_step(a,k,q,v,alpha,beta,y));REQUIRE(!co_gdn_cpu_export(a,got));
        for(unsigned i=0;i<m*n;i++)REQUIRE(fabs((float)got[i]-ref[i])<.0003);
        for(unsigned r=0;r<m;r++)REQUIRE(fabs((float)y[r]-fy[r])<.0003);
        REQUIRE(!co_gdn_cpu_step(b,k,q,v,alpha,beta,fy));REQUIRE(!co_gdn_cpu_export(b,fs));
        for(unsigned i=0;i<m*n;i++)if(got[i]!=(_Float16)fs[i])differs=1;
    }
    REQUIRE(differs);_Float16 before[133];memcpy(before,got,sizeof(got));
    REQUIRE(co_gdn_cpu_step(a,k,q,v,NAN,1,y)==ROCKET_E_SHAPE);REQUIRE(!co_gdn_cpu_export(a,got));REQUIRE(!memcmp(before,got,sizeof(got)));
    co_gdn_cpu_destroy(a);co_gdn_cpu_destroy(b);puts("CPU staged FP16, independent FP32 storage, invalid-input atomicity PASS");return 0;
}
