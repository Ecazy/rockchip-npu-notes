// SPDX-License-Identifier: GPL-3.0-or-later
#include "custom_ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRY(e) do {rc=(e);if(rc){fprintf(stderr,"line %d status=%d\n",__LINE__,rc);goto done;}} while(0)
static _Float16 sample(unsigned i,unsigned seed) {
    return (_Float16)(((int)((i*19+seed)%37)-18)/32.0);
}
static int run_case(int fd,unsigned m,unsigned k,unsigned n,enum co_mode mode) {
    int rc=0;co_linear_hsig *op=NULL;co_linear_plan plan;
    co_buffer *xb=NULL,*yb=NULL;co_view *xv=NULL,*yv=NULL;
    _Float16 *x=NULL,*w=NULL,*b=NULL,*host=NULL,*tile=NULL;
    TRY(co_linear_plan_init(&plan,m,k,n,0));
    x=malloc((size_t)m*k*2);w=malloc((size_t)n*k*2);b=malloc(n*2);
    host=malloc((size_t)m*n*2);tile=malloc(256u*256u*2);
    xb=calloc(plan.mtiles,sizeof(*xb));yb=calloc(plan.output_tiles,sizeof(*yb));
    xv=calloc(plan.mtiles,sizeof(*xv));yv=calloc(plan.output_tiles,sizeof(*yv));
    if(!x||!w||!b||!host||!tile||!xb||!yb||!xv||!yv){rc=ROCKET_E_NOMEM;goto done;}
    for(size_t i=0;i<(size_t)m*k;i++)x[i]=sample(i,3);
    for(size_t i=0;i<(size_t)n*k;i++)w[i]=sample(i,11)/4;
    for(unsigned i=0;i<n;i++)b[i]=sample(i,7);
    TRY(co_linear_hsig_create(fd,m,k,n,w,b,mode,&op));
    TRY(co_linear_hsig_run(op,x,host));
    for(unsigned mi=0;mi<plan.mtiles;mi++) {
        co_linear_tile_info info;co_tensor t;
        TRY(co_linear_plan_tile(&plan,mi*plan.ntiles,&info));
        TRY(co_tensor_input(&t,info.rows,k));TRY(co_buffer_alloc(fd,t.bytes+16,&xb[mi]));
        TRY(co_view_init(&xv[mi],&xb[mi],16,&t));
        TRY(co_upload_f16(&xv[mi],x+(size_t)info.m0*k));
    }
    for(unsigned i=0;i<plan.output_tiles;i++) {
        co_linear_tile_info info;co_tensor t;
        TRY(co_linear_plan_tile(&plan,i,&info));TRY(co_tensor_matrix(&t,info.rows,info.cols,CO_F16));
        TRY(co_buffer_alloc(fd,t.bytes+16,&yb[i]));TRY(co_view_init(&yv[i],&yb[i],16,&t));
        TRY(rocket_bo_prep(fd,&yb[i].bo,1,2000000000ull));
        memset(yb[i].bo.ptr,0x5a,16);TRY(rocket_bo_fini(fd,&yb[i].bo));
    }
    rocket_submit_counters_reset();
    for(unsigned rep=0;rep<3;rep++) TRY(co_linear_hsig_run_tiles(op,xv,plan.mtiles,yv,plan.output_tiles));
    unsigned long long submits=rocket_submit_ioctl_count(),tasks=rocket_submit_task_count();
    if(submits!=3*(mode==CO_CHAIN?plan.chain_submits:plan.tasks)||tasks!=3*plan.tasks) {
        fprintf(stderr,"counter mismatch submits=%llu tasks=%llu\n",submits,tasks);rc=ROCKET_E_DEVICE;goto done;
    }
    for(unsigned i=0;i<plan.output_tiles;i++) {
        co_linear_tile_info info;TRY(co_linear_plan_tile(&plan,i,&info));
        TRY(co_download(&yv[i],tile));
        for(unsigned r=0;r<info.rows;r++)for(unsigned c=0;c<info.cols;c++) {
            _Float16 expected=host[((size_t)info.m0+r)*n+info.n0+c];
            if(!isfinite((double)tile[(size_t)r*info.cols+c])||tile[(size_t)r*info.cols+c]!=expected) {
                fprintf(stderr,"device collection differs from host boundary tile=%u row=%u col=%u\n",i,r,c);
                rc=ROCKET_E_DEVICE;goto done;
            }
        }
        TRY(co_guard_check(&yb[i]));TRY(rocket_bo_prep(fd,&yb[i].bo,0,2000000000ull));
        for(unsigned q=0;q<16;q++)if(((unsigned char *)yb[i].bo.ptr)[q]!=0x5a)rc=ROCKET_E_DEVICE;
        int fini=rocket_bo_fini(fd,&yb[i].bo);if(!rc)rc=fini;if(rc)goto done;
    }
    printf("MULTITILE m=%u k=%u n=%u mode=%s output_tiles=%u device_bytes=%zu repeat=3 submits=%llu tasks=%llu PASS\n",
           m,k,n,mode==CO_CHAIN?"chain":"separate",plan.output_tiles,plan.device_bytes,submits,tasks);
done:
    co_linear_hsig_destroy(op);
    if(xb)for(unsigned i=0;i<plan.mtiles;i++)co_buffer_free(&xb[i]);
    if(yb)for(unsigned i=0;i<plan.output_tiles;i++)co_buffer_free(&yb[i]);
    free(xb);free(yb);free(xv);free(yv);free(x);free(w);free(b);free(host);free(tile);
    return rc;
}
int main(void) {
    int fd=rocket_open();if(fd<0)return 77;
    const unsigned shapes[][3]={{257,37,45},{29,37,257},{257,257,257},{513,33,289}};
    int rc=0;
    for(unsigned s=0;s<4&&!rc;s++)for(unsigned mode=0;mode<2&&!rc;mode++)
        rc=run_case(fd,shapes[s][0],shapes[s][1],shapes[s][2],(enum co_mode)mode);
    rocket_close(fd);return rc?1:0;
}
