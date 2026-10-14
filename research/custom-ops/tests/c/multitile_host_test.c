// SPDX-License-Identifier: GPL-3.0-or-later
/* Test-only BO and execution seam: production packing/emission/planning are real.
 * This validates orchestration and ownership, not hardware register numerics. */
#include "custom_ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned failures,checks,live,allocs,execs,fail_alloc,fail_exec;
static size_t resident;
static uint32_t next_handle=1;
#define EXPECT(e) do {checks++;if(!(e)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#e);failures++;}} while(0)
int __wrap_rocket_bo_alloc32(int fd,size_t size,rocket_bo *bo) {
    (void)fd;
    if (++allocs==fail_alloc) return ROCKET_E_NOMEM;
    memset(bo,0,sizeof(*bo));bo->ptr=calloc(1,size);
    if (!bo->ptr) return ROCKET_E_NOMEM;
    bo->size=size;bo->handle=next_handle++;bo->dma_address=(uint64_t)bo->handle*0x400000;
    live++;resident+=size;return 0;
}
void __wrap_rocket_bo_free(int fd,rocket_bo *bo) {
    (void)fd;free(bo->ptr);live--;resident-=bo->size;memset(bo,0,sizeof(*bo));
}
int __wrap_rocket_bo_prep(int fd,rocket_bo *bo,int dir,uint64_t timeout) {
    (void)fd;(void)bo;(void)dir;(void)timeout;return 0;
}
int __wrap_rocket_bo_fini(int fd,rocket_bo *bo) {(void)fd;(void)bo;return 0;}
int __wrap_rocket_batched_submit_supported(void) {return 1;}
static _Float16 *data(co_view v) {return (_Float16 *)((char *)v.buffer->bo.ptr+v.offset);}
int __wrap_co_execute(int fd,const co_program *p,unsigned count,enum co_mode mode) {
    (void)fd;(void)mode;
    if (++execs==fail_exec) return ROCKET_E_DEVICE;
    for (unsigned i=0;i<count;i++) {
        co_view y=p[i].output,x=p[i].reads[0];
        _Float16 *dst=data(y),*src=data(x);
        if (p[i].nreads==1) {
            for (size_t q=0;q<y.tensor.bytes/2;q++) {
                double v=(double)src[q]/6+0.5;
                dst[q]=(_Float16)(v<0?0:v>1?1:v);
            }
        } else {
            co_view w=p[i].reads[1];_Float16 *weights=data(w);
            for (unsigned r=0;r<y.tensor.physical_rows;r++)
                for (unsigned col=0;col<y.tensor.physical_cols;col++) {
                    double value=0;
                    for (unsigned q=0;q<x.tensor.physical_cols;q++)
                        value+=(double)src[co_index(&x.tensor,r,q)]*weights[co_index(&w.tensor,col,q)];
                    if (p[i].nreads==3) {
                        co_view prior=p[i].reads[2];
                        value+=data(prior)[co_index(&prior.tensor,r,col)];
                    }
                    dst[co_index(&y.tensor,r,col)]=(_Float16)value;
                }
        }
    }
    return 0;
}

static _Float16 value(unsigned i,unsigned seed) {return (_Float16)((int)((i*17+seed)%31)-15)/64;}
static void execution_case(unsigned m,unsigned k,unsigned n) {
    size_t xs=(size_t)m*k,ws=(size_t)n*k,ys=(size_t)m*n;
    _Float16 *x=malloc(xs*2),*w=malloc(ws*2),*b=malloc(n*2),*y=malloc(ys*2);
    EXPECT(x&&w&&b&&y);if (!x||!w||!b||!y) exit(1);
    for (size_t i=0;i<xs;i++)x[i]=value(i,2);
    for (size_t i=0;i<ws;i++)w[i]=value(i,9);
    for (unsigned i=0;i<n;i++)b[i]=value(i,5);
    co_linear_plan plan;EXPECT(!co_linear_plan_init(&plan,m,k,n,0));
    for (unsigned mode=0;mode<2;mode++) {
        co_linear_hsig *op=NULL;next_handle=1;
        EXPECT(!co_linear_hsig_create(3,m,k,n,w,b,(enum co_mode)mode,&op));
        if (!op) exit(1);
        EXPECT(resident==plan.device_bytes);
        unsigned before=execs;
        EXPECT(!co_linear_hsig_run(op,x,y));EXPECT(execs-before==plan.output_tiles);
        for (unsigned r=0;r<m;r++) for (unsigned c=0;c<n;c++) {
            _Float16 sum=0;
            for (unsigned k0=0;k0<k;k0+=256) {
                double v=sum;unsigned end=k-k0>256?k0+256:k;
                for (unsigned q=k0;q<end;q++)v+=(double)x[(size_t)r*k+q]*w[(size_t)c*k+q];
                sum=(_Float16)v;
            }
            double v=(double)(_Float16)(sum+b[c])/6+0.5;
            EXPECT(y[(size_t)r*n+c]==(_Float16)(v<0?0:v>1?1:v));
        }
        memset(y,0x23,ys*2);fail_exec=execs+plan.output_tiles;
        EXPECT(co_linear_hsig_run(op,x,y)==ROCKET_E_DEVICE);
        for (size_t i=0;i<ys*2;i++) EXPECT(((unsigned char *)y)[i]==0x23);
        fail_exec=0;
        /* Caller-owned tile cubes exercise the device collection entry. */
        co_buffer *xb=calloc(plan.mtiles,sizeof(*xb)),*yb=calloc(plan.output_tiles,sizeof(*yb));
        co_view *xv=calloc(plan.mtiles,sizeof(*xv)),*yv=calloc(plan.output_tiles,sizeof(*yv));
        EXPECT(xb&&yb&&xv&&yv);if(!xb||!yb||!xv||!yv)exit(1);
        for (unsigned mi=0;mi<plan.mtiles;mi++) {
            co_linear_tile_info info;co_tensor t;
            EXPECT(!co_linear_plan_tile(&plan,mi*plan.ntiles,&info));
            EXPECT(!co_tensor_input(&t,info.rows,k));EXPECT(!co_buffer_alloc(3,t.bytes,&xb[mi]));
            EXPECT(!co_view_init(&xv[mi],&xb[mi],0,&t));
            EXPECT(!co_upload_f16(&xv[mi],x+(size_t)info.m0*k));
        }
        for (unsigned i=0;i<plan.output_tiles;i++) {
            co_linear_tile_info info;co_tensor t;
            EXPECT(!co_linear_plan_tile(&plan,i,&info));EXPECT(!co_tensor_matrix(&t,info.rows,info.cols,CO_F16));
            EXPECT(!co_buffer_alloc(3,t.bytes+16,&yb[i]));EXPECT(!co_view_init(&yv[i],&yb[i],16,&t));
        }
        before=execs;co_view saved=yv[plan.output_tiles-1];yv[plan.output_tiles-1].tensor.cols++;
        EXPECT(co_linear_hsig_run_tiles(op,xv,plan.mtiles,yv,plan.output_tiles)==ROCKET_E_SHAPE);
        EXPECT(execs==before);yv[plan.output_tiles-1]=saved;
        EXPECT(!co_linear_hsig_run_tiles(op,xv,plan.mtiles,yv,plan.output_tiles));
        for (unsigned i=0;i<plan.output_tiles;i++) {
            co_linear_tile_info info;EXPECT(!co_linear_plan_tile(&plan,i,&info));
            _Float16 *tile=malloc((size_t)info.rows*info.cols*2);EXPECT(tile!=NULL);if(!tile)exit(1);
            EXPECT(!co_download(&yv[i],tile));
            for(unsigned r=0;r<info.rows;r++)for(unsigned col=0;col<info.cols;col++)
                y[((size_t)info.m0+r)*n+info.n0+col]=tile[(size_t)r*info.cols+col];
            free(tile);
        }
        /* Compare collection execution with the host assembly independently. */
        _Float16 *host=malloc(ys*2);EXPECT(host!=NULL);if(!host)exit(1);
        EXPECT(!co_linear_hsig_run(op,x,host));EXPECT(!memcmp(y,host,ys*2));free(host);
        if (plan.output_tiles>1) {
            EXPECT(co_linear_hsig_run_cube(op,xv[0],yv[0])==ROCKET_E_UNSUPPORTED);
            before=execs;saved=yv[1];
            co_buffer alias=*yv[0].buffer;
            yv[1].buffer=&alias;yv[1].offset=yv[0].offset;
            EXPECT(co_linear_hsig_run_tiles(op,xv,plan.mtiles,yv,plan.output_tiles)==ROCKET_E_SHAPE);
            EXPECT(execs==before);yv[1]=saved;
        }
        for(unsigned i=0;i<plan.mtiles;i++)co_buffer_free(&xb[i]);
        for(unsigned i=0;i<plan.output_tiles;i++)co_buffer_free(&yb[i]);
        free(xb);free(yb);free(xv);free(yv);
        co_linear_hsig_destroy(op);EXPECT(live==0&&resident==0);
    }
    free(x);free(w);free(b);free(y);
}

int main(void) {
    const unsigned dims[]={1,12,255,256,257,511,512,513,4096};
    for(unsigned a=0;a<9;a++)for(unsigned b=0;b<9;b++) {
        co_linear_plan p;EXPECT(!co_linear_plan_init(&p,dims[a],513,dims[b],0));
        size_t area=0;unsigned long long tasks=0;
        for(unsigned i=0;i<p.output_tiles;i++) {
            co_linear_tile_info t;EXPECT(!co_linear_plan_tile(&p,i,&t));
            EXPECT(t.rows&&t.rows<=256&&t.cols&&t.cols<=256);
            EXPECT(t.m0==i/p.ntiles*256&&t.n0==i%p.ntiles*256);
            EXPECT(t.m0+t.rows<=p.m&&t.n0+t.cols<=p.n);
            area+=(size_t)t.rows*t.cols;tasks+=t.programs;
        }
        EXPECT(area==(size_t)p.m*p.n&&tasks==p.tasks);
        co_linear_plan q;EXPECT(!co_linear_plan_init(&q,p.m,p.k,p.n,p.device_bytes));
        EXPECT(co_linear_plan_init(&q,p.m,p.k,p.n,p.device_bytes-1)==ROCKET_E_TILING);
    }
    co_linear_plan p;EXPECT(co_linear_plan_init(&p,0,1,1,0)==ROCKET_E_SHAPE);
    EXPECT(co_linear_plan_init(&p,4097,1,1,0)==ROCKET_E_SHAPE);
    EXPECT(co_linear_plan_init(&p,1,4097,1,0)==ROCKET_E_SHAPE);
    EXPECT(co_linear_plan_init(&p,1,1,4097,0)==ROCKET_E_SHAPE);
    execution_case(13,37,45);execution_case(257,257,257);execution_case(513,33,289);
    _Float16 w[1]={1},b[1]={0};co_linear_hsig *op=(void *)1;
    fail_alloc=allocs+3;
    EXPECT(co_linear_hsig_create(3,1,1,1,w,b,CO_CHAIN,&op)==ROCKET_E_NOMEM);
    EXPECT(!op&&live==0&&resident==0);
    printf("MULTITILE host checks=%u failures=%u\n",checks,failures);
    return failures?1:0;
}
