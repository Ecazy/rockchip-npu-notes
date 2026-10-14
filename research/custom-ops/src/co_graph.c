// SPDX-License-Identifier: GPL-3.0-or-later
#include "co_graph.h"
#include "co_row_ops.h"
#include "co_tiled.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static unsigned blocks(unsigned n) {return (n+255)/256;}
static unsigned extent(unsigned n,unsigned b) {unsigned r=n-b*256;return r>256?256:r;}
static int binary(enum co_graph_op op) {return op==CO_GRAPH_ADD||op==CO_GRAPH_MUL;}
static int rowop(enum co_graph_op op) {return op==CO_GRAPH_RMSNORM||op==CO_GRAPH_SOFTMAX;}
static int gated(enum co_graph_op op) {return op==CO_GRAPH_SILU||op==CO_GRAPH_GELU;}
static unsigned operands(const co_graph_node *n,unsigned refs[4]) {
    if(n->op==CO_GRAPH_INPUT)return 0;
    refs[0]=n->a;
    if(n->op==CO_GRAPH_ATTENTION){refs[1]=n->b;refs[2]=n->c;refs[3]=n->mask_node;return 3+n->has_mask;}
    if(binary(n->op)){refs[1]=n->b;return 2;}
    return 1;
}
static co_rope_config rope_config(const co_graph_node *n,unsigned rows) {
    return (co_rope_config){rows,n->heads,n->head_dim,n->rotary_dim,n->theta,n->rope_style};
}
static co_attention_config attention_config(const co_graph_node *n,unsigned rows) {
    return (co_attention_config){.max_query=rows,.capacity=rows,.heads=n->heads,.kv_heads=n->kv_heads,
        .head_dim=n->head_dim,.rotary_dim=n->rotary_dim,.theta=n->theta,.scale=n->scale,
        .style=n->rope_style,.causal=n->causal,.window=n->window};
}
static int visit(const co_graph_desc *d,unsigned id,unsigned char *state,co_graph_plan *p) {
    if(id>=d->count)return ROCKET_E_SHAPE;
    if(state[id]==1)return ROCKET_E_SHAPE;
    if(state[id]==2)return 0;
    const co_graph_node *n=&d->nodes[id];
    if(n->op<CO_GRAPH_INPUT||n->op>CO_GRAPH_ATTENTION)return ROCKET_E_UNSUPPORTED;
    if(n->flags&~CO_GRAPH_ALLOW_NEG_INF)return ROCKET_E_SHAPE;
    if(n->flags&&n->op!=CO_GRAPH_INPUT)return ROCKET_E_SHAPE;
    if(n->op==CO_GRAPH_ATTENTION&&n->has_mask>1)return ROCKET_E_SHAPE;
    state[id]=1;int rc=0;unsigned refs[4];unsigned nr=operands(n,refs);
    for(unsigned i=0;i<nr;i++){rc=visit(d,refs[i],state,p);if(rc)return rc;}
    state[id]=2;p->order[p->steps++]=id;return 0;
}
static int tensor_size(unsigned rows,unsigned cols,size_t *bytes,unsigned *tiles) {
    if(!rows||rows>CO_MAX_M||!cols||cols>CO_MAX_N)return ROCKET_E_SHAPE;
    *bytes=0;*tiles=blocks(rows)*blocks(cols);
    for(unsigned mi=0;mi<blocks(rows);mi++)for(unsigned ni=0;ni<blocks(cols);ni++) {
        co_tensor t;int rc=co_tensor_matrix(&t,extent(rows,mi),extent(cols,ni),CO_F16);
        if(rc)return rc;
        *bytes+=t.bytes;
    }
    return 0;
}
int co_graph_plan_many(const co_graph_desc *d,const unsigned *outputs,unsigned noutputs,co_graph_plan *out) {
    if(!out)return ROCKET_E_SHAPE;
    memset(out,0,sizeof(*out));
    if(!d||!d->nodes||!d->count||d->count>CO_GRAPH_MAX_NODES||!outputs||!noutputs||noutputs>CO_GRAPH_MAX_NODES)
        return ROCKET_E_SHAPE;
    co_graph_plan p={0};unsigned char state[CO_GRAPH_MAX_NODES]={0};
    int rc=0;p.noutputs=noutputs;
    for(unsigned i=0;i<noutputs;i++) {
        if(outputs[i]>=d->count)return ROCKET_E_SHAPE;
        for(unsigned j=0;j<i;j++)if(outputs[j]==outputs[i])return ROCKET_E_SHAPE;
        p.outputs[i]=outputs[i];rc=visit(d,outputs[i],state,&p);if(rc)return rc;
    }
    for(unsigned step=0;step<p.steps;step++) {
        unsigned id=p.order[step];const co_graph_node *node=&d->nodes[id];
        co_graph_tensor_plan *t=&p.tensors[id];
        t->first=step;t->last=step;
        if(node->op==CO_GRAPH_INPUT) {t->rows=node->rows;t->cols=node->cols;p.inputs++;}
        else {
            co_graph_tensor_plan *a=&p.tensors[node->a];t->rows=a->rows;t->cols=a->cols;
            if(node->rows)return ROCKET_E_SHAPE;
            if(node->op==CO_GRAPH_LINEAR) {
                if(!node->weights)return ROCKET_E_SHAPE;
                t->cols=node->cols;co_linear_plan lp;
                rc=co_linear_plan_init_ex(&lp,t->rows,a->cols,t->cols,node->bias!=NULL,CO_LINEAR_NONE,SIZE_MAX);
                if(rc)return rc;
                p.linear_bytes+=lp.device_bytes;p.tasks+=lp.tasks;p.chain_submits+=lp.chain_submits;
            } else {
                if(node->cols)return ROCKET_E_SHAPE;
                if(node->op==CO_GRAPH_RMSNORM&&(!node->gamma||!isfinite(node->epsilon)||node->epsilon<=0))
                    return ROCKET_E_SHAPE;
                if(binary(node->op)) {
                    co_graph_tensor_plan *b=&p.tensors[node->b];
                    if(a->rows!=b->rows||a->cols!=b->cols)return ROCKET_E_SHAPE;
                }
            }
            unsigned refs[4];unsigned nr=operands(node,refs);
            for(unsigned i=0;i<nr;i++)p.tensors[refs[i]].last=step;
        }
        rc=tensor_size(t->rows,t->cols,&t->bytes,&t->tiles);if(rc)return rc;
        if(node->op==CO_GRAPH_ROPE) {
            co_rope_config cfg=rope_config(node,t->rows);co_rope_plan rp;
            rc=co_rope_plan_init(&cfg,&rp);if(rc)return rc;
            if(rp.cols!=t->cols||node->position_base>UINT32_MAX-(t->rows-1))return ROCKET_E_SHAPE;
            p.rope_bytes+=rp.device_bytes;p.hybrid_nodes++;p.tasks+=rp.tasks;p.chain_submits+=rp.chain_submits;
            p.host_upload_calls+=rp.upload_calls;p.host_uploads+=rp.host_upload_elements;
            p.host_activation_reads+=rp.host_activation_reads;
        } else if(node->op==CO_GRAPH_ATTENTION) {
            co_attention_config cfg=attention_config(node,t->rows);co_attention_plan ap;
            rc=co_attention_plan_init(&cfg,&ap);if(rc)return rc;
            co_graph_tensor_plan *k=&p.tensors[node->b],*v=&p.tensors[node->c];
            if(t->cols!=cfg.heads*cfg.head_dim||k->rows!=t->rows||v->rows!=t->rows||
               k->cols!=cfg.kv_heads*cfg.head_dim||v->cols!=k->cols||
               node->position_base>UINT32_MAX-(2*t->rows-2))return ROCKET_E_SHAPE;
            if(node->has_mask) {
                co_graph_tensor_plan *mask=&p.tensors[node->mask_node];
                if(mask->rows!=t->rows||mask->cols!=t->rows)return ROCKET_E_SHAPE;
                p.activation_download_calls+=mask->tiles;
                p.activation_downloads+=(size_t)mask->rows*mask->cols;
            }
            size_t sources=(size_t)t->rows*(t->cols+k->cols+v->cols);
            unsigned long long stats_calls=(unsigned long long)cfg.heads*co_blocks(cfg.max_query)*co_blocks(cfg.capacity);
            p.attention_bytes+=ap.device_bytes;p.hybrid_nodes++;p.tasks+=ap.tasks_per_step;p.chain_submits+=ap.chain_submits_per_step;
            p.host_upload_calls+=ap.upload_calls+t->tiles;
            p.statistic_download_calls+=stats_calls;
            p.activation_download_calls+=ap.download_calls-stats_calls+p.tensors[node->a].tiles+k->tiles+v->tiles;
            p.host_uploads+=ap.host_upload_elements+(size_t)t->rows*t->cols;
            p.host_activation_reads+=ap.host_activation_reads;
            p.statistic_downloads+=ap.statistic_downloads;p.activation_downloads+=ap.activation_downloads+sources;
        } else if(rowop(node->op)) {
            co_row_plan rp;
            rc=co_row_plan_init(&rp,node->op==CO_GRAPH_RMSNORM?CO_ROW_RMSNORM:CO_ROW_SOFTMAX,t->rows,t->cols);
            if(rc)return rc;
            p.rowop_bytes+=rp.device_bytes;p.hybrid_nodes++;
            p.tasks+=rp.tasks;p.chain_submits+=rp.chain_submits;
            p.host_upload_calls+=(unsigned long long)rp.tiles*(node->op==CO_GRAPH_SOFTMAX?3:1);
            p.statistic_download_calls+=rp.tiles;
            p.host_activation_reads+=rp.host_activation_reads;
            p.statistic_downloads+=rp.statistic_downloads;p.host_uploads+=rp.host_uploads;
        } else if(node->op!=CO_GRAPH_INPUT&&node->op!=CO_GRAPH_LINEAR) {
            p.chain_submits+=t->tiles;
            for(unsigned mi=0;mi<blocks(t->rows);mi++)for(unsigned ni=0;ni<blocks(t->cols);ni++) {
                co_tensor tile;
                rc=co_tensor_matrix(&tile,extent(t->rows,mi),extent(t->cols,ni),CO_F16);if(rc)return rc;
                if(binary(node->op)||gated(node->op))p.identity_mask|=1u<<(tile.physical_cols/32-1);
                if(gated(node->op)&&tile.bytes>p.gate_bytes)p.gate_bytes=tile.bytes;
                p.tasks+=binary(node->op)?1:(tile.bytes/2+32767)/32768+(gated(node->op)?1:0);
            }
        }
    }
    for(unsigned step=0;step<p.steps;step++) {
        const co_graph_node *n=&d->nodes[p.order[step]];
        unsigned refs[4];unsigned nr=operands(n,refs);
        for(unsigned i=0;i<nr;i++)if(d->nodes[refs[i]].flags&&
            !((n->op==CO_GRAPH_SOFTMAX&&i==0)||(n->op==CO_GRAPH_ATTENTION&&n->has_mask&&i==3)))return ROCKET_E_SHAPE;
    }
    for(unsigned i=0;i<noutputs;i++) {
        if(d->nodes[outputs[i]].flags)return ROCKET_E_SHAPE;
        p.tensors[outputs[i]].last=p.steps;
    }
    /* Strictly earlier last-use is required: never overwrite a live operand. */
    unsigned slot_end[CO_GRAPH_MAX_NODES]={0};
    for(unsigned step=0;step<p.steps;step++) {
        co_graph_tensor_plan *t=&p.tensors[p.order[step]];
        unsigned chosen=CO_GRAPH_MAX_NODES;size_t best=SIZE_MAX;
        for(unsigned s=0;s<p.slots;s++)if(slot_end[s]<step) {
            size_t cost=p.slot_bytes[s]>t->bytes?p.slot_bytes[s]:t->bytes;
            if(cost<best){best=cost;chosen=s;}
        }
        if(chosen==CO_GRAPH_MAX_NODES)chosen=p.slots++;
        t->slot=chosen;slot_end[chosen]=t->last;
        if(t->bytes>p.slot_bytes[chosen])p.slot_bytes[chosen]=t->bytes;
    }
    for(unsigned s=0;s<p.slots;s++)p.tensor_bytes+=p.slot_bytes[s]+CO_SLACK+CO_GUARD;
    for(unsigned i=0;i<8;i++)if(p.identity_mask&(1u<<i)) {
        co_tensor t;rc=co_tensor_weight(&t,(i+1)*32,(i+1)*32);if(rc)return rc;
        p.workspace_bytes+=t.bytes+CO_SLACK+CO_GUARD;
    }
    if(p.gate_bytes)p.workspace_bytes+=p.gate_bytes+CO_SLACK+CO_GUARD;
    p.device_bytes=p.tensor_bytes+p.linear_bytes+p.rowop_bytes+p.rope_bytes+p.attention_bytes+p.workspace_bytes;
    if(p.device_bytes>(d->device_budget?d->device_budget:CO_LINEAR_DEVICE_BUDGET))return ROCKET_E_TILING;
    *out=p;return 0;
}

int co_graph_plan_init(const co_graph_desc *d,co_graph_plan *out) {
    return co_graph_plan_many(d,d?&d->output:NULL,1,out);
}

struct co_graph {
    int fd;enum co_mode mode;unsigned count,output;
    co_graph_node nodes[CO_GRAPH_MAX_NODES];co_graph_plan plan;
    co_buffer slots[CO_GRAPH_MAX_NODES],identities[8],gate;
    co_view ids[8],*views[CO_GRAPH_MAX_NODES];
    co_linear *linear[CO_GRAPH_MAX_NODES];
    co_rowop *rowops[CO_GRAPH_MAX_NODES];
    co_rope *ropes[CO_GRAPH_MAX_NODES];co_attention *attention[CO_GRAPH_MAX_NODES];
    _Float16 *q[CO_GRAPH_MAX_NODES],*k[CO_GRAPH_MAX_NODES],*v[CO_GRAPH_MAX_NODES],
        *mask[CO_GRAPH_MAX_NODES],*attout[CO_GRAPH_MAX_NODES];
    _Float16 *boundary,*result;
};
void co_graph_destroy(co_graph *g) {
    if(!g)return;
    for(unsigned i=0;i<CO_GRAPH_MAX_NODES;i++) {
        co_rope_destroy(g->ropes[i]);co_attention_destroy(g->attention[i]);
        free(g->q[i]);free(g->k[i]);free(g->v[i]);free(g->mask[i]);free(g->attout[i]);
        co_rowop_destroy(g->rowops[i]);co_linear_destroy(g->linear[i]);free(g->views[i]);co_buffer_free(&g->slots[i]);
    }
    for(unsigned i=0;i<8;i++)co_buffer_free(&g->identities[i]);
    co_buffer_free(&g->gate);free(g->boundary);free(g->result);free(g);
}
int co_graph_compile_many(int fd,const co_graph_desc *d,const unsigned *outputs,unsigned noutputs,enum co_mode mode,co_graph **out) {
    if(!out)return ROCKET_E_SHAPE;
    *out=NULL;
    if(fd<0||(mode!=CO_CHAIN&&mode!=CO_SEPARATE))return ROCKET_E_SHAPE;
    co_graph_plan plan;int rc=co_graph_plan_many(d,outputs,noutputs,&plan);if(rc)return rc;
    if(mode==CO_CHAIN&&!rocket_batched_submit_supported())return ROCKET_E_UNSUPPORTED;
    co_graph *g=calloc(1,sizeof(*g));if(!g)return ROCKET_E_NOMEM;
    g->fd=fd;g->mode=mode;g->count=d->count;g->output=plan.outputs[0];g->plan=plan;
    memcpy(g->nodes,d->nodes,d->count*sizeof(*d->nodes));
    g->boundary=malloc(256u*256u*2);
    size_t result_elements=0;
    for(unsigned i=0;i<plan.noutputs;i++) {
        co_graph_tensor_plan t=plan.tensors[plan.outputs[i]];
        result_elements+=(size_t)t.rows*t.cols;
    }
    g->result=malloc(result_elements*2);
    if(!g->boundary||!g->result){rc=ROCKET_E_NOMEM;goto fail;}
    for(unsigned s=0;s<plan.slots;s++) {
        rc=co_buffer_alloc(fd,plan.slot_bytes[s],&g->slots[s]);if(rc)goto fail;
    }
    for(unsigned i=0;i<8;i++)if(plan.identity_mask&(1u<<i)) {
        co_tensor t;rc=co_tensor_weight(&t,(i+1)*32,(i+1)*32);
        if(!rc)rc=co_buffer_alloc(fd,t.bytes,&g->identities[i]);
        if(!rc)rc=co_view_init(&g->ids[i],&g->identities[i],0,&t);
        if(!rc)rc=co_identity_upload(&g->ids[i]);
        if(rc)goto fail;
    }
    if(plan.gate_bytes){rc=co_buffer_alloc(fd,plan.gate_bytes,&g->gate);if(rc)goto fail;}
    for(unsigned step=0;step<plan.steps;step++) {
        unsigned id=plan.order[step];co_graph_tensor_plan *tp=&g->plan.tensors[id];
        g->views[id]=calloc(tp->tiles,sizeof(co_view));if(!g->views[id]){rc=ROCKET_E_NOMEM;goto fail;}
        unsigned index=0;size_t offset=0;
        for(unsigned mi=0;mi<blocks(tp->rows);mi++)for(unsigned ni=0;ni<blocks(tp->cols);ni++) {
            co_tensor t;rc=co_tensor_matrix(&t,extent(tp->rows,mi),extent(tp->cols,ni),CO_F16);
            if(!rc)rc=co_view_init(&g->views[id][index++],&g->slots[tp->slot],offset,&t);
            if(rc)goto fail;
            offset+=t.bytes;
        }
        co_graph_node *node=&g->nodes[id];
        if(node->op==CO_GRAPH_LINEAR) {
            rc=co_linear_create(fd,tp->rows,plan.tensors[node->a].cols,tp->cols,node->weights,node->bias,
                                CO_LINEAR_NONE,mode,&g->linear[id]);if(rc)goto fail;
        }
        if(rowop(node->op)) {
            rc=co_rowop_create(fd,node->op==CO_GRAPH_RMSNORM?CO_ROW_RMSNORM:CO_ROW_SOFTMAX,
                tp->rows,tp->cols,node->gamma,node->epsilon,mode,&g->rowops[id]);if(rc)goto fail;
        }
        if(node->op==CO_GRAPH_ROPE) {
            co_rope_config cfg=rope_config(node,tp->rows);rc=co_rope_create(fd,&cfg,mode,&g->ropes[id]);if(rc)goto fail;
        }
        if(node->op==CO_GRAPH_ATTENTION) {
            size_t qb=(size_t)tp->rows*tp->cols*2;
            size_t kb=(size_t)tp->rows*plan.tensors[node->b].cols*2;
            g->q[id]=malloc(qb);g->k[id]=malloc(kb);g->v[id]=malloc(kb);g->attout[id]=malloc(qb);
            if(node->has_mask)g->mask[id]=malloc((size_t)tp->rows*tp->rows*2);
            if(!g->q[id]||!g->k[id]||!g->v[id]||!g->attout[id]||(node->has_mask&&!g->mask[id])){rc=ROCKET_E_NOMEM;goto fail;}
            co_attention_config cfg=attention_config(node,tp->rows);
            rc=co_attention_create(fd,&cfg,mode,&g->attention[id]);if(rc)goto fail;
        }
        node->weights=NULL;node->bias=NULL;node->gamma=NULL;
    }
    *out=g;return 0;
fail:co_graph_destroy(g);return rc;
}

int co_graph_compile(int fd,const co_graph_desc *d,enum co_mode mode,co_graph **out) {
    return co_graph_compile_many(fd,d,d?&d->output:NULL,1,mode,out);
}

static int emit_lut_chunks(co_program *p,unsigned *count,co_view x,co_view y,enum co_act act) {
    for(size_t base=0;base<x.tensor.bytes;base+=65536) {
        size_t bytes=x.tensor.bytes-base;if(bytes>65536)bytes=65536;
        co_tensor t;co_view a,b;int rc=co_tensor_linear(&t,(unsigned)(bytes/2));
        if(!rc)rc=co_view_init(&a,x.buffer,x.offset+base,&t);
        if(!rc)rc=co_view_init(&b,y.buffer,y.offset+base,&t);
        if(!rc)rc=co_emit_lut(&p[(*count)++],a,b,act);
        if(rc)return rc;
    }
    return 0;
}
static int execute_node(co_graph *g,unsigned id) {
    co_graph_node node=g->nodes[id];co_graph_tensor_plan *t=&g->plan.tensors[id];
    if(node.op==CO_GRAPH_ROPE)return co_rope_run(g->ropes[id],g->views[node.a],t->tiles,
                                                g->views[id],t->tiles,node.position_base);
    if(node.op==CO_GRAPH_ATTENTION) {
        unsigned kw=g->plan.tensors[node.b].cols;
        int rc=co_tiles_read(g->views[node.a],t->rows,t->cols,g->q[id],g->boundary);
        if(!rc)rc=co_tiles_read(g->views[node.b],t->rows,kw,g->k[id],g->boundary);
        if(!rc)rc=co_tiles_read(g->views[node.c],t->rows,kw,g->v[id],g->boundary);
        if(!rc&&node.has_mask)rc=co_tiles_read(g->views[node.mask_node],t->rows,t->rows,g->mask[id],g->boundary);
        if(!rc)rc=co_attention_reset(g->attention[id],node.position_base);
        if(!rc)rc=co_attention_step(g->attention[id],t->rows,g->q[id],g->k[id],g->v[id],g->mask[id],g->attout[id]);
        if(!rc)rc=co_tiles_write(g->views[id],t->rows,t->cols,g->attout[id],g->boundary);
        return rc;
    }
    if(rowop(node.op))return co_rowop_run(g->rowops[id],g->views[node.a],t->tiles,g->views[id],t->tiles);
    if(node.op==CO_GRAPH_LINEAR)
        return co_linear_run_tiled(g->linear[id],g->views[node.a],g->plan.tensors[node.a].tiles,
                                   g->views[id],t->tiles);
    for(unsigned i=0;i<t->tiles;i++) {
        co_view x=g->views[node.a][i],y=g->views[id][i];co_program p[3];unsigned count=0;int rc=0;
        co_view identity=g->ids[y.tensor.physical_cols/32-1];
        if(binary(node.op))rc=co_emit_ew(&p[count++],x,g->views[node.b][i],identity,y,node.op==CO_GRAPH_MUL);
        else if(gated(node.op)) {
            co_view gate;rc=co_view_init(&gate,&g->gate,0,&x.tensor);
            if(!rc)rc=emit_lut_chunks(p,&count,x,gate,node.op==CO_GRAPH_SILU?CO_SIGMOID:CO_GELU_GATE);
            if(!rc)rc=co_emit_ew(&p[count++],x,gate,identity,y,1);
        } else rc=emit_lut_chunks(p,&count,x,y,CO_HARDSIGMOID);
        if(!rc)rc=co_execute(g->fd,p,count,g->mode);
        if(rc)return rc;
    }
    return 0;
}
static int finite_vector(const _Float16 *x,size_t n,int allow_neg_inf) {
    if(!x)return 0;
    for(size_t i=0;i<n;i++)if(!isfinite((double)x[i])&&!(allow_neg_inf&&x[i]==-INFINITY))return 0;
    return 1;
}
int co_graph_run_many(co_graph *g,const co_graph_input *inputs,unsigned count,
                      const co_graph_output *outputs,unsigned noutputs) {
    if(!g||!inputs||!outputs||count!=g->plan.inputs||noutputs!=g->plan.noutputs)return ROCKET_E_SHAPE;
    _Float16 *targets[CO_GRAPH_MAX_NODES]={0};
    for(unsigned i=0;i<noutputs;i++) {
        unsigned id=outputs[i].node;int found=0;
        for(unsigned j=0;j<g->plan.noutputs;j++)if(id==g->plan.outputs[j])found=1;
        if(!found||!outputs[i].data||targets[id])return ROCKET_E_SHAPE;
        size_t bytes=(size_t)g->plan.tensors[id].rows*g->plan.tensors[id].cols*2;
        uintptr_t a=(uintptr_t)outputs[i].data;
        if(a>UINTPTR_MAX-bytes)return ROCKET_E_SHAPE;
        for(unsigned j=0;j<i;j++) {
            co_graph_tensor_plan t=g->plan.tensors[outputs[j].node];
            size_t bbytes=(size_t)t.rows*t.cols*2;uintptr_t b=(uintptr_t)outputs[j].data;
            if(a<b+bbytes&&b<a+bytes)return ROCKET_E_SHAPE;
        }
        targets[id]=outputs[i].data;
    }
    const _Float16 *bindings[CO_GRAPH_MAX_NODES]={0};
    for(unsigned i=0;i<count;i++) {
        unsigned id=inputs[i].node;
        if(id>=g->count||!g->views[id]||g->nodes[id].op!=CO_GRAPH_INPUT||bindings[id]||
           !finite_vector(inputs[i].data,(size_t)g->plan.tensors[id].rows*g->plan.tensors[id].cols,g->nodes[id].flags!=0))
            return ROCKET_E_SHAPE;
        bindings[id]=inputs[i].data;
    }
    for(unsigned step=0;step<g->plan.steps;step++) {
        unsigned id=g->plan.order[step];co_graph_tensor_plan *t=&g->plan.tensors[id];int rc=0;
        if(g->nodes[id].op==CO_GRAPH_INPUT) {
            unsigned index=0;
            for(unsigned mi=0;mi<blocks(t->rows);mi++)for(unsigned ni=0;ni<blocks(t->cols);ni++) {
                co_view *v=&g->views[id][index++];
                for(unsigned r=0;r<v->tensor.rows;r++)
                    memcpy(g->boundary+(size_t)r*v->tensor.cols,
                        bindings[id]+((size_t)mi*256+r)*t->cols+ni*256,(size_t)v->tensor.cols*2);
                rc=co_upload_f16(v,g->boundary);if(rc)return rc;
            }
        } else {rc=execute_node(g,id);if(rc)return rc;}
    }
    for(unsigned s=0;s<g->plan.slots;s++){int rc=co_guard_check(&g->slots[s]);if(rc)return rc;}
    for(unsigned i=0;i<8;i++)if(g->identities[i].bo.handle) {
        int rc=co_guard_check(&g->identities[i]);if(rc)return rc;
    }
    if(g->gate.bo.handle){int rc=co_guard_check(&g->gate);if(rc)return rc;}
    size_t offset=0;
    for(unsigned i=0;i<g->plan.noutputs;i++) {
        unsigned id=g->plan.outputs[i];co_graph_tensor_plan t=g->plan.tensors[id];
        int rc=co_tiles_read(g->views[id],t.rows,t.cols,g->result+offset,g->boundary);if(rc)return rc;
        size_t elements=(size_t)t.rows*t.cols;
        if(!finite_vector(g->result+offset,elements,0))return ROCKET_E_DEVICE;
        offset+=elements;
    }
    offset=0;
    for(unsigned i=0;i<g->plan.noutputs;i++) {
        unsigned id=g->plan.outputs[i];co_graph_tensor_plan t=g->plan.tensors[id];
        size_t elements=(size_t)t.rows*t.cols;
        memcpy(targets[id],g->result+offset,elements*2);offset+=elements;
    }
    return 0;
}
int co_graph_run(co_graph *g,const co_graph_input *inputs,unsigned count,_Float16 *out) {
    if(!g||g->plan.noutputs!=1)return ROCKET_E_SHAPE;
    co_graph_output output={g->plan.outputs[0],out};return co_graph_run_many(g,inputs,count,&output,1);
}
