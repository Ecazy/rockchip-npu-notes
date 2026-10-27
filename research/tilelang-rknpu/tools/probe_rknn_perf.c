#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_api.h"

static unsigned char *read_file(const char *path, size_t *size) {
  FILE *f=fopen(path,"rb"); if(!f) return NULL;
  fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
  unsigned char *p=malloc((size_t)n); if(!p){fclose(f);return NULL;}
  if(fread(p,1,(size_t)n,f)!=(size_t)n){free(p);fclose(f);return NULL;}
  fclose(f); *size=(size_t)n; return p;
}

int main(int argc,char **argv){
  if(argc<3){fprintf(stderr,"usage: %s model.rknn batch\n",argv[0]);return 2;}
  int batch=atoi(argv[2]);
  size_t size=0; unsigned char *model=read_file(argv[1],&size);
  if(!model){perror("model");return 2;}
  rknn_context ctx=0;
  int rc=rknn_init(&ctx,model,(uint32_t)size,RKNN_FLAG_COLLECT_PERF_MASK,NULL);
  if(rc){fprintf(stderr,"rknn_init rc=%d\n",rc);return 1;}
  rknn_input_output_num io;
  rc=rknn_query(ctx,RKNN_QUERY_IN_OUT_NUM,&io,sizeof(io));
  if(rc){fprintf(stderr,"query io rc=%d\n",rc);return 1;}
  printf("n_input=%u n_output=%u\n",io.n_input,io.n_output);
  rknn_tensor_attr in; memset(&in,0,sizeof(in)); in.index=0;
  rc=rknn_query(ctx,RKNN_QUERY_INPUT_ATTR,&in,sizeof(in));
  if(rc){fprintf(stderr,"query in rc=%d\n",rc);return 1;}
  printf("input dims=%u,%u,%u,%u fmt=%u type=%u elems=%u\n",
    in.dims[0],in.dims[1],in.dims[2],in.dims[3],in.fmt,in.type,in.n_elems);
  float *buf=calloc((size_t)batch*784,sizeof(float));
  rknn_input input; memset(&input,0,sizeof(input));
  input.index=0; input.type=RKNN_TENSOR_FLOAT32; input.fmt=RKNN_TENSOR_NHWC;
  input.size=(uint32_t)((size_t)batch*784*sizeof(float)); input.buf=buf;
  rc=rknn_inputs_set(ctx,1,&input);
  if(rc){fprintf(stderr,"inputs_set rc=%d\n",rc);return 1;}
  rc=rknn_run(ctx,NULL);
  if(rc){fprintf(stderr,"run rc=%d\n",rc);return 1;}
  rknn_output *outs=calloc(io.n_output,sizeof(rknn_output));
  for(uint32_t i=0;i<io.n_output;i++){outs[i].index=i;outs[i].want_float=1;}
  rc=rknn_outputs_get(ctx,io.n_output,outs,NULL);
  if(rc){fprintf(stderr,"outputs_get rc=%d\n",rc);return 1;}
  for(uint32_t i=0;i<io.n_output;i++) printf("out[%u] size=%u first=%f\n",i,outs[i].size,((float*)outs[i].buf)[0]);
  rknn_perf_detail detail={0};
  rc=rknn_query(ctx,RKNN_QUERY_PERF_DETAIL,&detail,sizeof(detail));
  if(rc) fprintf(stderr,"perf_detail rc=%d\n",rc);
  else printf("PERF_DETAIL_BEGIN\n%.*s\nPERF_DETAIL_END\n",(int)detail.data_len,detail.perf_data?detail.perf_data:"");
  rknn_perf_run run={0};
  rc=rknn_query(ctx,RKNN_QUERY_PERF_RUN,&run,sizeof(run));
  if(!rc) printf("run_us=%lld\n",(long long)run.run_duration);
  rknn_outputs_release(ctx,io.n_output,outs);
  rknn_destroy(ctx); free(outs);free(buf);free(model);return 0;
}
