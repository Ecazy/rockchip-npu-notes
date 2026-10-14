#define main benchmark_original_main
#include "../examples/bench_llm_i8.cpp"
#undef main
#include "rknn_api.h"
#include "rknn_matmul_api.h"
#include "../src/rk_npu_cpu_kernels.h"
int main() {
    const int loops=std::getenv("VALIDATE_CORRECTNESS")?1:20;
    bool native=std::getenv("VALIDATE_NATIVE")!=nullptr;
    printf("layout=%s\n",native?"native":"normal");
    puts("Official INT8 -> INT32, core_mask=1, normal A/C, B prepacked; memcpy A + blocking run + memcpy C; median of 20 after 3 warmups");
    for(const auto& p:kProjections) {
        std::vector<int8_t> B((size_t)p.K*p.N);
        fill_i8(&B,0x425f4c4c4dull ^ ((uint64_t)p.K<<32) ^ (uint64_t)p.N);
        for(int M:{1,4,128}) {
            rknn_matmul_info info{}; info.M=M;info.K=p.K;info.N=p.N;info.type=RKNN_INT8_MM_INT8_TO_INT32;info.B_layout=RKNN_MM_LAYOUT_NATIVE;info.AC_layout=native?RKNN_MM_LAYOUT_NATIVE:RKNN_MM_LAYOUT_NORM;
            rknn_matmul_io_attr io{};rknn_matmul_ctx ctx=0;
            if(rknn_matmul_create(&ctx,&info,&io)!=0){printf("%s %d %d %d UNSUPPORTED\n",p.name,M,p.K,p.N);continue;}
            if(rknn_matmul_set_core_mask(ctx,RKNN_NPU_CORE_0)!=0)return 2;
            auto a=rknn_create_mem(ctx,io.A.size),b=rknn_create_mem(ctx,io.B.size),c=rknn_create_mem(ctx,io.C.size);
            if(!a||!b||!c)return 3;
            if(rknn_B_normal_layout_to_native_layout(B.data(),b->virt_addr,p.K,p.N,&info)!=0)return 4;
            if(rknn_matmul_set_io_mem(ctx,a,&io.A)||rknn_matmul_set_io_mem(ctx,b,&io.B)||rknn_matmul_set_io_mem(ctx,c,&io.C))return 5;
            std::vector<int8_t> A((size_t)M*p.K);std::vector<int32_t>C((size_t)M*p.N);
            fill_i8(&A,0x415f4c4c4dull^(uint64_t)M^((uint64_t)p.K<<16));
            std::vector<double> samples;
            for(int i=-3;i<loops;++i){
                auto start=std::chrono::steady_clock::now();
                if(native) rknpu2_matmul_open::cpu::i8_pack_a_native_k16_m16(M,p.K,p.K,A.data(),(int8_t*)a->virt_addr,0);
                else memcpy(a->virt_addr,A.data(),A.size());
                if(rknn_matmul_run(ctx)!=0)return 6;
                if(native) rknpu2_matmul_open::cpu::i8_unpack_c_native_n4_m4(M,p.N,c->virt_addr,C.data(),0);
                else memcpy(C.data(),c->virt_addr,C.size()*sizeof(int32_t));
                double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
                if(i>=0)samples.push_back(us);
            }
            if(!verify_samples(M,p.N,p.K,A.data(),B.data(),C.data()))return 7;
            printf("%-9s %4d %6d %6d %9.2f PASS\n",p.name,M,p.K,p.N,median(samples));
            rknn_destroy_mem(ctx,a);rknn_destroy_mem(ctx,b);rknn_destroy_mem(ctx,c);rknn_matmul_destroy(ctx);
        }
    }
}
