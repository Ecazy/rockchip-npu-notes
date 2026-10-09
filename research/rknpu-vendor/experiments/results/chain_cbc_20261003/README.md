# §7.184: vendor conv-chain ground truth (conv_bn_conv_fp16, full-window recapture)

Recaptured with the CURRENT hook (full fetch windows, 8 B entries):
`rknn_runner conv_bn_conv_fp16.rknn` + `EPERF_RKNPU_REGCMD_MAX_BYTES=8192`
-> `cbc_fp16.jsonl` (on board), archived here.

## Structure (SUBMIT seq1)
- 1 submit, flags=5, task_number=6, hw=20708 ns, task_counter=6 (ALL SIX executed)
- subcore_tasks = [[0,2],[0,2],[0,2],[0,0],[0,0]]  (3 cores x 2-task chains)
- descriptors: op_idx=[1,2,1,2,2,1], enable_mask=0x1d (PPU task4: 0x60),
  int_mask=0x300 (t4: 0xc00), int_clear=0x1ffff, regcfg_amount=126/126/122/122/26/122,
  regcmd_addr = 0xffffa940,0xffffad80,0xffffb1c0,0xffffb5c0,0xffffb9c0,0xffffbac0
  (stride = ALIGN_UP(fetch_entries,8) entries = 64 B aligned blocks, contiguous)

## Tail truth (conv chain, DIFFERENT from attn!)
- task0 (conv, amount=126, fetch=130): tail = [PC_BASE=next(tgt 0x101), AMOUNTS=0x40(tgt 0x101),
  VERSION=0(tgt 0x41), **PC_OPERATION_ENABLE=0x1d**(tgt 0x81)]
- task1 (LAST of its chain): tail = [pad, PC_REGISTER_AMOUNTS=0(tgt 0x101), VERSION=0(tgt 0x41),
  **OP_EN=0x1d**(tgt 0x81)]  <- terminal OP_EN = 0x1d, NOT 0x18!
- task4 (PPU op): terminal OP_EN = 0x60, enable_mask=0x60, int_mask=0xc00 (third form)

## Conclusion
1. Vendor DOES chain convs (conv->BN->conv = 6 tasks in 1 submit; per-core 2-task chains).
2. Conv-chain tail PC_OPERATION_ENABLE = 0x1d (OP_EN bit0 set) everywhere; the attn's
   0x18 = the DPU-light special case. Our --chain-op-en 0x1d knob = CORRECT.
3. Descriptor enable_mask for conv tasks = 0x1d (matches tail value).
4. Our emitter's stream (ourchain2.jsonl) = structurally IDENTICAL (tail form/values match);
   remaining diff = 31 config values, of which geometry/precision = expected;
   control candidates: 0x100c(=0x6000a120 vs 0), 0x1180=CVT per-ch en (0xfff vs 0),
   0x3010 CORE_MISC_CFG (bit9 PROC_PRECISION=2 vs our bit0 QD_EN).
5. STILL STALLING at task_counter=1 in our emitter with vendor-identical tails/descriptors:
   next step = replay the vendor's own conv blocks through vendor_replay with FULL
   address adaptation (act/wt/dst/src), then bisect the remaining config values.
