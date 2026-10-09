#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
probe_log=${IC1_PROBE_LOG:-"ic1_cvt_sweep_$(date +%Y%m%d_%H%M%S).log"}

gcc -O2 -o min_conv_ic1_probe min_conv_exp.c -lm

{
    echo "# ic_real=1 CNA input CVT probe"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true
    for sample in 0x00:-128 0x01:-127 0x7f:-1 0x80:0 0xff:127; do
        raw_byte=${sample%%:*}
        expected=${sample#*:}
        echo
        echo "### raw input byte ${raw_byte}; hypothesis output=${expected}"
        probe_output=$(./min_conv_ic1_probe \
            --ic 1 --oc 16 --iw 8 --ih 8 --ow 1 --oh 1 \
            --kw 1 --kh 1 --pad 0 --in-byte "$raw_byte" \
            --w-byte 1 --out-zp 128 --w-oc2 16 --w-kernels 16)
        printf '%s\n' "$probe_output"

        channel_values=$(printf '%s\n' "$probe_output" |
            sed -n '/^ch00: /,/^ch01:/p' |
            sed '$d' |
            sed '1s/^ch00: //' |
            grep -oE -- '-?[0-9]+' || true)
        value_count=$(printf '%s\n' "$channel_values" | awk 'NF { n++ } END { print n+0 }')
        mismatches=$(printf '%s\n' "$channel_values" |
            awk -v expected="$expected" 'NF && $1 != expected { n++ } END { print n+0 }')
        if [[ $value_count -eq 1 && $mismatches -eq 0 ]]; then
            echo "CVT_HYPOTHESIS_MATCH byte=${raw_byte} output=${expected} count=${value_count}"
        else
            echo "CVT_HYPOTHESIS_DIFF byte=${raw_byte} expected=${expected} count=${value_count} mismatches=${mismatches}"
        fi
    done
} | tee "$probe_log"

echo "Saved results to ${probe_log}"
