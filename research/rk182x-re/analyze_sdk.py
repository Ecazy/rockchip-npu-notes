#!/usr/bin/env python3
"""Read SDK artifacts without importing or executing vendor code.

Run from any directory: python3 research/rk182x-re/analyze_sdk.py
Inputs live under refs/; compact, reproducible evidence is written to evidence/.
Capstone is optional and only decodes a short raw firmware startup sequence.
"""
import hashlib
import json
import re
import struct
import subprocess
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "evidence"
OUT.mkdir(exist_ok=True)
SDK = ROOT / "refs/rknn3-toolkit"
BOARD = ROOT / "refs/rk182x-board-v1.1.0"


def save(name, text):
    (OUT / name).write_text(text, encoding="utf-8")


def run(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def strings(data):
    return [(m.start(), m.group().decode("ascii"))
            for m in re.finditer(rb"[\x20-\x7e]{6,}", data)]


def selected_strings(name, data, pattern):
    rx = re.compile(pattern, re.I)
    save(name, "# Hex offsets are file offsets, not runtime addresses.\n" +
         "\n".join(f"0x{off:08x} {s}" for off, s in strings(data) if rx.search(s)) + "\n")


inputs = []


def record(path, **extra):
    data = path.read_bytes()
    inputs.append(dict(path=str(path.relative_to(ROOT)), bytes=len(data),
                       sha256=digest(data), **extra))
    return data


wheel = SDK / "rknn3-toolkit/packages/rknn3_toolkit-1.1.0-cp310-cp310-manylinux2014_x86_64.whl"
record(wheel)
member = "rknn/api/lib/linux-x86_64/rknx/librknnc.so"
with zipfile.ZipFile(wheel) as z:
    compiler = z.read(member)
    inputs.append(dict(archive=str(wheel.relative_to(ROOT)), member=member,
                       bytes=len(compiler), sha256=digest(compiler)))
    toolchain = next(n for n in z.namelist() if "rknrv_toolchain" in n)
    save("wheel-members.txt", "\n".join(
        f"{i.file_size:12d} {i.filename}" for i in z.infolist()
        if any(t in i.filename for t in ["librknnc", "rknrv_toolchain", "rknn_compiler", "rknn/fbs/Task.", "rknn/fbs/MemoryRegion."])) + "\n")

# These are complete NUL-terminated plaintext resources, not decompiled source.
templates = {
    "boot_params.h": b"#ifndef __RKNRV_BOOT_PARAMS_H__",
    "boot_params.c": b'#include "boot_params.h"',
    "exec_params_model.h": b"#ifndef __RKNRV_EXEC_PARAMS_MODEL_H__",
    "rknrv_reg_base.h": b"#ifndef __RKNRV_REG_BASE_H__",
    "rknrv_reg_commit.h": b"#ifndef __RKNRV_REG_COMMIT_H__",
    "model-build.mk": b"RKNRV_TOOLCHAIN_ROOT_PATH ?=",
    "model-linker.lds": b'OUTPUT_ARCH("riscv")',
}
template_meta = []
for name, needle in templates.items():
    start = compiler.index(needle)
    end = compiler.index(b"\0", start)
    data = compiler[start:end]
    save(name + ".txt", data.decode("utf-8"))
    template_meta.append(dict(output=name + ".txt", compiler_file_offset=hex(start),
                              bytes=len(data), sha256=digest(data)))
save("compiler-templates.json", json.dumps(template_meta, indent=2) + "\n")

model_path = ROOT / "refs/models/mobilenet_v2.rknn"
model = record(model_path)
assert model[:4] == b"RKNX"
boot = model.index(b"NNRV")
fields = struct.unpack_from("<24I", model, boot)
assert fields[0] == 0x56524E4E and fields[1] == 2 and fields[5] == 1
entry = fields[2]
assert boot + entry < len(model)
exec_offset = boot + fields[6]
assert struct.unpack_from("<I", model, exec_offset)[0] == 0x45584D44
mmio_hits = []
for off in range(boot + entry, len(model) - 4, 2):
    word = struct.unpack_from("<I", model, off)[0]
    if word & 0x7F == 0x37 and word & 0xFFFFF000 in (
            0xFFF00000, 0xFFF01000, 0xFFF03000, 0xFFF04000, 0xFFF05000):
        mmio_hits.append(off)
save("model-command.json", json.dumps(dict(
    boot_file_offset=hex(boot), boot_words=[hex(x) for x in fields],
    entry_command_offset=hex(entry), entry_file_offset=hex(boot + entry),
    exec_params_file_offset=hex(exec_offset),
    candidate_mmio_lui_count=len(mmio_hits),
    candidate_mmio_lui_file_offsets=[hex(x) for x in mmio_hits]), indent=2) + "\n")
selected_strings("model-versions.txt", model, r"version|toolkit|RKNX|task_sub")
try:
    import capstone
    decoder = capstone.Cs(capstone.CS_ARCH_RISCV,
                         capstone.CS_MODE_RISCV32 | capstone.CS_MODE_RISCVC)
    blocks = [(boot + entry, 128), (mmio_hits[1], 96)]
    listing = []
    for off, size in blocks:
        listing.append(f"\n# File offset {hex(off)}; addresses below relative to command at {hex(boot)}\n")
        for ins in decoder.disasm(model[off:off + size], off - boot):
            listing.append(f"0x{ins.address:08x} {ins.bytes.hex():12s} {ins.mnemonic} {ins.op_str}\n")
    save("model-command.dis.txt", "".join(listing))
except ImportError:
    save("model-command.dis.txt", "Optional capstone required for RV32 disassembly.\n")

for archive in sorted(BOARD.glob("*.tgz")):
    record(archive)
    with tarfile.open(archive) as t:
        save("board-installer-members.txt", "\n".join(
            f"{m.size:12d} {m.name}" for m in t.getmembers()) + "\n")
for path in sorted(BOARD.glob("*.deb")):
    record(path)
    save("board-deb-members.txt", run("dpkg-deb", "-c", str(path)))
for path in sorted(BOARD.glob("*.pdf")):
    record(path)

firmware = BOARD / "installer/system_root/lib/firmware/rknn3_rk1820.img"
fw = record(firmware)
other = BOARD / "x86/usr/lib/firmware/rknn3_rk1820.img"
other_fw = record(other)
save("firmware-variants.json", json.dumps(dict(
    arm_installer_sha256=digest(fw), x86_deb_sha256=digest(other_fw),
    identical=fw == other_fw), indent=2) + "\n")
selected_strings("firmware-x86-versions.txt", other_fw, r"librknn3rt_|version info:")
assert fw[:4] == b"RKFW"
af = fw.index(b"RKAF")
count = struct.unpack_from("<I", fw, af + 136)[0]
assert 0 < count < 32
parts = []
startup = []
for i in range(count):
    pos = af + 140 + 112 * i
    name = fw[pos:pos + 32].split(b"\0")[0].decode()
    filename = fw[pos + 32:pos + 96].split(b"\0")[0].decode()
    offset, flash_sector, blocks, size = struct.unpack_from("<IIII", fw, pos + 96)
    data = fw[af + offset:af + offset + size]
    assert len(data) == size and offset >= 140 + count * 112
    info = dict(name=name, filename=filename, image_file_offset=hex(af + offset),
                size=size, flash_sector=hex(flash_sector), blocks=blocks, sha256=digest(data))
    if data[:4] == b"RESC":
        payload, length, address = struct.unpack_from("<III", data, 0x50)
        assert payload + length <= len(data)
        info.update(payload_offset=hex(payload), payload_length=length,
                    header_address_field=hex(address), csky_offset=hex(data.index(b"CSKY")))
        # The startup branch skips the CSKY header; no full VMA mapping is assumed.
        try:
            import capstone
            decoder = capstone.Cs(capstone.CS_ARCH_RISCV,
                                 capstone.CS_MODE_RISCV64 | capstone.CS_MODE_RISCVC)
            startup.append(f"\n{name}: partition-relative offsets; RV64+C decode\n")
            for ins in decoder.disasm(data[payload + 0x34:payload + 0x64], payload + 0x34):
                startup.append(f"0x{ins.address:04x} {ins.bytes.hex():12s} {ins.mnemonic} {ins.op_str}\n")
        except ImportError:
            startup.append(f"{name}: optional capstone unavailable\n")
    parts.append(info)
save("firmware-partitions.json", json.dumps(dict(rkaf_offset=hex(af), parts=parts), indent=2) + "\n")
save("firmware-startup.txt", "".join(startup))
selected_strings("firmware-strings.txt", fw,
                 r"version info:|librknn3rt_|Loading command|command.*encrypted|loadCommand|"
                 r"command size|command_addr|command_dma_addr|Invalid boot_params|"
                 r"update task|update task_id|submit RKNPU|RKNPU submit|rknpu_tasks_(submit|wait)|"
                 r"Worker.*receive RKNN3_SUB_MSG_(RUN|LOAD_NETWORK|NETWORK_INIT)|"
                 r"command.*align|Updated.*(addr|command)|hardware task|kvcache.*align")

runtime = SDK / "rknn3-runtime"
elfs = {
    "sdk-proxy-arm64": runtime / "rknn3_transfer_proxy/linux-aarch64/rknn3_transfer_proxy",
    "sdk-rkcp-arm64": runtime / "rknn3-api/Linux/aarch64/librknn3_api_rkcp.so",
    "sdk-native-arm64": runtime / "rknn3-api/Linux/aarch64/librknn3_api_native.so",
    "board-proxy-arm64": BOARD / "installer/system_root/bin/rknn3_transfer_proxy",
    "board-proxy-x86": BOARD / "x86/usr/bin/rknn3_transfer_proxy",
    "board-rkcp-x86": BOARD / "x86/usr/lib/librknn3_api_rkcp.so",
}
for name, path in elfs.items():
    data = record(path)
    save(name + "-elf.txt", run("file", str(path)) + run("readelf", "-h", "-d", str(path)))
    syms = run("nm", "-DCS", str(path))
    lines = [s for s in syms.splitlines() if re.search(
        r"TransferClient::(send|recv)|TransferClientProxy::|cc_cmp_pcie|rknn3_run$|"
        r"rknn3_model_init$|rknn3_load_model_from_path$| ioctl@|sendmsg@|recvmsg@", s)]
    save(name + "-symbols.txt", "\n".join(lines) + "\n")
    selected_strings(name + "-strings.txt", data,
                     r"cc_cmp_pcie_(task_create|send_buff|start_dma|get_buff)|"
                     r"PCIe.*(send|buffer|chunk)|pcie-rkep|/dev/rknpu|"
                     r"submodel.*command start|Sending encrypted model|"
                     r"RKNPUEmitter|REGCMD|Failed to submit RKNPU|command_dma_addr")
    if name == "board-rkcp-x86":
        m = re.search(r"^([0-9a-f]+) ([0-9a-f]+) T rknn3_run$", syms, re.M)
        assert m
        start, size = (int(x, 16) for x in m.groups())
        save("host-rknn3_run.dis.txt", run("objdump", "-d", "-C",
             f"--start-address={start}", f"--stop-address={start + size}", str(path)))
    if name == "sdk-proxy-arm64":
        m = re.search(r"^([0-9a-f]+) ([0-9a-f]+) T RK::TransferClientProxy::dispatcher", syms, re.M)
        assert m
        start, size = (int(x, 16) for x in m.groups())
        save("proxy-dispatcher.dis.txt", run("llvm-objdump-16", "-d", "--demangle",
             f"--start-address={start}", f"--stop-address={start + size}", str(path)))

driver = BOARD / "x86/usr/src/pcie-rkep-3.3.1"
for path in sorted(driver.iterdir()):
    if path.is_file():
        record(path)
save("host-driver-index.txt", "\n".join(
    f"{i}: {line}" for i, line in enumerate((driver / "pcie-rkep.c").read_text().splitlines(), 1)
    if re.search(r"DRV_VERSION|case PCIE_|static .*pcie_rkep_(ioctl|mmap)|PCI_VDEVICE", line)) + "\n")
save("input-manifest.json", json.dumps(dict(
    toolkit_commit=run("git", "-C", str(SDK), "rev-parse", "HEAD").strip(),
    artifacts=inputs), indent=2) + "\n")
print(f"Wrote {len(list(OUT.iterdir()))} evidence files to {OUT}")
