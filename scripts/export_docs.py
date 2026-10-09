#!/usr/bin/env python3
"""Build a small, self-contained Obsidian documentation snapshot."""
import argparse,datetime,hashlib,json,pathlib,re,tarfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
SELECTED={
 'docs/CAPABILITIES.md':'01-当前能力总览.md',
 'docs/PROJECT-STRUCTURE.md':'02-项目结构与扩展约定.md',
 'research/MULTICORE-SCHEDULING-PROFILING.md':'03-多核分配与调度Profiling.md',
 'research/FP16-K-TILING.md':'04-FP16-K分片与加速.md',
 'research/TWO-TASK-CHAINING.md':'05-Task链与寄存器组验收.md',
 'research/custom-ops/README.md':'06-自写算子使用指南.md',
 'research/custom-ops/results/2026-10-09/OPERATOR-DELIVERY.md':'07-完整算子交付验收.md',
 'hardware-overview.md':'08-硬件结构参考.md',
}
LINK=re.compile(r'(?<!!)\[([^\]]+)\]\(([^)]+)\)')
def build(out,date):
 out.mkdir(parents=True,exist_ok=False)
 mapped={(ROOT/src).resolve():name for src,name in SELECTED.items()}
 manifest={}
 for src,name in SELECTED.items():
  source=ROOT/src;text=source.read_text();title=text.splitlines()[0].lstrip('# ')
  def rewrite(m):
   label,target=m.groups()
   if target.startswith(('http://','https://','#','mailto:')):return m[0]
   path,sep,anchor=target.partition('#');dest=(source.parent/path).resolve()
   if dest in mapped:return f'[{label}]({mapped[dest]}'+('#'+anchor if sep else '')+')'
   try:relative=dest.relative_to(ROOT).as_posix()
   except ValueError:return label
   note='仓库路径' if dest.exists() else '历史引用，当前工作区缺失'
   return f'{label}（{note}：`{relative}`）'
  text=LINK.sub(rewrite,text)
  # Typographic Greek in prose is a mathematical symbol; code remains verbatim.
  segments=re.split(r'(```[\s\S]*?```|`[^`\n]*`)',text)
  greek={'γ':r'\gamma','β':r'\beta'}
  for i in range(0,len(segments),2):
   for char,latex in greek.items():segments[i]=segments[i].replace(char,'$'+latex+'$')
  text=''.join(segments)
  header=f'---\ntitle: {json.dumps(title,ensure_ascii=False)}\ntags:\n  - project\n  - RKNPU\ncreated: {date}\nupdated: {date}\nsources:\n  - {json.dumps(src)}\n---\n\n'
  text=header+text+f'\n\n> 同步快照：{date}；源文件 `{src}`。源码与大型原始证据保留在开发工作区。\n'
  (out/name).write_text(text)
  manifest[name]={'source':src,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'export_sha256':hashlib.sha256(text.encode()).hexdigest()}
 index=f'---\ntitle: RKNPU 当前能力与项目结构\ntags:\n  - project-index\n  - RKNPU\ncreated: {date}\nupdated: {date}\n---\n\n# RKNPU 当前能力与项目结构\n\n{date} 精选快照。先读能力总览与调度报告，再按开发需求读其余文档。\n\n'
 for name in SELECTED.values():index+=f'- [{name.removesuffix(".md")}]({name})\n'
 index+='\n“本地真机”“历史厂商”“组件/上游”“待补齐”分开判断；驱动服务包络不等于纯硬件执行。源码、构建和原始二进制证据保留在 Linux 开发工作区。\n'
 index+='\n## 版本摘要\n\n| 文档 | 源文件 SHA256 |\n|---|---|\n'
 for name,data in manifest.items():index+=f'| {name} | `{data["source_sha256"]}` |\n'
 (out/'README.md').write_text(index)
 (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 with tarfile.open(out.parent/(out.name+'.tar.gz'),'w:gz') as archive:archive.add(out,arcname=out.name)
 print(out)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('destination',type=pathlib.Path);ap.add_argument('--date',default=datetime.date.today().isoformat());args=ap.parse_args();datetime.date.fromisoformat(args.date);build(args.destination,args.date)
