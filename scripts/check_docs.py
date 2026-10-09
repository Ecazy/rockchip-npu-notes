#!/usr/bin/env python3
"""Check portable Markdown links; optionally run the vault math check on prose.
The vault checker treats API snake_case as math. Code, URLs and multi-character
technical identifiers are masked in a temporary view; mathematical x_i is kept.
"""
import argparse,pathlib,re,subprocess,tempfile
LINK=re.compile(r'(?<!!)\[[^\]]+\]\(([^)]+)\)')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory',type=pathlib.Path);ap.add_argument('--math-checker',type=pathlib.Path);a=ap.parse_args();files=list(a.directory.glob('*.md'));errors=[]
 for p in files:
  for target in LINK.findall(p.read_text()):
   if target.startswith(('http://','https://','#','mailto:')):continue
   if not (p.parent/target.partition('#')[0]).exists():errors.append(f'{p}: missing {target}')
 if errors:raise SystemExit('\n'.join(errors))
 print(f'LINKS PASS: {len(files)} documents')
 if a.math_checker:
  with tempfile.TemporaryDirectory(prefix='rknpu-prose-check-') as d:
   for p in files:
    text=p.read_text();text=re.sub(r'```[\s\S]*?```','',text);text=re.sub(r'`[^`\n]*`','',text);text=re.sub(r'https?://[^\s)\]>]+','',text)
    text=re.sub(r'\b[A-Za-z][A-Za-z0-9]{1,}_[A-Za-z0-9_]+\b','IDENTIFIER',text)
    (pathlib.Path(d)/p.name).write_text(text)
   r=subprocess.run(['python3',str(a.math_checker),d+'/*.md'],capture_output=True,text=True)
   print(r.stdout.strip())
   if r.returncode or '共 ' in r.stdout or 'OK：' not in r.stdout:raise SystemExit(1)
if __name__=='__main__':main()
