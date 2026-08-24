from __future__ import annotations
import argparse,json
from datetime import datetime
from pathlib import Path
from ..ops.deploy import set_deployment_mode,verify_v2_baseline
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('NORMAL','SHADOW_ONLY'));p.add_argument('--root',type=Path,required=True);p.add_argument('--timestamp',default=datetime.now().astimezone().isoformat());p.add_argument('--repo-root',type=Path);p.add_argument('--verify-v2',action='store_true');a=p.parse_args()
 if a.verify_v2:
  if a.repo_root is None:p.error('--repo-root is required with --verify-v2')
  verify_v2_baseline(a.repo_root,a.root,a.timestamp)
 print(json.dumps(set_deployment_mode(a.root,a.mode,a.timestamp),ensure_ascii=False,sort_keys=True,indent=2))
if __name__=='__main__':main()
