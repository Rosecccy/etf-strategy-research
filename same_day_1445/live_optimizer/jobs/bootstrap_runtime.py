from __future__ import annotations
import argparse,json
from pathlib import Path
from ..ops.bootstrap import bootstrap_runtime
def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--runtime-root',type=Path,required=True);p.add_argument('--optimizer-root',type=Path,required=True);p.add_argument('--provider',choices=('eastmoney','file'),default='eastmoney');p.add_argument('--file-root',default='');a=p.parse_args();print(json.dumps(bootstrap_runtime(a.source_root,a.runtime_root,a.optimizer_root,a.provider,a.file_root),ensure_ascii=False,sort_keys=True,indent=2))
if __name__=='__main__':main()
