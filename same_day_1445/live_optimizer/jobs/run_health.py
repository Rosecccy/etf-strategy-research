from __future__ import annotations
import argparse,json
from pathlib import Path
from ..ops.health import build_health_report
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--now');a=p.parse_args();r=build_health_report(a.root,a.now);print(json.dumps(r,ensure_ascii=False,sort_keys=True,indent=2));raise SystemExit(2 if r['status']=='BLOCKED' else 0)
if __name__=='__main__':main()
