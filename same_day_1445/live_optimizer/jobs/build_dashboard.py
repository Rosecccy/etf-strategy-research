from __future__ import annotations
import argparse
from pathlib import Path
from ..ops.dashboard import write_dashboard
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();print(write_dashboard(a.root))
if __name__=='__main__':main()
