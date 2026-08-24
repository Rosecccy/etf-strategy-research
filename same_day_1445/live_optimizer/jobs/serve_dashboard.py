from __future__ import annotations
import argparse,functools
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8765);a=p.parse_args();handler=functools.partial(SimpleHTTPRequestHandler,directory=str(a.root/'site'));server=ThreadingHTTPServer((a.host,a.port),handler);print(f'http://{a.host}:{a.port}/');server.serve_forever()
if __name__=='__main__':main()
