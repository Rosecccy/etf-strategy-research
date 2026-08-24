from __future__ import annotations
import json
from datetime import date,datetime
from urllib.parse import urlencode
from urllib.request import Request,urlopen
from .base import MinuteBar
class EastmoneyMinuteProvider:
    name='eastmoney_1m';endpoint='https://push2his.eastmoney.com/api/qt/stock/kline/get'
    @staticmethod
    def secid(symbol):symbol=str(symbol).zfill(6);return f"1.{symbol}" if symbol.startswith(('5','6')) else f"0.{symbol}"
    def fetch(self,symbol,trade_date):
        ds=trade_date.strftime('%Y%m%d');params={'secid':self.secid(symbol),'klt':'1','fqt':'0','beg':ds,'end':ds,'fields1':'f1,f2,f3,f4,f5,f6','fields2':'f51,f52,f53,f54,f55,f56,f57,f58'};req=Request(self.endpoint+'?'+urlencode(params),headers={'User-Agent':'Mozilla/5.0','Referer':'https://quote.eastmoney.com/'})
        with urlopen(req,timeout=15) as response:payload=json.loads(response.read().decode('utf-8'))
        out=[]
        for item in ((payload or {}).get('data') or {}).get('klines') or []:
            parts=str(item).split(',')
            if len(parts)<7:continue
            ts=datetime.fromisoformat(parts[0].replace(' ','T'));out.append(MinuteBar(ts,float(parts[1]),float(parts[3]),float(parts[4]),float(parts[2]),float(parts[5]),float(parts[6])))
        return sorted(out,key=lambda x:x.timestamp)
