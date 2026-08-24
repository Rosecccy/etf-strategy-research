from __future__ import annotations
import csv, os, tempfile
from pathlib import Path
class LedgerConflictError(RuntimeError):pass
def _canon(row):
    out={}
    for k,v in row.items():out[str(k)]='True' if v is True else 'False' if v is False else '' if v is None else str(v)
    return out
class AppendOnlyCsvLedger:
    def __init__(self,path,key_fields):self.path=Path(path);self.key_fields=tuple(key_fields);assert self.key_fields
    def rows(self):
        if not self.path.exists() or self.path.stat().st_size==0:return []
        with self.path.open('r',encoding='utf-8-sig',newline='') as h:return list(csv.DictReader(h))
    def _key(self,row):return tuple(str(row.get(f,'')) for f in self.key_fields)
    def append(self,row):
        c=_canon(row)
        if any(f not in c for f in self.key_fields):raise ValueError(f'missing ledger key fields: {self.key_fields}')
        existing=self.rows();key=self._key(c)
        for old in existing:
            if self._key(old)==key:
                if old==c:return False
                raise LedgerConflictError(f'append-only conflict for key {key}')
        fields=list(existing[0].keys()) if existing else list(c.keys())
        if set(c)!=set(fields):
            if existing:raise LedgerConflictError('ledger schema change is not allowed')
            fields=list(c.keys())
        self.path.parent.mkdir(parents=True,exist_ok=True);all_rows=existing+[c];fd,tmp=tempfile.mkstemp(prefix=self.path.name+'.',suffix='.tmp',dir=str(self.path.parent))
        try:
            with os.fdopen(fd,'w',encoding='utf-8',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(all_rows);h.flush();os.fsync(h.fileno())
            os.replace(tmp,self.path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
        return True
