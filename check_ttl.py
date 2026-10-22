"""Check the requested TTLCache behavior in a short-lived isolated interpreter."""
import importlib.util
import json
import resource
import sys
import traceback
from pathlib import Path
import types

resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
resource.setrlimit(resource.RLIMIT_CPU,(4,4))
try:
    source=Path(sys.argv[1]).read_text()
    wrapper_removed=False
    if '--allow-fenced-module' in sys.argv[2:]:
        lines=source.strip().splitlines()
        if lines and lines[0] in ('```python','```py','```') and lines[-1]=='```':
            source='\n'.join(lines[1:-1])+'\n'
            wrapper_removed=True
    module=types.ModuleType('generated')
    sys.modules['generated']=module
    exec(compile(source,sys.argv[1],'exec'),module.__dict__)
    cls=module.TTLCache
    clock=[0.]
    cache=cls(2,10.,timer=lambda:clock[0])
    cache.put('a',1);cache.put('b',2)
    assert cache.get('a')==1
    cache.put('c',3)
    assert cache.get('b') is None
    assert cache.get('a')==1 and cache.get('c')==3
    stats=cache.stats()
    assert set(stats)=={'size','maxsize','hits','misses','evictions','expired'}
    assert all(type(v)==int for v in stats.values())
    assert stats['size']==2 and stats['maxsize']==2 and stats['evictions']==1
    assert cache.delete('a') is True and cache.delete('a') is False
    clock[0]=10.
    assert cache.delete('c') is False
    clock[0]=0.
    cache=cls(2,10.,timer=lambda:clock[0]);cache.put('x',1)
    clock[0]=9.;assert cache.get('x')==1
    clock[0]=10.;assert cache.get('x') is None
    clock[0]=0.
    cache=cls(2,10.,timer=lambda:clock[0]);cache.put('x',1)
    clock[0]=9.;cache.put('x',2)
    clock[0]=10.;assert cache.get('x')==2
    clock[0]=19.;assert cache.cleanup()==1 and cache.stats()['size']==0
    for size,ttl in [(0,1),(-1,1),(1,-1)]:
        try:cls(size,ttl)
        except Exception:pass
        else:raise AssertionError('Invalid constructor arguments accepted')
    clock[0]=0.
    cache=cls(2,0.,timer=lambda:clock[0]);cache.put('zero',1)
    assert cache.get('zero') is None
    print(json.dumps({'passed':True,'outer_code_fence_removed':wrapper_removed}))
except BaseException as error:
    print(json.dumps({'passed':False,'exception':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()}))
    raise SystemExit(1)
