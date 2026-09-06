"""Resolve the manuscript's actual local TeX inputs for audits and releases."""
from pathlib import Path
import re
INPUT = re.compile(r'\\(?:input|include)\s*\{([^}]+)\}')

def _text(path):
    return re.sub(r'(?m)(?<!\\)%.*$', '', Path(path).read_text(encoding='utf-8'))

def tex_sources(path, active=()):
    path=Path(path).resolve()
    if path in active: raise ValueError(f'Cyclic TeX input: {path}')
    yield path
    for match in INPUT.finditer(_text(path)):
        child=path.parent/match.group(1)
        if not child.suffix: child=child.with_suffix('.tex')
        yield from tex_sources(child, (*active,path))

def expanded_tex(path, active=()):
    path=Path(path).resolve()
    if path in active: raise ValueError(f'Cyclic TeX input: {path}')
    def sub(match):
        child=path.parent/match.group(1)
        if not child.suffix: child=child.with_suffix('.tex')
        return expanded_tex(child,(*active,path))
    return INPUT.sub(sub,_text(path))
