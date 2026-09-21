"""Resolve local TeX inputs using the document's compilation directory."""
from pathlib import Path
import re
INPUT = re.compile(r'\\(?:input|include)\s*\{([^}]+)\}')


def _text(path):
    return re.sub(r'(?m)(?<!\\)%.*$', '', Path(path).read_text(encoding='utf-8'))


def _child(name, root):
    path = root / name
    return path if path.suffix else path.with_suffix('.tex')


def tex_sources(path, active=(), root=None):
    path = Path(path).resolve()
    root = Path(root) if root is not None else path.parent
    if path in active:
        raise ValueError(f'Cyclic TeX input: {path}')
    yield path
    for match in INPUT.finditer(_text(path)):
        yield from tex_sources(_child(match.group(1), root), (*active, path), root)


def expanded_tex(path, active=(), root=None):
    path = Path(path).resolve()
    root = Path(root) if root is not None else path.parent
    if path in active:
        raise ValueError(f'Cyclic TeX input: {path}')
    def sub(match):
        return expanded_tex(_child(match.group(1), root), (*active, path), root)
    return INPUT.sub(sub, _text(path))
