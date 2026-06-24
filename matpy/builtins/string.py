"""String processing built-in functions for MatPy."""

import re
import numpy as np
from matpy.builtins import register
from matpy.runtime.types import Mat


@register("strcmp")
def _strcmp(s1, s2):
    return str(s1) == str(s2)

@register("strcmpi")
def _strcmpi(s1, s2):
    return str(s1).lower() == str(s2).lower()

@register("strncmp")
def _strncmp(s1, s2, n):
    return str(s1)[:int(n)] == str(s2)[:int(n)]

@register("strcat")
def _strcat(*args):
    return "".join(str(a) for a in args)

@register("strfind")
def _strfind(s, pattern):
    s = str(s)
    pattern = str(pattern)
    indices = []
    start = 0
    while True:
        idx = s.find(pattern, start)
        if idx == -1:
            break
        indices.append(idx + 1)
        start = idx + 1
    if indices:
        return Mat(np.array(indices))
    return Mat(np.array([]))

@register("strrep")
def _strrep(s, old, new):
    return str(s).replace(str(old), str(new))

@register("strsplit")
def _strsplit(s, delimiter=None):
    s = str(s)
    if delimiter is None:
        parts = s.split()
    else:
        parts = s.split(str(delimiter))
    return parts

@register("strjoin")
def _strjoin(parts, delimiter=" "):
    if isinstance(parts, list):
        return str(delimiter).join(str(p) for p in parts)
    return str(parts)

@register("upper")
def _upper(s):
    return str(s).upper()

@register("lower")
def _lower(s):
    return str(s).lower()

@register("strtrim")
def _strtrim(s):
    return str(s).strip()

@register("contains")
def _contains(s, pattern):
    return str(pattern) in str(s)

@register("startsWith")
def _startsWith(s, prefix):
    return str(s).startswith(str(prefix))

@register("endsWith")
def _endsWith(s, suffix):
    return str(s).endswith(str(suffix))

@register("replace")
def _replace(s, old, new):
    return str(s).replace(str(old), str(new))

@register("reverse")
def _reverse(s):
    return str(s)[::-1]

@register("regexp")
def _regexp(s, pattern, *args):
    s = str(s)
    pattern = str(pattern)
    match = re.search(pattern, s)
    if match:
        if len(args) > 0:
            group = int(args[0])
            if group <= len(match.groups()):
                return match.group(group)
        return match.start() + 1
    return Mat(np.array([]))

@register("regexprep")
def _regexprep(s, pattern, replacement):
    return re.sub(str(pattern), str(replacement), str(s))

@register("sscanf")
def _sscanf(s, fmt):
    s = str(s)
    if '%d' in fmt:
        try:
            return int(s.strip())
        except ValueError:
            return 0
    elif '%f' in fmt:
        try:
            return float(s.strip())
        except ValueError:
            return 0.0
    return s.strip()
