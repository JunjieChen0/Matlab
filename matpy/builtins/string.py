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


# ── Additional String Functions ────────────────────────────────

@register("isletter")
def _isletter(s):
    """Check if characters are letters."""
    s = str(s)
    return Mat(np.array([c.isalpha() for c in s]))


@register("isspace")
def _isspace(s):
    """Check if characters are whitespace."""
    s = str(s)
    return Mat(np.array([c.isspace() for c in s]))


@register("isstrprop")
def _isstrprop(s, prop):
    """Check if characters are of specified type."""
    s = str(s)
    prop = str(prop).lower()
    if prop == "alpha":
        return Mat(np.array([c.isalpha() for c in s]))
    elif prop == "digit":
        return Mat(np.array([c.isdigit() for c in s]))
    elif prop == "alphanum":
        return Mat(np.array([c.isalnum() for c in s]))
    elif prop == "lower":
        return Mat(np.array([c.islower() for c in s]))
    elif prop == "upper":
        return Mat(np.array([c.isupper() for c in s]))
    return Mat(np.array([False] * len(s)))


@register("insertBefore")
def _insertBefore(s, pos, newstr):
    """Insert string before position."""
    s = str(s)
    pos = int(pos) - 1  # 1-based to 0-based
    newstr = str(newstr)
    return s[:pos] + newstr + s[pos:]


@register("insertAfter")
def _insertAfter(s, pos, newstr):
    """Insert string after position."""
    s = str(s)
    pos = int(pos)  # 1-based
    newstr = str(newstr)
    return s[:pos] + newstr + s[pos:]


@register("extractBefore")
def _extractBefore(s, pos):
    """Extract substring before position."""
    s = str(s)
    pos = int(pos) - 1  # 1-based to 0-based
    return s[:pos]


@register("extractAfter")
def _extractAfter(s, pos):
    """Extract substring after position."""
    s = str(s)
    pos = int(pos)  # 1-based
    return s[pos:]


@register("extractBetween")
def _extractBetween(s, start, stop):
    """Extract substring between positions."""
    s = str(s)
    start = int(start) - 1  # 1-based to 0-based
    stop = int(stop)
    return s[start:stop]


@register("pad")
def _pad(s, width, side="right", padchar=" "):
    """Pad string."""
    s = str(s)
    width = int(width)
    side = str(side).lower()
    padchar = str(padchar)
    if side == "right":
        return s.ljust(width, padchar)
    elif side == "left":
        return s.rjust(width, padchar)
    else:  # both
        total_pad = width - len(s)
        left_pad = total_pad // 2
        right_pad = total_pad - left_pad
        return padchar * left_pad + s + padchar * right_pad


@register("strip")
def _strip(s, side="both", chars=None):
    """Strip whitespace."""
    s = str(s)
    side = str(side).lower()
    if chars is not None:
        chars = str(chars)
    if side == "left":
        return s.lstrip(chars)
    elif side == "right":
        return s.rstrip(chars)
    return s.strip(chars)


@register("compose")
def _compose(fmt, *args):
    """Format string using sprintf-style formatting."""
    result = str(fmt)
    for a in args:
        if isinstance(a, Mat):
            a = a.to_python()
        result = result.replace('%d', str(int(a)), 1) if '%d' in result else result
        result = result.replace('%f', str(float(a)), 1) if '%f' in result else result
        result = result.replace('%s', str(a), 1) if '%s' in result else result
    return result


@register("double")
def _double(s):
    """Convert to double."""
    if isinstance(s, str):
        return Mat(np.array([ord(c) for c in s], dtype=float))
    return float(s)


@register("native2unicode")
def _native2unicode(bytes_in, charset="utf-8"):
    """Convert bytes to Unicode."""
    if isinstance(bytes_in, Mat):
        bytes_in = bytes_in.data.astype(np.uint8).tobytes()
    return bytes_in.decode(str(charset))


@register("unicode2native")
def _unicode2native(s, charset="utf-8"):
    """Convert Unicode to bytes."""
    bytes_out = str(s).encode(str(charset))
    return Mat(np.frombuffer(bytes_out, dtype=np.uint8))
