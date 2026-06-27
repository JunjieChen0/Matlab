"""Script to remove duplicate @register calls from builtin modules."""

import re
import os

builtin_dir = "matpy/builtins"


def remove_register_and_func(content, func_names):
    """Remove @register('name') + def _name(...): ... from content."""
    lines = content.split("\n")
    result = []
    i = 0
    removed = []
    while i < len(lines):
        line = lines[i]
        match = re.match(r"""@register\(['"]([^'"]+)['"]\)$""", line.strip())
        if match and match.group(1) in func_names:
            func_name = match.group(1)
            i += 1
            if i < len(lines) and re.match(r"\s*def\s+", lines[i]):
                i += 1
                while i < len(lines) and (
                    lines[i].startswith("    ") or lines[i].strip() == ""
                ):
                    if lines[i].strip() and not lines[i].startswith("    "):
                        break
                    i += 1
                removed.append(func_name)
                continue
            else:
                removed.append(func_name)
                continue
        result.append(line)
        i += 1
    return "\n".join(result), removed


def remove_register_only(content, func_names):
    """Remove only @register('name') lines (keep the function def)."""
    lines = content.split("\n")
    result = []
    removed = []
    for line in lines:
        match = re.match(r"""@register\(['"]([^'"]+)['"]\)$""", line.strip())
        if match and match.group(1) in func_names:
            removed.append(match.group(1))
            continue
        result.append(line)
    return "\n".join(result), removed


# Cross-module: remove entire functions from non-owner modules
files_to_remove_funcs = {
    "io.py": ["addpath", "genpath", "rmpath", "path"],
    "math.py": ["eps"],
    "plotting.py": ["pause"],
    "string.py": ["char"],
    "string_array.py": ["isstring"],
}

# Cross-module: remove only @register lines (keep def for internal use)
files_remove_register_only = {
    "advanced_math.py": [
        "anova1",
        "chi2gof",
        "fminbnd",
        "fmincon",
        "fminsearch",
        "fzero",
        "integral",
        "integral2",
        "interp1",
        "interp2",
        "interp3",
        "kron",
        "linprog",
        "mkpp",
        "mode",
        "ode15s",
        "ode23",
        "ode45",
        "ppval",
        "prctile",
        "quantile",
        "spline",
        "ttest",
        "ttest2",
    ],
    "matrix_ops.py": [
        "blkdiag",
        "conv",
        "cross",
        "deconv",
        "det",
        "dot",
        "fft",
        "fftshift",
        "find",
        "hankel",
        "hess",
        "ifft",
        "ifftshift",
        "intersect",
        "inv",
        "ischar",
        "isempty",
        "isequal",
        "ismember",
        "islogical",
        "isnumeric",
        "kron",
        "meshgrid",
        "ndgrid",
        "norm",
        "pinv",
        "rank",
        "schur",
        "setdiff",
        "setxor",
        "sort",
        "toeplitz",
        "union",
        "unique",
    ],
    "data_struct.py": [
        "cell2mat",
        "iscell",
        "isstruct",
        "mat2cell",
        "num2cell",
        "numel",
    ],
}

# Intra-module: remove second+ @register occurrence
intra_to_remove = {
    "data_struct.py": [
        "array2table",
        "cellfun",
        "containers_Map",
        "isKey",
        "keys",
        "table",
        "table2array",
        "values",
    ],
    "statistics.py": [
        "bootstrp",
        "geomean",
        "harmmean",
        "iqr",
        "jackknife",
        "kurtosis",
        "skewness",
        "tabulate",
        "zscore",
    ],
    "matrix_ops.py": [
        "cat",
        "diag",
        "fliplr",
        "flipud",
        "horzcat",
        "length",
        "linspace",
        "logspace",
        "ndims",
        "repmat",
        "reshape",
        "size",
        "tril",
        "triu",
        "vertcat",
    ],
    "advanced_math.py": ["cond", "expm", "logm", "sqrtm"],
    "plotting.py": [
        "contour",
        "contourf",
        "gcf",
        "mesh",
        "plot3",
        "saveas",
        "subplot",
        "surf",
    ],
    "image.py": [
        "edge",
        "histeq",
        "imadjust",
        "imbinarize",
        "imclose",
        "imcrop",
        "imdilate",
        "imerode",
        "imfilter",
        "imhist",
        "imopen",
        "imresize",
        "imrotate",
    ],
    "control.py": [
        "dcgain",
        "feedback",
        "isstable",
        "minreal",
        "parallel",
        "pole",
        "series",
        "zero",
        "zpk",
    ],
    "sparse.py": ["issparse", "nnz", "nonzeros", "spconvert"],
    "io.py": ["input", "nargin", "nargout"],
    "common.py": ["datetime", "isfloat", "isinteger"],
    "file_io.py": ["exist"],
}

print("=== Phase 1: Cross-module - Remove entire functions ===")
for fname, func_names in sorted(files_to_remove_funcs.items()):
    fpath = os.path.join(builtin_dir, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    new_content, removed = remove_register_and_func(content, func_names)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"  {fname}: removed {len(removed)} functions: {removed}")

print("\n=== Phase 2: Cross-module - Remove @register lines only ===")
for fname, func_names in sorted(files_remove_register_only.items()):
    fpath = os.path.join(builtin_dir, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    new_content, removed = remove_register_only(content, func_names)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"  {fname}: removed {len(removed)} @register lines: {removed}")

print("\n=== Phase 3: Intra-module - Remove duplicate @register lines ===")
for fname, func_names in sorted(intra_to_remove.items()):
    fpath = os.path.join(builtin_dir, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    seen = {}
    new_lines = []
    removed = []
    for line in lines:
        match = re.match(r"""@register\(['"]([^'"]+)['"]\)$""", line.strip())
        if match and match.group(1) in func_names:
            name = match.group(1)
            if name in seen:
                removed.append(name)
                continue
            seen[name] = 1
        new_lines.append(line)

    with open(fpath, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"  {fname}: removed {len(removed)} intra-module duplicates: {removed}")

print("\nDone! Run tests to verify.")
