"""File I/O built-in functions for MatPy."""

import os
import numpy as np
from matpy.builtins import register
from matpy.runtime.types import Mat

_file_handles: dict[int, object] = {}
_next_handle = 1


@register("fopen")
def _fopen(filename, permission="r"):
    global _next_handle
    filename = str(filename)
    try:
        f = open(filename, permission)
        handle = _next_handle
        _file_handles[handle] = f
        _next_handle += 1
        return handle
    except FileNotFoundError:
        return -1


@register("fclose")
def _fclose(handle):
    handle = int(handle)
    if handle in _file_handles:
        _file_handles[handle].close()
        del _file_handles[handle]
        return 0
    return -1


@register("fgetl")
def _fgetl(handle):
    handle = int(handle)
    if handle not in _file_handles:
        return ""
    f = _file_handles[handle]
    line = f.readline()
    if line.endswith("\n"):
        line = line[:-1]
    return line


@register("fgets")
def _fgets(handle):
    return _fgetl(handle)


@register("csvread")
def _csvread(filename):
    filename = str(filename)
    try:
        data = np.loadtxt(filename, delimiter=",")
        return Mat(data)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return Mat(np.array([]))


@register("csvwrite")
def _csvwrite(filename, data):
    filename = str(filename)
    if isinstance(data, Mat):
        arr = data.data
    else:
        arr = np.array(data)
    try:
        np.savetxt(filename, arr, delimiter=",", fmt="%g")
        return 0
    except Exception as e:
        print(f"Error writing CSV: {e}")
        return -1


@register("load")
def _load(filename, *args):
    filename = str(filename)
    if filename.endswith(".mat"):
        try:
            from scipy.io import loadmat
            data = loadmat(filename)
            result = {}
            for key, val in data.items():
                if not key.startswith("__"):
                    result[key] = Mat(val)
            return result
        except ImportError:
            print("scipy required for .mat files")
            return {}
        except Exception as e:
            print(f"Error loading .mat file: {e}")
            return {}
    elif filename.endswith(".h5") or filename.endswith(".hdf5"):
        try:
            import h5py
            result = {}
            with h5py.File(filename, 'r') as f:
                for key in f.keys():
                    result[key] = Mat(np.array(f[key]))
            return result
        except ImportError:
            print("h5py required for HDF5 files")
            return {}
        except Exception as e:
            print(f"Error loading HDF5 file: {e}")
            return {}
    else:
        try:
            data = np.loadtxt(filename)
            return Mat(data)
        except Exception as e:
            print(f"Error loading file: {e}")
            return Mat(np.array([]))


@register("save")
def _save(filename, *args):
    filename = str(filename)
    try:
        if filename.endswith(".mat"):
            try:
                from scipy.io import savemat
                data = {}
                for i, arg in enumerate(args):
                    if isinstance(arg, Mat):
                        data[f"var{i}"] = arg.data
                    else:
                        data[f"var{i}"] = np.array(arg)
                savemat(filename, data)
                return 0
            except ImportError:
                print("scipy required for .mat files")
                return -1
        elif filename.endswith(".h5") or filename.endswith(".hdf5"):
            try:
                import h5py
                with h5py.File(filename, 'w') as f:
                    for i, arg in enumerate(args):
                        if isinstance(arg, Mat):
                            f.create_dataset(f"var{i}", data=arg.data)
                        else:
                            f.create_dataset(f"var{i}", data=np.array(arg))
                return 0
            except ImportError:
                print("h5py required for HDF5 files")
                return -1
        else:
            with open(filename, "w") as f:
                for arg in args:
                    if isinstance(arg, Mat):
                        np.savetxt(f, arg.data, fmt="%g")
                    else:
                        f.write(str(arg) + "\n")
            return 0
    except Exception as e:
        print(f"Error saving: {e}")
        return -1


@register("matfile")
def _matfile(filename, mode='r'):
    """MATLAB .mat file access."""
    try:
        from scipy.io import loadmat, savemat
        filename = str(filename)
        if mode == 'r':
            return loadmat(filename)
        else:
            return {}  # Return empty dict for writing
    except ImportError:
        print("scipy required for .mat files")
        return {}


@register("h5read")
def _h5read(filename, dataset):
    """Read dataset from HDF5 file."""
    try:
        import h5py
        with h5py.File(str(filename), 'r') as f:
            return Mat(np.array(f[str(dataset)]))
    except ImportError:
        print("h5py required for HDF5 files")
        return Mat(np.array([]))
    except Exception as e:
        print(f"Error reading HDF5: {e}")
        return Mat(np.array([]))


@register("h5write")
def _h5write(filename, dataset, data):
    """Write dataset to HDF5 file."""
    try:
        import h5py
        with h5py.File(str(filename), 'a') as f:
            if isinstance(data, Mat):
                f.create_dataset(str(dataset), data=data.data)
            else:
                f.create_dataset(str(dataset), data=np.array(data))
        return 0
    except ImportError:
        print("h5py required for HDF5 files")
        return -1
    except Exception as e:
        print(f"Error writing HDF5: {e}")
        return -1


@register("h5create")
def _h5create(filename, dataset, size, datatype='double'):
    """Create dataset in HDF5 file."""
    try:
        import h5py
        dtype_map = {'double': np.float64, 'single': np.float32, 'int32': np.int32, 'int64': np.int64}
        dtype = dtype_map.get(datatype, np.float64)
        with h5py.File(str(filename), 'a') as f:
            f.create_dataset(str(dataset), shape=tuple(int(s) for s in size), dtype=dtype)
        return 0
    except ImportError:
        print("h5py required for HDF5 files")
        return -1
    except Exception as e:
        print(f"Error creating HDF5 dataset: {e}")
        return -1


@register("h5info")
def _h5info(filename):
    """Get information about HDF5 file."""
    try:
        import h5py
        with h5py.File(str(filename), 'r') as f:
            print(f"File: {filename}")
            print(f"Datasets:")
            for key in f.keys():
                ds = f[key]
                print(f"  {key}: shape={ds.shape}, dtype={ds.dtype}")
        return 0
    except ImportError:
        print("h5py required for HDF5 files")
        return -1
    except Exception as e:
        print(f"Error reading HDF5 info: {e}")
        return -1


@register("jsonencode")
def _jsonencode(x):
    """Encode to JSON string."""
    import json
    if isinstance(x, Mat):
        return json.dumps(x.data.tolist())
    elif isinstance(x, dict):
        return json.dumps(x)
    elif isinstance(x, (list, tuple)):
        return json.dumps(x)
    return json.dumps(x)


@register("jsondecode")
def _jsondecode(s):
    """Decode JSON string."""
    import json
    data = json.loads(str(s))
    if isinstance(data, list):
        return Mat(np.array(data))
    return data


@register("readtable")
def _readtable(filename, *args):
    """Read table from file."""
    try:
        import pandas as pd
        return pd.read_csv(str(filename))
    except ImportError:
        print("pandas required for readtable")
        return {}


@register("writetable")
def _writetable(T, filename):
    """Write table to file."""
    try:
        import pandas as pd
        if isinstance(T, pd.DataFrame):
            T.to_csv(str(filename), index=False)
            return 0
        return -1
    except ImportError:
        print("pandas required for writetable")
        return -1


@register("exist")
def _exist(name, kind="var"):
    name = str(name)
    if kind == "file":
        return os.path.exists(name)
    elif kind == "dir":
        return os.path.isdir(name)
    return False


@register("dir")
def _dir(path="."):
    path = str(path)
    try:
        entries = os.listdir(path)
        for entry in entries:
            print(entry)
        return entries
    except Exception as e:
        print(f"Error: {e}")
        return []


@register("pwd")
def _pwd():
    return os.getcwd()


@register("cd")
def _cd(path=None):
    if path is None:
        return os.getcwd()
    try:
        os.chdir(str(path))
        return os.getcwd()
    except Exception as e:
        print(f"Error: {e}")
        return os.getcwd()


@register("mkdir")
def _mkdir(path):
    try:
        os.makedirs(str(path), exist_ok=True)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return -1


@register("delete")
def _delete(filename):
    try:
        os.remove(str(filename))
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return -1
