"""Comprehensive tests for file_io.py to boost coverage."""

import numpy as np
import os
from tests.conftest import run_matlab
from matpy.runtime.types import Mat, Struct


class TestFileHandleFunctions:
    """Test file handle functions (fopen, fclose, fgetl, fgets)."""

    def test_fopen_fclose(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello")
        from matpy.builtins.file_io import _fopen, _fclose

        fid = _fopen(str(filepath), "r")
        assert fid > 0
        result = _fclose(fid)
        assert result == 0

    def test_fopen_not_found(self):
        from matpy.builtins.file_io import _fopen

        fid = _fopen("/nonexistent/file.txt", "r")
        assert fid == -1

    def test_fclose_invalid(self):
        from matpy.builtins.file_io import _fclose

        result = _fclose(999)
        assert result == -1

    def test_fgetl(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello\nworld\n")
        from matpy.builtins.file_io import _fopen, _fgetl, _fclose

        fid = _fopen(str(filepath), "r")
        line = _fgetl(fid)
        assert line == "hello"
        _fclose(fid)

    def test_fgets(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello\nworld\n")
        from matpy.builtins.file_io import _fopen, _fgets, _fclose

        fid = _fopen(str(filepath), "r")
        line = _fgets(fid)
        assert line == "hello"
        _fclose(fid)

    def test_fgetl_invalid_handle(self):
        from matpy.builtins.file_io import _fgetl

        result = _fgetl(999)
        assert result == ""


class TestCSVFunctions:
    """Test CSV read/write functions."""

    def test_csvwrite(self, tmp_path):
        filepath = tmp_path / "test.csv"
        from matpy.builtins.file_io import _csvwrite

        result = _csvwrite(str(filepath), Mat(np.array([[1, 2], [3, 4]])))
        assert result == 0
        assert filepath.exists()

    def test_csvread(self, tmp_path):
        filepath = tmp_path / "test.csv"
        filepath.write_text("1,2\n3,4\n")
        from matpy.builtins.file_io import _csvread

        result = _csvread(str(filepath))
        assert isinstance(result, Mat)

    def test_csvread_error(self):
        from matpy.builtins.file_io import _csvread

        result = _csvread("/nonexistent/file.csv")
        assert isinstance(result, Mat)


class TestMatFileFunctions:
    """Test .mat file functions."""

    def test_save_load_mat(self, tmp_path):
        filepath = tmp_path / "test.mat"
        from matpy.builtins.file_io import _save, _load

        data = Mat(np.array([[1, 2], [3, 4]]))
        result = _save(str(filepath), data)
        assert result == 0
        loaded = _load(str(filepath))
        assert isinstance(loaded, dict)

    def test_save_load_struct(self, tmp_path):
        filepath = tmp_path / "test.mat"
        from matpy.builtins.file_io import _save

        s = Struct({"a": Mat(np.array([1, 2, 3]))})
        result = _save(str(filepath), s)
        assert result == 0

    def test_save_txt(self, tmp_path):
        filepath = tmp_path / "test.txt"
        from matpy.builtins.file_io import _save

        data = Mat(np.array([[1, 2], [3, 4]]))
        result = _save(str(filepath), data)
        assert result == 0
        assert filepath.exists()

    def test_load_txt(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("1 2\n3 4\n")
        from matpy.builtins.file_io import _load

        result = _load(str(filepath))
        assert isinstance(result, Mat)

    def test_matfile(self, tmp_path):
        filepath = tmp_path / "test.mat"
        # Create the file first
        from scipy.io import savemat

        savemat(str(filepath), {"data": np.array([1, 2, 3])})
        from matpy.builtins.file_io import _matfile

        result = _matfile(str(filepath), "r")
        assert isinstance(result, dict)


class TestJSONFunctions:
    """Test JSON encode/decode functions."""

    def test_jsonencode_mat(self):
        from matpy.builtins.file_io import _jsonencode

        result = _jsonencode(Mat(np.array([1, 2, 3])))
        assert isinstance(result, str)

    def test_jsonencode_dict(self):
        from matpy.builtins.file_io import _jsonencode

        result = _jsonencode({"a": 1, "b": 2})
        assert isinstance(result, str)

    def test_jsonencode_list(self):
        from matpy.builtins.file_io import _jsonencode

        result = _jsonencode([1, 2, 3])
        assert isinstance(result, str)

    def test_jsondecode(self):
        from matpy.builtins.file_io import _jsondecode

        result = _jsondecode("[1, 2, 3]")
        assert isinstance(result, Mat)

    def test_jsondecode_dict(self):
        from matpy.builtins.file_io import _jsondecode

        result = _jsondecode('{"a": 1, "b": 2}')
        assert isinstance(result, dict)


class TestExistFunctions:
    """Test file existence check functions."""

    def test_exist_file(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello")
        from matpy.builtins.file_io import _exist

        assert _exist(str(filepath), "file")

    def test_exist_dir(self, tmp_path):
        from matpy.builtins.file_io import _exist

        assert _exist(str(tmp_path), "dir")

    def test_exist_func(self):
        from matpy.builtins.file_io import _exist

        assert _exist("sin", "func")

    def test_exist_builtin(self):
        from matpy.builtins.file_io import _exist

        assert _exist("zeros", "builtin")

    def test_exist_any(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello")
        from matpy.builtins.file_io import _exist

        assert _exist(str(filepath))

    def test_isfile(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello")
        from matpy.builtins.file_io import _isfile

        assert _isfile(str(filepath))

    def test_isfolder(self, tmp_path):
        from matpy.builtins.file_io import _isfolder

        assert _isfolder(str(tmp_path))


class TestPathFunctions:
    """Test path manipulation functions."""

    def test_pwd(self):
        from matpy.builtins.file_io import _pwd

        result = _pwd()
        assert isinstance(result, str)
        assert os.path.isdir(result)

    def test_cd(self, tmp_path):
        from matpy.builtins.file_io import _cd, _pwd

        original = _pwd()
        result = _cd(str(tmp_path))
        assert result == str(tmp_path)
        _cd(original)

    def test_mkdir(self, tmp_path):
        newdir = tmp_path / "newdir"
        from matpy.builtins.file_io import _mkdir

        result = _mkdir(str(newdir))
        assert result == 0
        assert newdir.exists()

    def test_delete(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello")
        from matpy.builtins.file_io import _delete

        result = _delete(str(filepath))
        assert result == 0
        assert not filepath.exists()

    def test_dir(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello")
        from matpy.builtins.file_io import _dir

        result = _dir(str(tmp_path))
        assert isinstance(result, list)
        assert "test.txt" in result

    def test_filesep(self):
        from matpy.builtins.file_io import _filesep

        result = _filesep()
        assert result in ("/", "\\")

    def test_pathsep(self):
        from matpy.builtins.file_io import _pathsep

        result = _pathsep()
        assert result in (":", ";")

    def test_fullfile(self):
        from matpy.builtins.file_io import _fullfile

        result = _fullfile("home", "user", "file.txt")
        assert isinstance(result, str)
        assert "file.txt" in result

    def test_fileparts(self):
        from matpy.builtins.file_io import _fileparts

        directory, base, ext = _fileparts("/home/user/test.txt")
        assert directory == "/home/user"
        assert base == "test"
        assert ext == ".txt"


class TestTempFunctions:
    """Test temporary file functions."""

    def test_tempname(self):
        from matpy.builtins.file_io import _tempname

        result = _tempname()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_tempdir(self):
        from matpy.builtins.file_io import _tempdir

        result = _tempdir()
        assert isinstance(result, str)
        assert os.path.isdir(result)


class TestPathManipulation:
    """Test path manipulation functions."""

    def test_path_get(self):
        from matpy.builtins.file_io import _path

        result = _path()
        assert isinstance(result, str)

    def test_path_set(self, tmp_path):
        from matpy.builtins.file_io import _path
        import sys

        original_path = sys.path.copy()
        _path(str(tmp_path))
        assert str(tmp_path) in sys.path
        sys.path = original_path

    def test_addpath(self, tmp_path):
        from matpy.builtins.file_io import _addpath
        import sys

        original_path = sys.path.copy()
        _addpath(str(tmp_path))
        assert str(tmp_path) in sys.path
        sys.path = original_path

    def test_rmpath(self, tmp_path):
        from matpy.builtins.file_io import _rmpath, _addpath
        import sys

        original_path = sys.path.copy()
        _addpath(str(tmp_path))
        _rmpath(str(tmp_path))
        assert str(tmp_path) not in sys.path
        sys.path = original_path


class TestFileIOIntegration:
    """Integration tests through interpreter."""

    def test_fopen_fclose_interpreter(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello")
        run_matlab(f"fid = fopen('{filepath}');\nfclose(fid);")
        assert True

    def test_csvwrite_interpreter(self, tmp_path):
        filepath = tmp_path / "test.csv"
        run_matlab(f"csvwrite('{filepath}', [1 2; 3 4]);")
        assert filepath.exists()

    def test_csvread_interpreter(self, tmp_path):
        filepath = tmp_path / "test.csv"
        filepath.write_text("1,2\n3,4\n")
        interp = run_matlab(f"M = csvread('{filepath}');")
        M = interp.global_env.get("M")
        assert isinstance(M, Mat)

    def test_exist_interpreter(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello")
        interp = run_matlab(f"r = exist('{filepath}', 'file');")
        r = interp.global_env.get("r")
        assert r

    def test_isfile_interpreter(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello")
        interp = run_matlab(f"r = isfile('{filepath}');")
        r = interp.global_env.get("r")
        assert r

    def test_isfolder_interpreter(self, tmp_path):
        interp = run_matlab(f"r = isfolder('{tmp_path}');")
        r = interp.global_env.get("r")
        assert r

    def test_pwd_interpreter(self):
        interp = run_matlab("d = pwd();")
        d = interp.global_env.get("d")
        assert d is not None

    def test_tempname_interpreter(self):
        interp = run_matlab("t = tempname();")
        t = interp.global_env.get("t")
        assert t is not None

    def test_tempdir_interpreter(self):
        interp = run_matlab("d = tempdir();")
        d = interp.global_env.get("d")
        assert d is not None

    def test_jsonencode_interpreter(self):
        interp = run_matlab("j = jsonencode([1 2 3]);")
        j = interp.global_env.get("j")
        assert j is not None

    def test_jsondecode_interpreter(self):
        interp = run_matlab('v = jsondecode("[1, 2, 3]");')
        v = interp.global_env.get("v")
        assert isinstance(v, Mat)
