"""Comprehensive tests for data_struct.py to boost coverage."""

import numpy as np
from tests.conftest import run_matlab, get_val
from matpy.runtime.types import Mat, CellArray, Struct, Map, Table


class TestCellFunctions:
    """Test cell array functions."""

    def test_cell_1d(self):
        from matpy.builtins.data_struct import _cell

        result = _cell(3)
        assert isinstance(result, CellArray)

    def test_cell_2d(self):
        from matpy.builtins.data_struct import _cell

        result = _cell(2, 3)
        assert isinstance(result, CellArray)
        assert result.shape == (2, 3)

    def test_cell_empty(self):
        from matpy.builtins.data_struct import _cell

        result = _cell()
        assert isinstance(result, CellArray)

    def test_cell_numel(self):
        from matpy.builtins.data_struct import _numel

        c = CellArray([[1, 2], [3, 4]])
        assert _numel(c) == 4

    def test_cell_iscell(self):
        from matpy.builtins.data_struct import _iscell

        c = CellArray([[1, 2]])
        assert _iscell(c)
        assert not _iscell(Mat(np.array([1])))

    def test_celldisp(self, capsys):
        from matpy.builtins.data_struct import _celldisp

        c = CellArray([[1, 2], [3, 4]])
        _celldisp(c)
        captured = capsys.readouterr()
        assert "{1,1}" in captured.out

    def test_cellfun(self):
        from matpy.builtins.data_struct import _cellfun

        c = CellArray([[1, 2], [3, 4]])
        result = _cellfun(lambda x: x * 2, c)
        assert isinstance(result, Mat)

    def test_cell2mat(self):
        from matpy.builtins.data_struct import _cell2mat

        c = CellArray([[Mat(np.array([1, 2]))], [Mat(np.array([3, 4]))]])
        result = _cell2mat(c)
        assert isinstance(result, Mat)

    def test_mat2cell(self):
        from matpy.builtins.data_struct import _mat2cell

        x = Mat(np.array([[1, 2], [3, 4]]))
        result = _mat2cell(x)
        assert isinstance(result, CellArray)

    def test_num2cell(self):
        from matpy.builtins.data_struct import _num2cell

        x = Mat(np.array([[1, 2], [3, 4]]))
        result = _num2cell(x)
        assert isinstance(result, CellArray)

    def test_num2cell_dim(self):
        from matpy.builtins.data_struct import _num2cell

        x = Mat(np.array([[1, 2], [3, 4]]))
        result = _num2cell(x, 1)
        assert isinstance(result, CellArray)

    def test_struct2cell(self):
        from matpy.builtins.data_struct import _struct2cell

        s = Struct({"a": 1, "b": 2})
        result = _struct2cell(s)
        assert isinstance(result, CellArray)

    def test_cell2struct(self):
        from matpy.builtins.data_struct import _cell2struct

        c = CellArray([[1], [2]])
        names = ["a", "b"]
        result = _cell2struct(c, names)
        assert isinstance(result, Struct)
        assert result.has_field("a")
        assert result.has_field("b")


class TestStructFunctions:
    """Test struct functions."""

    def test_struct_create(self):
        from matpy.builtins.data_struct import _struct

        result = _struct("name", "test", "value", 42)
        assert isinstance(result, Struct)
        assert result.get_field("name") == "test"
        assert result.get_field("value") == 42

    def test_struct_empty(self):
        from matpy.builtins.data_struct import _struct

        result = _struct()
        assert isinstance(result, Struct)

    def test_fieldnames(self):
        from matpy.builtins.data_struct import _fieldnames

        s = Struct({"a": 1, "b": 2, "c": 3})
        result = _fieldnames(s)
        assert isinstance(result, list)
        assert "a" in result
        assert "b" in result
        assert "c" in result

    def test_fieldnames_non_struct(self):
        from matpy.builtins.data_struct import _fieldnames

        result = _fieldnames(Mat(np.array([1, 2, 3])))
        assert result == []

    def test_isfield_true(self):
        from matpy.builtins.data_struct import _isfield

        s = Struct({"a": 1, "b": 2})
        assert _isfield(s, "a")
        assert _isfield(s, "b")

    def test_isfield_false(self):
        from matpy.builtins.data_struct import _isfield

        s = Struct({"a": 1, "b": 2})
        assert not _isfield(s, "c")

    def test_isfield_non_struct(self):
        from matpy.builtins.data_struct import _isfield

        assert not _isfield(Mat(np.array([1])), "a")

    def test_rmfield(self):
        from matpy.builtins.data_struct import _rmfield

        s = Struct({"a": 1, "b": 2, "c": 3})
        result = _rmfield(s, "b")
        assert isinstance(result, Struct)
        assert result.has_field("a")
        assert not result.has_field("b")
        assert result.has_field("c")

    def test_rmfield_non_struct(self):
        from matpy.builtins.data_struct import _rmfield

        m = Mat(np.array([1, 2, 3]))
        result = _rmfield(m, "a")
        assert isinstance(result, Mat)

    def test_isstruct(self):
        from matpy.builtins.data_struct import _isstruct

        s = Struct({"a": 1})
        assert _isstruct(s)
        assert not _isstruct(Mat(np.array([1])))

    def test_struct_numel(self):
        from matpy.builtins.data_struct import _numel

        s = Struct({"a": 1, "b": 2})
        assert _numel(s) == 1


class TestTableFunctions:
    """Test table functions."""

    def test_table_create(self):
        from matpy.builtins.data_struct import _table

        col1 = Mat(np.array([1, 2, 3]))
        col2 = Mat(np.array([4, 5, 6]))
        result = _table(col1, col2, VariableNames=["A", "B"])
        assert isinstance(result, Table)

    def test_array2table(self):
        from matpy.builtins.data_struct import _array2table

        A = Mat(np.array([[1, 2], [3, 4], [5, 6]]))
        result = _array2table(A, VariableNames=["X", "Y"])
        assert isinstance(result, Table)

    def test_table2array(self):
        from matpy.builtins.data_struct import _table2array

        data = {"A": np.array([1, 2, 3]), "B": np.array([4, 5, 6])}
        t = Table(data)
        result = _table2array(t)
        assert isinstance(result, Mat)


class TestMapFunctions:
    """Test containers.Map functions."""

    def test_map_create(self):

        m = Map(Mat(np.array(["a", "b", "c"])), Mat(np.array([1, 2, 3])))
        assert isinstance(m, Map)

    def test_map_iskey(self):

        m = Map(Mat(np.array(["a", "b"])), Mat(np.array([1, 2])))
        assert m.isKey("a")
        assert not m.isKey("c")

    def test_map_keys(self):

        m = Map(Mat(np.array(["a", "b"])), Mat(np.array([1, 2])))
        result = m.keys()
        assert isinstance(result, list)

    def test_map_values(self):

        m = Map(Mat(np.array(["a", "b"])), Mat(np.array([1, 2])))
        result = m.values()
        assert isinstance(result, list)

    def test_map_remove(self):

        m = Map(Mat(np.array(["a", "b"])), Mat(np.array([1, 2])))
        m.remove("a")
        assert not m.isKey("a")


class TestDealFunction:
    """Test deal function."""

    def test_deal_single(self):
        from matpy.builtins.data_struct import _deal

        result = _deal(42)
        assert result == 42

    def test_deal_multiple(self):
        from matpy.builtins.data_struct import _deal

        result = _deal(1, 2, 3)
        assert isinstance(result, list)
        assert len(result) == 3


class TestDataStructIntegration:
    """Integration tests through interpreter."""

    def test_cell_creation_interpreter(self):
        interp = run_matlab("c = cell(2, 3);")
        c = interp.global_env.get("c")
        assert isinstance(c, CellArray)

    def test_struct_creation_interpreter(self):
        interp = run_matlab("s = struct('a', 1, 'b', 2);")
        s = interp.global_env.get("s")
        assert isinstance(s, Struct)
        assert s.get_field("a") == 1

    def test_fieldnames_interpreter(self):
        from matpy.builtins.data_struct import _fieldnames

        s = Struct({"x": 1, "y": 2})
        n = _fieldnames(s)
        assert isinstance(n, list)
        assert "x" in n
        assert "y" in n

    def test_isfield_interpreter(self):
        from matpy.builtins.data_struct import _isfield

        s = Struct({"a": 1})
        assert _isfield(s, "a")

    def test_rmfield_interpreter(self):
        from matpy.builtins.data_struct import _rmfield

        s = Struct({"a": 1, "b": 2})
        s2 = _rmfield(s, "b")
        assert isinstance(s2, Struct)
        assert s2.has_field("a")
        assert not s2.has_field("b")

    def test_struct_field_access(self):
        interp = run_matlab("s = struct('name', 'test', 'value', 42);\nv = s.value;")
        v = interp.global_env.get("v")
        assert get_val(v) == 42

    def test_struct_field_set(self):
        interp = run_matlab("s = struct();\ns.a = 1;\ns.b = 2;")
        s = interp.global_env.get("s")
        assert isinstance(s, Struct)
        assert s.get_field("a") == 1
        assert s.get_field("b") == 2

    def test_cell_indexing(self):
        from matpy.builtins.data_struct import _cell

        c = _cell(2, 2)
        c.set(1, 1, 42)
        v = c.get(1, 1)
        assert v == 42

    def test_nested_struct(self):
        interp = run_matlab("s = struct('inner', struct('x', 1));\nv = s.inner.x;")
        v = interp.global_env.get("v")
        assert get_val(v) == 1
