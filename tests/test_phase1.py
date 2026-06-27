"""Phase 1 tests: Language core enhancements."""

import numpy as np
import pytest
from tests.conftest import run_matlab, get_val


class TestCommandSyntax:
    """Test command-style syntax generalization."""

    def test_disp_command(self, run):
        """Test disp with command syntax."""
        interp = run("disp hello")
        # Should not raise

    def test_load_command(self, run):
        """Test load with command syntax."""
        # Just verify it parses correctly
        interp = run("load data.mat")

    def test_multiple_args(self, run):
        """Test command syntax with multiple arguments."""
        interp = run("disp hello world")

    def test_grid_on(self, run):
        """Test grid on command."""
        interp = run("grid on")

    def test_hold_on(self, run):
        """Test hold on command."""
        interp = run("hold on")


class TestSwitchCaseCell:
    """Test switch/case with cell arrays."""

    def test_switch_cell_basic(self, run):
        """Test switch with cell array case."""
        interp = run("""
            x = 2;
            result = 0;
            switch x
                case {1, 2, 3}
                    result = 1;
                otherwise
                    result = 2;
            end
        """)
        assert get_val(interp.global_env.get("result")) == 1

    def test_switch_cell_no_match(self, run):
        """Test switch with cell array case - no match."""
        interp = run("""
            x = 5;
            result = 0;
            switch x
                case {1, 2, 3}
                    result = 1;
                otherwise
                    result = 2;
            end
        """)
        assert get_val(interp.global_env.get("result")) == 2

    def test_switch_string(self, run):
        """Test switch with string case."""
        interp = run("""
            x = 'hello';
            result = 0;
            switch x
                case 'hello'
                    result = 1;
                case 'world'
                    result = 2;
                otherwise
                    result = 3;
            end
        """)
        assert get_val(interp.global_env.get("result")) == 1

    def test_switch_multiple_cases(self, run):
        """Test switch with multiple cases."""
        interp = run("""
            x = 3;
            result = 0;
            switch x
                case 1
                    result = 10;
                case 2
                    result = 20;
                case 3
                    result = 30;
                otherwise
                    result = 99;
            end
        """)
        assert get_val(interp.global_env.get("result")) == 30


class TestNestedFunctions:
    """Test nested function support."""

    def test_nested_basic(self, run):
        """Test basic nested function."""
        interp = run("""
            function result = outer()
                x = 10;
                result = inner();
                function val = inner()
                    val = x;
                end
            end
            y = outer();
        """)
        assert get_val(interp.global_env.get("y")) == 10

    def test_nested_with_args(self, run):
        """Test nested function with arguments."""
        interp = run("""
            function result = outer(a)
                b = 5;
                result = inner(a);
                function val = inner(x)
                    val = x + b;
                end
            end
            y = outer(3);
        """)
        assert get_val(interp.global_env.get("y")) == 8

    def test_nested_modifies_outer(self, run):
        """Test nested function modifying outer variable."""
        interp = run("""
            function result = outer()
                x = 10;
                inner();
                result = x;
                function inner()
                    x = 20;
                end
            end
            y = outer();
        """)
        # In MATLAB, nested functions can modify outer variables
        # This tests our closure implementation


class TestLocalFunctions:
    """Test local function support."""

    def test_local_function(self, run):
        """Test local function is accessible within file."""
        interp = run("""
            function result = main()
                result = helper(5);
            end

            function val = helper(x)
                val = x * 2;
            end
            y = main();
        """)
        assert get_val(interp.global_env.get("y")) == 10

    def test_local_function_not_global(self, run):
        """Test local function is registered in function table."""
        interp = run("""
            function result = main()
                result = helper(5);
            end

            function val = helper(x)
                val = x * 2;
            end
        """)
        # Both functions should be in the function table
        assert "main" in interp.functions
        assert "helper" in interp.functions


class TestNdimArrayOperations:
    """Test N-D array operations."""

    def test_cat_basic(self, run):
        """Test cat function."""
        interp = run("""
            A = [1, 2; 3, 4];
            B = [5, 6; 7, 8];
            C = cat(1, A, B);
        """)
        C = interp.global_env.get("C")
        assert C.data.shape == (4, 2)

    def test_vertcat(self, run):
        """Test vertcat function."""
        interp = run("""
            A = [1, 2; 3, 4];
            B = [5, 6; 7, 8];
            C = vertcat(A, B);
        """)
        C = interp.global_env.get("C")
        assert C.data.shape == (4, 2)

    def test_horzcat(self, run):
        """Test horzcat function."""
        interp = run("""
            A = [1, 2; 3, 4];
            B = [5, 6; 7, 8];
            C = horzcat(A, B);
        """)
        C = interp.global_env.get("C")
        assert C.data.shape == (2, 4)

    def test_ndims(self, run):
        """Test ndims function."""
        interp = run("""
            A = zeros(2, 3, 4);
            n = ndims(A);
        """)
        assert get_val(interp.global_env.get("n")) == 3

    def test_permute(self, run):
        """Test permute function."""
        interp = run("""
            A = zeros(2, 3, 4);
            B = permute(A, [3, 1, 2]);
        """)
        B = interp.global_env.get("B")
        assert B.data.shape == (4, 2, 3)

    def test_ipermute(self, run):
        """Test ipermute function."""
        interp = run("""
            A = zeros(2, 3, 4);
            B = permute(A, [3, 1, 2]);
            C = ipermute(B, [3, 1, 2]);
        """)
        C = interp.global_env.get("C")
        assert C.data.shape == (2, 3, 4)

    def test_shiftdim(self, run):
        """Test shiftdim function."""
        interp = run("""
            A = zeros(1, 2, 3);
            B = shiftdim(A);
        """)
        B = interp.global_env.get("B")
        assert B.data.shape == (2, 3)

    def test_squeeze(self, run):
        """Test squeeze function."""
        interp = run("""
            A = zeros(2, 1, 3);
            B = squeeze(A);
        """)
        B = interp.global_env.get("B")
        assert B.data.shape == (2, 3)

    def test_reshape(self, run):
        """Test reshape function."""
        interp = run("""
            A = [1, 2, 3, 4, 5, 6];
            B = reshape(A, 2, 3);
        """)
        B = interp.global_env.get("B")
        assert B.data.shape == (2, 3)

    def test_circshift(self, run):
        """Test circshift function."""
        interp = run("""
            A = [1; 2; 3; 4];
            B = circshift(A, 2);
        """)
        B = interp.global_env.get("B")
        expected = np.array([[3], [4], [1], [2]])
        np.testing.assert_array_equal(B.data, expected)

    def test_flip(self, run):
        """Test flip function."""
        interp = run("""
            A = [1, 2, 3; 4, 5, 6];
            B = flip(A, 1);
        """)
        B = interp.global_env.get("B")
        expected = np.array([[4, 5, 6], [1, 2, 3]])
        np.testing.assert_array_equal(B.data, expected)

    def test_fliplr(self, run):
        """Test fliplr function."""
        interp = run("""
            A = [1, 2, 3; 4, 5, 6];
            B = fliplr(A);
        """)
        B = interp.global_env.get("B")
        expected = np.array([[3, 2, 1], [6, 5, 4]])
        np.testing.assert_array_equal(B.data, expected)

    def test_flipud(self, run):
        """Test flipud function."""
        interp = run("""
            A = [1, 2, 3; 4, 5, 6];
            B = flipud(A);
        """)
        B = interp.global_env.get("B")
        expected = np.array([[4, 5, 6], [1, 2, 3]])
        np.testing.assert_array_equal(B.data, expected)

    def test_rot90(self, run):
        """Test rot90 function."""
        interp = run("""
            A = [1, 2; 3, 4];
            B = rot90(A);
        """)
        B = interp.global_env.get("B")
        expected = np.array([[2, 4], [1, 3]])
        np.testing.assert_array_equal(B.data, expected)

    def test_size_3d(self, run):
        """Test size function on 3D array."""
        interp = run("""
            A = zeros(2, 3, 4);
            s = size(A);
        """)
        s = interp.global_env.get("s")
        np.testing.assert_array_equal(s.data, np.array([2, 3, 4]))

    def test_size_3d_dim(self, run):
        """Test size function on 3D array with dimension."""
        interp = run("""
            A = zeros(2, 3, 4);
            n = size(A, 3);
        """)
        assert get_val(interp.global_env.get("n")) == 4

    def test_length_3d(self, run):
        """Test length function on 3D array."""
        interp = run("""
            A = zeros(2, 5, 3);
            n = length(A);
        """)
        assert get_val(interp.global_env.get("n")) == 5

    def test_numel_3d(self, run):
        """Test numel function on 3D array."""
        interp = run("""
            A = zeros(2, 3, 4);
            n = numel(A);
        """)
        assert get_val(interp.global_env.get("n")) == 24

    def test_for_loop_3d(self, run):
        """Test for loop over 3D array."""
        interp = run("""
            A = zeros(2, 3, 4);
            total = 0;
            for i = 1:2
                for j = 1:3
                    for k = 1:4
                        A(i,j,k) = i*100 + j*10 + k;
                    end
                end
            end
            result = A(2,3,4);
        """)
        assert get_val(interp.global_env.get("result")) == 234

    def test_3d_indexing(self, run):
        """Test 3D array indexing."""
        interp = run("""
            A = zeros(2, 3, 4);
            A(1,2,3) = 42;
            result = A(1,2,3);
        """)
        assert get_val(interp.global_env.get("result")) == 42

    def test_3d_assignment(self, run):
        """Test 3D array assignment."""
        interp = run("""
            A = zeros(2, 3, 4);
            A(2,3,4) = 99;
            result = A(2,3,4);
        """)
        assert get_val(interp.global_env.get("result")) == 99
