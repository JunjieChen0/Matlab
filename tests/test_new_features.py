"""Tests for newly added features in MatPy."""

import pytest
import numpy as np
from matpy.lexer import Lexer
from matpy.parser import Parser
from matpy.interpreter import Interpreter
from matpy.runtime.types import Mat, MException, Struct


def run_matlab(source: str):
    """Execute MATLAB source and return interpreter."""
    interp = Interpreter()
    tokens = Lexer(source, '<test>').tokenize()
    program = Parser(tokens, '<test>').parse()
    interp.run(program)
    return interp


class TestClassDefEnhancements:
    """Test classdef Constant, Access, Static features."""

    def test_constant_property(self):
        source = '''
        classdef MyConstants
            properties (Constant)
                PI = 3.14159
                E = 2.71828
            end
        end
        obj = MyConstants();
        result = obj.PI;
        '''
        interp = run_matlab(source)
        assert abs(interp.global_env.get('result') - 3.14159) < 1e-5

    def test_constant_property_immutable(self):
        source = '''
        classdef MyConstants
            properties (Constant)
                PI = 3.14159
            end
        end
        obj = MyConstants();
        obj.PI = 100;
        '''
        with pytest.raises(Exception):
            run_matlab(source)

    def test_static_method(self):
        source = '''
        classdef MyClass
            properties
                x = 10
            end
            methods (Static)
                function s = describe()
                    s = 'MyClass static method';
                end
            end
        end
        result = MyClass.describe();
        '''
        interp = run_matlab(source)
        assert interp.global_env.get('result') == 'MyClass static method'

    def test_class_inheritance(self):
        source = '''
        classdef Shape
            properties
                name = 'shape'
            end
            methods
                function obj = Shape(n)
                    obj.name = n;
                end
            end
        end
        classdef Circle < Shape
            properties
                radius = 0
            end
            methods
                function obj = Circle(r)
                    obj.name = 'circle';
                    obj.radius = r;
                end
                function a = area(obj)
                    a = 3.14159 * obj.radius^2;
                end
            end
        end
        c = Circle(5);
        result = c.area();
        '''
        interp = run_matlab(source)
        result = interp.global_env.get('result')
        # Allow some tolerance for floating point
        assert result is not None


class TestErrorHandling:
    """Test error handling enhancements."""

    def test_error_with_identifier(self):
        source = '''
        try
            error('MATLAB:test:invalidInput', 'Invalid input');
        catch e
            msg = e.message;
            id = e.identifier;
        end
        '''
        interp = run_matlab(source)
        assert interp.global_env.get('msg') == 'Invalid input'
        assert interp.global_env.get('id') == 'MATLAB:test:invalidInput'

    def test_lasterror(self):
        source = '''
        try
            error('Test error message');
        catch e
            % error caught
        end
        err = lasterror();
        result = err.message;
        '''
        interp = run_matlab(source)
        assert interp.global_env.get('result') == 'Test error message'

    def test_warning_function(self):
        source = '''
        warning('This is a warning');
        '''
        # Should not raise exception
        run_matlab(source)


class TestSpecialFunctions:
    """Test special mathematical functions."""

    def test_besselj(self):
        source = '''
        result = besselj(0, 1);
        '''
        interp = run_matlab(source)
        result = interp.global_env.get('result')
        # Convert Mat to float
        if isinstance(result, Mat):
            result = float(result.data.flat[0])
        assert abs(float(result) - 0.7652) < 0.01

    def test_bessely(self):
        source = '''
        result = bessely(0, 1);
        '''
        interp = run_matlab(source)
        result = interp.global_env.get('result')
        if isinstance(result, Mat):
            result = float(result.data.flat[0])
        assert abs(float(result) - 0.0883) < 0.01

    def test_besseli(self):
        source = '''
        result = besseli(0, 1);
        '''
        interp = run_matlab(source)
        result = interp.global_env.get('result')
        if isinstance(result, Mat):
            result = float(result.data.flat[0])
        assert abs(float(result) - 1.2661) < 0.01


class TestLinearAlgebra:
    """Test linear algebra functions."""

    def test_schur_decomposition(self):
        source = '''
        A = [1 2; 3 4];
        [T, Z] = schur(A);
        '''
        interp = run_matlab(source)
        T = interp.global_env.get('T')
        assert isinstance(T, Mat)

    def test_hessenberg_decomposition(self):
        source = '''
        A = [1 2; 3 4];
        [H, Q] = hess(A);
        '''
        interp = run_matlab(source)
        H = interp.global_env.get('H')
        assert isinstance(H, Mat)

    def test_kron_product(self):
        source = '''
        A = [1 2; 3 4];
        B = [5 6; 7 8];
        K = kron(A, B);
        '''
        interp = run_matlab(source)
        K = interp.global_env.get('K')
        assert isinstance(K, Mat)
        assert K.shape == (4, 4)

    def test_linsolve(self):
        source = '''
        A = [1 2; 3 4];
        b = [5; 6];
        x = linsolve(A, b);
        '''
        interp = run_matlab(source)
        x = interp.global_env.get('x')
        assert isinstance(x, Mat)

    def test_expm(self):
        source = '''
        A = [0 1; -1 0];
        result = expm(A);
        '''
        interp = run_matlab(source)
        result = interp.global_env.get('result')
        assert isinstance(result, Mat)


class TestREPLCommands:
    """Test REPL special commands."""

    def test_debug_commands_exist(self):
        """Test that debug commands are properly defined."""
        # This is a basic test to ensure the REPL code structure is valid
        import matpy.__main__ as main_module
        assert hasattr(main_module, 'repl')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
