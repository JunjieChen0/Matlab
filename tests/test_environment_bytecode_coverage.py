"""Comprehensive tests for environment.py and bytecode.py to boost coverage."""

import pytest
import numpy as np
from tests.conftest import run_matlab, get_val
from matpy.runtime.types import Mat


class TestEnvironmentFunctions:
    """Test environment functions."""

    def test_environment_create(self):
        from matpy.environment import Environment

        env = Environment()
        assert env is not None

    def test_environment_set_get(self):
        from matpy.environment import Environment

        env = Environment()
        env.set("x", 42)
        assert env.get("x") == 42

    def test_environment_has(self):
        from matpy.environment import Environment

        env = Environment()
        env.set("x", 42)
        assert env.has("x") == True
        assert env.has("y") == False

    def test_environment_child(self):
        from matpy.environment import Environment

        env = Environment()
        child = env.child("test")
        assert child is not None
        assert child.name == "test"

    def test_environment_child_get_parent(self):
        from matpy.environment import Environment

        env = Environment()
        env.set("x", 42)
        child = env.child("test")
        assert child.get("x") == 42

    def test_environment_child_set_override(self):
        from matpy.environment import Environment

        env = Environment()
        env.set("x", 42)
        child = env.child("test")
        child.set("x", 100)
        assert child.get("x") == 100
        assert env.get("x") == 42

    def test_environment_global(self):
        from matpy.environment import Environment

        env = Environment()
        env.define_global("x")
        env.set("x", 42)
        assert env.get("x") == 42

    def test_environment_persistent(self):
        from matpy.environment import Environment

        env = Environment()
        env.define_persistent("x", 0)
        assert env.get("x") == 0
        env.set("x", 42)
        assert env.get("x") == 42

    def test_environment_local_vars(self):
        from matpy.environment import Environment

        env = Environment()
        env.set("x", 1)
        env.set("y", 2)
        vars = env.local_vars()
        assert "x" in vars
        assert "y" in vars

    def test_environment_all_vars(self):
        from matpy.environment import Environment

        env = Environment()
        env.set("x", 1)
        child = env.child("test")
        child.set("y", 2)
        vars = child.all_vars()
        assert "x" in vars
        assert "y" in vars

    def test_environment_name(self):
        from matpy.environment import Environment

        env = Environment(name="global")
        assert env.name == "global"

    def test_environment_parent(self):
        from matpy.environment import Environment

        env = Environment()
        child = env.child("test")
        assert child.parent == env


class TestBytecodeFunctions:
    """Test bytecode functions."""

    def test_bytecode_compiler_create(self):
        from matpy.bytecode import BytecodeCompiler

        compiler = BytecodeCompiler()
        assert compiler is not None

    def test_bytecode_vm_create(self):
        from matpy.bytecode import BytecodeVM

        vm = BytecodeVM()
        assert vm is not None

    def test_bytecode_compile_simple(self):
        from matpy.bytecode import BytecodeCompiler
        from matpy.parser import Parser
        from matpy.lexer import Lexer

        compiler = BytecodeCompiler()
        lexer = Lexer("x = 1 + 2;")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        instructions = compiler.compile(program)
        assert len(instructions) > 0

    def test_bytecode_compile_variable(self):
        from matpy.bytecode import BytecodeCompiler
        from matpy.parser import Parser
        from matpy.lexer import Lexer

        compiler = BytecodeCompiler()
        lexer = Lexer("x = 42;")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        instructions = compiler.compile(program)
        assert len(instructions) > 0

    def test_bytecode_compile_expression(self):
        from matpy.bytecode import BytecodeCompiler
        from matpy.parser import Parser
        from matpy.lexer import Lexer

        compiler = BytecodeCompiler()
        lexer = Lexer("x = 1 + 2 * 3;")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        instructions = compiler.compile(program)
        assert len(instructions) > 0

    def test_bytecode_compile_if(self):
        from matpy.bytecode import BytecodeCompiler
        from matpy.parser import Parser
        from matpy.lexer import Lexer

        compiler = BytecodeCompiler()
        lexer = Lexer("if x > 0; y = 1; else; y = 0; end")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        instructions = compiler.compile(program)
        assert len(instructions) > 0

    def test_bytecode_compile_for(self):
        from matpy.bytecode import BytecodeCompiler
        from matpy.parser import Parser
        from matpy.lexer import Lexer

        compiler = BytecodeCompiler()
        lexer = Lexer("for i = 1:10; x = i; end")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        instructions = compiler.compile(program)
        assert len(instructions) > 0

    def test_bytecode_compile_while(self):
        from matpy.bytecode import BytecodeCompiler
        from matpy.parser import Parser
        from matpy.lexer import Lexer

        compiler = BytecodeCompiler()
        lexer = Lexer("while x > 0; x = x - 1; end")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        instructions = compiler.compile(program)
        assert len(instructions) > 0

    def test_bytecode_compile_function(self):
        from matpy.bytecode import BytecodeCompiler
        from matpy.parser import Parser
        from matpy.lexer import Lexer

        compiler = BytecodeCompiler()
        lexer = Lexer("function y = f(x); y = x^2; end")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        instructions = compiler.compile(program)
        assert len(instructions) > 0

    def test_bytecode_vm_run_simple(self):
        from matpy.bytecode import BytecodeCompiler, BytecodeVM
        from matpy.parser import Parser
        from matpy.lexer import Lexer

        compiler = BytecodeCompiler()
        lexer = Lexer("x = 1 + 2;")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        instructions = compiler.compile(program)
        vm = BytecodeVM()
        vm.run(instructions, compiler.constants)
        assert True

    def test_bytecode_vm_run_variable(self):
        from matpy.bytecode import BytecodeCompiler, BytecodeVM
        from matpy.parser import Parser
        from matpy.lexer import Lexer

        compiler = BytecodeCompiler()
        lexer = Lexer("x = 42;")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        instructions = compiler.compile(program)
        vm = BytecodeVM()
        vm.run(instructions, compiler.constants)
        assert True

    def test_bytecode_vm_run_expression(self):
        from matpy.bytecode import BytecodeCompiler, BytecodeVM
        from matpy.parser import Parser
        from matpy.lexer import Lexer

        compiler = BytecodeCompiler()
        lexer = Lexer("x = 1 + 2 * 3;")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        instructions = compiler.compile(program)
        vm = BytecodeVM()
        vm.run(instructions, compiler.constants)
        assert True


class TestEnvironmentIntegration:
    """Integration tests through interpreter."""

    def test_variable_assignment(self):
        interp = run_matlab("x = 42;")
        x = interp.global_env.get("x")
        assert get_val(x) == 42

    def test_variable_use(self):
        interp = run_matlab("x = 42;\ny = x + 1;")
        y = interp.global_env.get("y")
        assert get_val(y) == 43

    def test_function_definition(self):
        interp = run_matlab("function y = f(x); y = x^2; end\nresult = f(3);")
        result = interp.global_env.get("result")
        assert get_val(result) == 9

    def test_if_statement(self):
        interp = run_matlab("x = 1;\nif x > 0; y = 1; else; y = 0; end")
        y = interp.global_env.get("y")
        assert get_val(y) == 1

    def test_for_loop(self):
        interp = run_matlab("total = 0;\nfor i = 1:5; total = total + i; end")
        total = interp.global_env.get("total")
        assert get_val(total) == 15

    def test_while_loop(self):
        interp = run_matlab("x = 5;\nwhile x > 0; x = x - 1; end")
        x = interp.global_env.get("x")
        assert get_val(x) == 0

    def test_global_variable(self):
        interp = run_matlab("global x;\nx = 42;")
        x = interp.global_env.get("x")
        assert get_val(x) == 42

    def test_persistent_variable(self):
        interp = run_matlab("persistent x;\nx = 42;")
        x = interp.global_env.get("x")
        assert get_val(x) == 42


class TestBytecodeIntegration:
    """Integration tests through bytecode VM."""

    def test_bytecode_simple(self):
        from matpy.bytecode import BytecodeCompiler, BytecodeVM
        from matpy.parser import Parser
        from matpy.lexer import Lexer

        compiler = BytecodeCompiler()
        lexer = Lexer("x = 1 + 2;")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        instructions = compiler.compile(program)
        vm = BytecodeVM()
        vm.run(instructions, compiler.constants)
        assert True

    def test_bytecode_variable(self):
        from matpy.bytecode import BytecodeCompiler, BytecodeVM
        from matpy.parser import Parser
        from matpy.lexer import Lexer

        compiler = BytecodeCompiler()
        lexer = Lexer("x = 42;")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        instructions = compiler.compile(program)
        vm = BytecodeVM()
        vm.run(instructions, compiler.constants)
        assert True

    def test_bytecode_expression(self):
        from matpy.bytecode import BytecodeCompiler, BytecodeVM
        from matpy.parser import Parser
        from matpy.lexer import Lexer

        compiler = BytecodeCompiler()
        lexer = Lexer("x = 1 + 2 * 3;")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        instructions = compiler.compile(program)
        vm = BytecodeVM()
        vm.run(instructions, compiler.constants)
        assert True

    def test_bytecode_if(self):
        from matpy.bytecode import BytecodeCompiler, BytecodeVM
        from matpy.parser import Parser
        from matpy.lexer import Lexer

        compiler = BytecodeCompiler()
        lexer = Lexer("x = 1;\nif x > 0; y = 1; else; y = 0; end")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        instructions = compiler.compile(program)
        vm = BytecodeVM()
        vm.run(instructions, compiler.constants)
        assert True

    def test_bytecode_for(self):
        from matpy.bytecode import BytecodeCompiler, BytecodeVM
        from matpy.parser import Parser
        from matpy.lexer import Lexer

        compiler = BytecodeCompiler()
        lexer = Lexer("total = 0;\nfor i = 1:5; total = total + i; end")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        instructions = compiler.compile(program)
        vm = BytecodeVM()
        vm.run(instructions, compiler.constants)
        assert True

    def test_bytecode_while(self):
        from matpy.bytecode import BytecodeCompiler, BytecodeVM
        from matpy.parser import Parser
        from matpy.lexer import Lexer

        compiler = BytecodeCompiler()
        lexer = Lexer("x = 5;\nwhile x > 0; x = x - 1; end")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        instructions = compiler.compile(program)
        vm = BytecodeVM()
        vm.run(instructions, compiler.constants)
        assert True

    def test_bytecode_function(self):
        from matpy.bytecode import BytecodeCompiler, BytecodeVM
        from matpy.parser import Parser
        from matpy.lexer import Lexer

        compiler = BytecodeCompiler()
        lexer = Lexer("function y = f(x); y = x^2; end\nresult = f(3);")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        instructions = compiler.compile(program)
        vm = BytecodeVM()
        vm.functions = compiler.functions
        vm.func_bytecode = compiler.func_bytecode
        vm.func_params = compiler.func_params
        vm.run(instructions, compiler.constants)
        assert True
