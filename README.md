# MatPy v0.5.0 — Python实现的本地Matlab编译器/解释器

一个用Python从零构建的Matlab运行时环境，无需Matlab许可证即可在本地执行`.m`文件。

## 功能特性

### 核心语言
- ✅ 完整的词法分析器（支持所有Matlab运算符、字符串、注释、行续接）
- ✅ 递归下降语法分析器
- ✅ 树遍历解释器 + 字节码编译器
- ✅ 矩阵运算（基于numpy，1-based索引）
- ✅ 控制流（if/elseif/else, for, while, switch/case, try/catch）
- ✅ 函数定义和调用（支持多返回值）
- ✅ 逻辑索引 `A(A > 5)`
- ✅ 命令式语法（grid on, hold on等）
- ✅ eval/feval动态执行

### 数据类型
- ✅ 双精度矩阵（支持N-D数组）
- ✅ 字符串（单引号字符数组 + 双引号字符串数组）
- ✅ Cell数组和Struct
- ✅ 函数句柄和匿名函数
- ✅ classdef OOP（支持继承）
- ✅ 稀疏矩阵

### 内置函数（284个）

| 类别 | 数量 | 关键函数 |
|------|------|----------|
| 数学 | 60+ | sin, cos, exp, log, fzero, ode45, integral |
| 矩阵 | 80+ | zeros, ones, eye, inv, eig, svd, lu, qr |
| 统计 | 20+ | mean, std, cov, regress, ttest, anova1 |
| 信号处理 | 15+ | fft, filter, butter, freqz |
| 字符串 | 25+ | strcat, strcmp, regexp, string, upper |
| 绘图 | 30+ | plot, scatter, bar, surf, contour |
| 文件I/O | 20+ | load, save, csvread, h5read, jsonencode |
| 稀疏矩阵 | 15+ | sparse, full, eigs, svds |
| 数据结构 | 15+ | struct, cell, fieldnames |
| N-D数组 | 14 | permute, squeeze, circshift, rot90 |

## 安装

```bash
cd D:\mat
pip install -e .
```

## 使用方法

### 执行.m文件
```bash
python -m matpy script.m
```

### 交互式REPL
```bash
python -m matpy
```

### REPL命令
| 命令 | 功能 |
|------|------|
| `who` | 列出变量名 |
| `whos` | 列出变量详情 |
| `clear` | 清除所有变量 |
| `clc` | 清屏 |
| `demo` | 运行演示 |
| `help` | 显示帮助 |

## 示例

### 基本矩阵运算
```matlab
A = [1 2; 3 4];
B = [5 6; 7 8];
C = A + B;
D = A * B;
disp(C);
disp(D);
```

### Fibonacci序列
```matlab
n = 10;
f = zeros(1, n);
f(1) = 1;
f(2) = 1;
for i = 3:n
    f(i) = f(i-1) + f(i-2);
end
disp(f);
```

### 绘图
```matlab
x = 0:0.1:2*pi;
y = sin(x);
figure;
plot(x, y);
xlabel('x');
ylabel('sin(x)');
title('Sine Wave');
grid on;
saveas(gcf, 'sine.png');
```

### OOP类定义
```matlab
classdef Point
    properties
        x = 0
        y = 0
    end
    methods
        function obj = Point(x, y)
            obj.x = x;
            obj.y = y;
        end
        function d = distance(obj)
            d = sqrt(obj.x^2 + obj.y^2);
        end
    end
end

p = Point(3, 4);
disp(p.distance());  % 5.0
```

### 类继承
```matlab
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
disp(c.name);      % circle
disp(c.area());    % 78.53975
```

### 逻辑索引
```matlab
A = [1 2 3 4 5 6 7 8 9 10];
B = A(A > 5);      % [6 7 8 9 10]
A(A > 7) = 0;      % [1 2 3 4 5 6 7 0 0 0]
```

### 函数句柄和匿名函数
```matlab
f = @sin;
disp(f(0));        % 0.0

g = @(x, y) x.^2 + y.^2;
disp(g(3, 4));     % 25

result = feval(f, 1.5708);
disp(result);      % 1.0
```

### 稀疏矩阵
```matlab
A = speye(5);
disp(nnz(A));      % 5
B = full(A);
disp(B);
```

### N-D数组
```matlab
A = zeros(2, 3, 4);
B = squeeze(A(1, :, :));
C = permute(A, [2 1 3]);
```

### 统计和优化
```matlab
x = [1 2 3 4 5];
disp(mean(x));     % 3.0
disp(std(x));      % 1.5811

result = integral(@(x) x^2, 0, 1);
disp(result);      % 0.3333
```

## 项目结构

```
D:\mat\
├── pyproject.toml
├── README.md
├── MATPY_GAP_ANALYSIS.md
├── matpy/
│   ├── __init__.py          (v0.4.0)
│   ├── __main__.py          (CLI + REPL)
│   ├── lexer.py             (词法分析)
│   ├── tokens.py            (Token定义)
│   ├── parser.py            (语法分析)
│   ├── ast_nodes.py         (AST节点)
│   ├── interpreter.py       (解释器)
│   ├── bytecode.py          (字节码编译器+VM)
│   ├── environment.py       (作用域)
│   ├── runtime/
│   │   ├── types.py         (Mat, Struct, CellArray, ClassInstance, StringArray)
│   │   └── matrix.py        (矩阵操作)
│   └── builtins/            (10个内置函数模块, 284个函数)
│       ├── math.py
│       ├── matrix_ops.py
│       ├── io.py
│       ├── plotting.py
│       ├── data_struct.py
│       ├── string.py
│       ├── file_io.py
│       ├── advanced_math.py
│       ├── sparse.py
│       └── string_array.py
└── tests/
    ├── test_matpy.py        (43个单元测试)
    └── examples/            (13个示例)
```

## 测试

```bash
# 运行所有测试
pytest tests/test_matpy.py -v

# 运行示例
python -m matpy tests/examples/hello.m
```

## 依赖

- Python >= 3.10
- numpy >= 1.24
- matplotlib >= 3.7
- scipy >= 1.0 (可选，用于高级数学函数)

## 已知限制

- 不支持完整的eval/feval动态特性
- 不支持Simulink
- 不支持完整的MEX接口
- 部分内置函数参数处理简化
- 性能比原生MATLAB慢（纯解释器）

## 路线图

- [ ] 字节码VM完善（函数调用、完整控制流）
- [ ] JIT编译（基于numba或自定义）
- [ ] 更多内置函数（目标500+）
- [ ] 完整的.mat文件支持
- [ ] 性能基准测试
- [ ] IDE支持（Language Server Protocol）

## 许可证

MIT License
