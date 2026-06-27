# MatPy 对齐生产级 MATLAB 解释器 — 详细实施计划

> 基于代码审计结果，本文档定义了从当前 v0.3.0 原型到生产级 MATLAB 解释器的完整路线图。
> 每个任务标注具体文件、验证标准和工时估算。

---

## 一、当前状态量化

| 指标 | 当前值 | 生产级目标 | 差距 |
|------|--------|-----------|------|
| 内置函数数 | 478 (去重后约420) | 1,200+ | 2.5x |
| 语言覆盖率 | ~40% | ~90% | 2.2x |
| 测试用例数 | 103 | 2,000+ | 20x |
| 代码行数(核心) | ~4,500 | ~15,000 | 3.3x |
| 性能(循环密集) | 纯树遍历 | 字节码+JIT | 100x |
| .mat 文件支持 | 不支持 | 完整 | ∞ |
| 版本号一致性 | 3处不一致 | 统一 | 修复 |

### 已完成功能

| 模块 | 功能数 | 完成度 | 说明 |
|------|--------|--------|------|
| 核心语言 | - | 85% | 变量、运算符、控制流、函数、OOP |
| 数学函数 | 120+ | 70% | 基础数学、优化、积分、插值 |
| 矩阵操作 | 80+ | 75% | 基础操作、分解、稀疏矩阵 |
| 统计函数 | 60+ | 60% | 描述统计、分布、假设检验 |
| 字符串 | 30+ | 50% | 基础操作、正则表达式 |
| 绘图 | 40+ | 60% | 2D/3D绘图、基本定制 |
| 文件I/O | 25+ | 50% | CSV、JSON、.mat、HDF5 |
| 图像处理 | 10+ | 20% | 基础读写、简单处理 |
| 类型系统 | 15+ | 70% | 类型检查、转换 |

---

## 二、总体阶段划分

```
Phase 0: 工程基础 (1周)          ← 地基，必须先做
Phase 1: 语言核心补全 (3周)      ← 正确性优先
Phase 2: 字节码VM完善 (3周)      ← 性能关键路径
Phase 3: 内置函数扩展 (4周)      ← 功能覆盖
Phase 4: 运行时类型增强 (2周)    ← 高级数据类型
Phase 5: 工程质量与生态 (2周)    ← 可维护性
Phase 6: 高级特性 (3周)          ← 差异化
────────────────────────────────
总计: 约18周
```

---

## Phase 0: 工程基础 (第1周)

### 0.1 版本号统一

**问题**: `pyproject.toml` 写 `0.1.0`，`__init__.py` 写 `0.3.0`，`README.md` 写 `v0.4.0`

**任务**:
- [ ] `pyproject.toml` line 6: 改为 `version = "0.5.0"` (对齐后的首个正式版)
- [ ] `matpy/__init__.py` line 1: 改为 `__version__ = "0.5.0"`
- [ ] `README.md`: 版本号统一为 v0.5.0
- [ ] 添加版本管理策略: 每次发布同步更新三处

**验证**: `python -c "import matpy; print(matpy.__version__)"` 输出 `0.5.0`

### 0.2 清理 extra.py 重复函数

**问题**: `builtins/extra.py` 中 58 个函数与其他模块完全重复

**任务**:
- [ ] 编写脚本对比 `extra.py` 与其他模块的函数注册，生成重复清单
- [ ] 将 `extra.py` 中独有的函数（如有）迁移到对应模块
- [ ] 删除 `extra.py`，从 `builtins/__init__.py` 移除导入
- [ ] 运行全量测试确认无破坏

**验证**: `pytest tests/test_matpy.py -v` 全部通过，无重复注册警告

### 0.3 测试基础设施升级

**任务**:
- [ ] `pyproject.toml` 添加 pytest 配置:
  ```toml
  [tool.pytest.ini_options]
  testpaths = ["tests"]
  addopts = "-v --tb=short"
  ```
- [ ] 创建 `tests/conftest.py` 提供共享 fixture:
  ```python
  @pytest.fixture
  def interp():
      """创建干净的 Interpreter 实例"""
      return Interpreter()

  @pytest.fixture
  def run(interp):
      """执行 MATLAB 代码并返回解释器"""
      def _run(source):
          lexer = Lexer(source)
          tokens = lexer.tokenize()
          parser = Parser(tokens)
          program = parser.parse()
          interp.run(program)
          return interp
      return _run
  ```
- [ ] 将现有 `test_matpy.py` (658行) 拆分为模块化测试文件:
  - `tests/test_lexer.py`
  - `tests/test_parser.py`
  - `tests/test_interpreter.py`
  - `tests/test_builtins_math.py`
  - `tests/test_builtins_matrix.py`
  - `tests/test_builtins_string.py`
  - `tests/test_builtins_io.py`
  - `tests/test_builtins_plotting.py`
  - `tests/test_runtime_types.py`
  - `tests/test_examples.py` (集成测试，运行所有 .m 文件)

**验证**: `pytest --co` 列出 100+ 测试，`pytest -v` 全部通过

### 0.4 CI/CD 流水线

**任务**:
- [ ] 创建 `.github/workflows/test.yml`:
  ```yaml
  name: Tests
  on: [push, pull_request]
  jobs:
    test:
      runs-on: ubuntu-latest
      strategy:
        matrix:
          python-version: ["3.10", "3.11", "3.12"]
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
        - run: pip install -e ".[dev]"
        - run: pytest --tb=short -q
  ```
- [ ] 创建 `.github/workflows/lint.yml` (ruff 检查)

**验证**: 推送到 GitHub 后 CI 自动运行并绿色通过

---

## Phase 1: 语言核心补全 (第2-4周)

### 1.1 N-D 数组完整支持 [interpreter.py]

**问题**: 高维 for 循环被展平为 1D，`permute`/`shiftdim` 行为不完整

**任务**:
- [ ] `interpreter.py` `_exec_for` (约 line 225): 高维数组迭代改为沿第一维遍历:
  ```python
  # 当前: items = data.flatten() (错误)
  # 目标: 沿第1维切片，与MATLAB一致
  if data.ndim >= 2:
      items = [Mat(data[i, ...]) for i in range(data.shape[0])]
  ```
- [ ] `runtime/types.py` `Mat` 类: 添加 `__getitem__` 支持高维切片
- [ ] `builtins/matrix_ops.py`: 修复 `permute`, `shiftdim`, `ipermute` 对高维数组的行为
- [ ] 新增测试: `tests/test_ndim.py` 覆盖 3D/4D 数组的创建、索引、遍历、运算

**验证**:
```matlab
A = zeros(2,3,4);
for i = 1:2
    B = A(i,:,:);  % 应返回 1x3x4 而非展平
end
C = permute(A, [3 1 2]);  % 应返回 4x2x3
```

### 1.2 增强 varargin/varargout [interpreter.py]

**问题**: 当前实现基础可用，但缺 edge case 处理

**任务**:
- [ ] `interpreter.py` `_call_user_func`: 确保 `varargin` 正确处理传入的 cell 元素
- [ ] 支持 `varargout` 作为 cell 数组返回
- [ ] 支持 `narginchk`/`nargoutchk` 参数校验
- [ ] 新增测试: 嵌套 varargin、varargout 为空、nargin/nargout 边界

**验证**:
```matlab
function varargout = multi_out(x)
    varargout{1} = x;
    varargout{2} = x^2;
end
[a, b] = multi_out(3);  % a=3, b=9
```

### 1.3 命令式语法扩展 [parser.py]

**问题**: 仅支持 `grid`, `hold`, `box`, `shading`, `colormap` 5 个命令式调用

**任务**:
- [ ] `parser.py` `_parse_assign_or_expr` (约 line 389): 扩展命令式语法为通用机制:
  ```python
  # 任何 identifier 后跟 string/number args 都可作为命令式调用
  # disp hello  →  disp('hello')
  # load data.mat  →  load('data.mat')
  ```
- [ ] 添加配置列表或通用规则，而非硬编码 5 个函数名
- [ ] 新增测试: 覆盖各种命令式调用场景

**验证**: `load data.mat`, `disp hello`, `cd /tmp` 等命令式语法正常工作

### 1.4 switch/case 增强 [interpreter.py]

**任务**:
- [ ] 支持 cell 数组作为 case 值: `case {1, 2, 3}`
- [ ] 支持字符串匹配: `case 'hello'`
- [ ] 支持范围表达式: `case {1:5}` (需要先解析)
- [ ] 新增测试: 各种 switch/case 组合

### 1.5 try/catch 增强 [interpreter.py]

**任务**:
- [ ] 支持 `catch e` 捕获 MException 对象
- [ ] 支持 `e.message`, `e.identifier` 字段访问
- [ ] 支持 `rethrow(e)`
- [ ] 创建 `MException` 运行时类型: `runtime/types.py`
- [ ] 新增测试: 嵌套 try/catch、rethrow、MException 字段

**验证**:
```matlab
try
    x = 1/0;
catch e
    disp(e.message);
end
```

### 1.6 完善 classdef OOP [interpreter.py, parser.py]

**任务**:
- [ ] 支持 `Abstract` 类: `classdef (Abstract) Shape`
- [ ] 支持 `Sealed` 类: `classdef (Sealed) Circle`
- [ ] 支持 `Access` 属性: `properties (Access = private)`
- [ ] 支持 `Constant` 属性: `properties (Constant) PI = 3.14159`
- [ ] 支持 `events` 块 (基础): `events Tick, Click end`
- [ ] 支持 `enumeration` 块 (基础)
- [ ] 支持方法属性: `methods (Static)`, `methods (Access = private)`
- [ ] 新增测试: 各种 classdef 特性组合

**验证**:
```matlab
classdef (Abstract) Shape
    properties (Constant)
        PI = 3.14159
    end
    methods (Abstract)
        a = area(obj)
    end
end
```

### 1.7 嵌套函数与局部函数 [interpreter.py]

**任务**:
- [ ] 支持函数内定义函数 (嵌套函数，可访问外层变量)
- [ ] 支持同一文件内多个函数 (局部函数，仅主函数对外可见)
- [ ] 作用域规则: 嵌套函数可读写外层变量，局部函数不可
- [ ] 新增测试

---

## Phase 2: 字节码 VM 完善 (第5-7周)

### 2.1 完善 BytecodeCompiler [bytecode.py]

**当前状态**: 仅支持基本算术和内建函数调用，不支持用户函数、switch、try/catch

**任务**:
- [ ] 支持用户函数调用:
  - 添加 `CALL_USER_FUNC` opcode
  - 编译 `FuncDef` 时存储到 functions 字典
  - VM 中实现函数调用栈帧
- [ ] 支持 `SwitchStmt` 编译
- [ ] 支持 `TryCatchStmt` 编译 (异常处理机制)
- [ ] 支持 `INDEX_ASSIGN` (当前 opcode 存在但未编译)
- [ ] 支持 `GlobalStmt`, `PersistentStmt`
- [ ] 支持 `FuncHandle`, `AnonFuncExpr`
- [ ] 支持 `ClassDef` (至少构造函数)
- [ ] 支持多返回值赋值

**新增 opcode**:
```
CALL_USER_FUNC  arg=nargs     # 调用用户函数
PUSH_FRAME                     # 创建栈帧
POP_FRAME                      # 销毁栈帧
LOAD_UPVALUE    arg=name      # 加载闭包变量
STORE_UPVALUE   arg=name      # 存储闭包变量
TRY_BEGIN       arg=label     # try 块开始
TRY_END                        # try 块结束
CATCH_BEGIN                    # catch 块开始
SWITCH_START                   # switch 开始
CASE_CMP         arg=value     # 比较 case 值
```

### 2.2 接入主流程 [__main__.py]

**任务**:
- [ ] `__main__.py` 添加 `--bytecode` 命令行参数
- [ ] `run_source` 函数支持选择执行引擎:
  ```python
  def run_source(source, filename="<stdin>", engine="tree"):
      lexer = Lexer(source)
      tokens = lexer.tokenize()
      parser = Parser(tokens)
      program = parser.parse()
      if engine == "bytecode":
          compiler = BytecodeCompiler()
          instructions, constants = compiler.compile(program)
          vm = BytecodeVM()
          return vm.run(instructions, constants)
      else:
          interp = Interpreter()
          interp.run(program)
          return interp
  ```
- [ ] REPL 支持切换引擎: `%engine bytecode`

### 2.3 性能基准测试 [tests/benchmark.py]

**任务**:
- [ ] 创建 `tests/benchmark.py`:
  ```python
  BENCHMARKS = {
      "fibonacci": """
          function f = fib(n)
              f = zeros(1, n);
              f(1) = 1; f(2) = 1;
              for i = 3:n
                  f(i) = f(i-1) + f(i-2);
              end
          end
          result = fib(1000);
      """,
      "matrix_multiply": """
          A = rand(100);
          B = rand(100);
          C = A * B;
      """,
      "loop_heavy": """
          s = 0;
          for i = 1:10000
              s = s + sin(i);
          end
      """,
  }
  ```
- [ ] 测量树遍历 vs 字节码执行时间
- [ ] 输出对比报告

**验证**: 字节码引擎在循环密集场景比树遍历快 5-10x

---

## Phase 3: 内置函数扩展 (第8-11周)

### 3.1 目标函数清单 (按优先级)

#### P0 — 核心数学 (第8周)

| 函数 | 模块 | 说明 |
|------|------|------|
| `interp2` | matrix_ops.py | **新增** 二维插值 |
| `interp3` | matrix_ops.py | **新增** 三维插值 |
| `griddata` | advanced_math.py | **新增** 散点插值 |
| `fzero` | advanced_math.py | 已有，验证正确性 |
| `fminsearch` | advanced_math.py | 已有，验证正确性 |
| `ode45` | advanced_math.py | 已有，验证正确性 |
| `integral` | advanced_math.py | 已有，验证正确性 |
| `polyfit` | advanced_math.py | 已有，验证正确性 |
| `interp1` | matrix_ops.py | 已有，验证正确性 |
| `ppval` | advanced_math.py | 已有，验证正确性 |
| `mkpp` | advanced_math.py | 已有，验证正确性 |

#### P1 — 线性代数 (第8-9周)

| 函数 | 模块 | 说明 |
|------|------|------|
| `schur` | matrix_ops.py | **新增** Schur 分解 |
| `hess` | matrix_ops.py | **新增** Hessenberg 分解 |
| `gsvd` | matrix_ops.py | **新增** 广义 SVD |
| `linsolve` | matrix_ops.py | **新增** 线性方程组求解 |
| `mldivide` | matrix_ops.py | **新增** `\` 左除的独立实现 |
| `mrdivide` | matrix_ops.py | **新增** `/` 右除的独立实现 |
| `lu` | matrix_ops.py | 已有，验证正确性 |
| `qr` | matrix_ops.py | 已有，验证正确性 |
| `chol` | advanced_math.py | 已有，验证正确性 |
| `expm` | advanced_math.py | 已有，验证正确性 |
| `logm` | advanced_math.py | 已有，验证正确性 |
| `sqrtm` | advanced_math.py | 已有，验证正确性 |

#### P2 — 统计 (第9-10周)

| 函数 | 模块 | 说明 |
|------|------|------|
| `mean`, `std`, `var` | advanced_math.py | 已有，验证多维支持 |
| `cov`, `corrcoef` | advanced_math.py | 已有，验证正确性 |
| `regress` | advanced_math.py | 已有，验证正确性 |
| `ttest`, `ttest2` | advanced_math.py | 已有，验证正确性 |
| `anova1` | advanced_math.py | 已有，验证正确性 |
| `pca` | statistics.py | 已有，验证正确性 |
| `kmeans` | statistics.py | 已有，验证正确性 |
| `fitlm` | statistics.py | 已有，验证正确性 |
| `glmfit` | statistics.py | 已有，验证正确性 |
| `stepwiselm` | statistics.py | 已有，验证正确性 |
| `bootstrp` | statistics.py | 已有，验证正确性 |
| `mle` | statistics.py | 已有，验证正确性 |

#### P3 — 信号处理 (第10周)

| 函数 | 模块 | 说明 |
|------|------|------|
| `fft`, `ifft` | matrix_ops.py | 已有，验证正确性 |
| `fft2`, `ifft2` | signal.py | 已有，验证正确性 |
| `butter`, `cheby1` | signal.py | 已有，验证正确性 |
| `freqz` | signal.py | 已有，验证正确性 |
| `fir1` | signal.py | 已有，验证正确性 |
| `filtfilt` | signal.py | 已有，验证正确性 |
| `spectrogram` | signal.py | 已有，验证正确性 |
| `findpeaks` | signal.py | 已有，验证正确性 |
| `resample` | signal.py | 已有，验证正确性 |
| `xcorr` | signal.py | 已有，验证正确性 |

#### P4 — 优化 (第10-11周)

| 函数 | 模块 | 说明 |
|------|------|------|
| `fmincon` | optimization.py | 已有，验证正确性 |
| `linprog` | optimization.py | 已有，验证正确性 |
| `quadprog` | optimization.py | 已有，验证正确性 |
| `ga` | optimization.py | 已有，验证正确性 |
| `particleswarm` | optimization.py | 已有，验证正确性 |
| `fsolve` | optimization.py | 已有，验证正确性 |
| `fminunc` | optimization.py | 已有，验证正确性 |
| `lsqnonlin` | optimization.py | 已有，验证正确性 |
| `intlinprog` | optimization.py | 已有，验证正确性 |

#### P5 — 文件 I/O (第11周)

| 函数 | 模块 | 说明 |
|------|------|------|
| `load`, `save` | file_io.py | 已有，**需增强 .mat 格式支持** |
| `readtable`, `writetable` | file_io.py | 已有，验证正确性 |
| `jsonencode`, `jsondecode` | file_io.py | 已有，验证正确性 |
| `h5read`, `h5write` | file_io.py | 已有，验证正确性 |
| `csvread`, `csvwrite` | file_io.py | 已有，验证正确性 |
| `fopen`, `fclose`, `fgetl` | file_io.py | 已有，验证正确性 |
| `matfile` | file_io.py | 已有，验证正确性 |

### 3.2 每个函数的验证流程

```
1. 编写 MATLAB 参考脚本 (在 MATLAB/Octave 中运行)
2. 编写 MatPy 测试用例 (期望相同输出)
3. 运行测试，对比结果
4. 修复差异
5. 添加到回归测试套件
```

### 3.3 .mat 文件格式支持 [file_io.py]

**任务**:
- [ ] 使用 `scipy.io.loadmat`/`savemat` 实现 .mat v5 格式读写
- [ ] 支持 .mat v7.3 (HDF5) 格式
- [ ] 处理 MATLAB 特有类型映射: struct → Struct, cell → CellArray
- [ ] 新增测试: 保存/加载各种数据类型

**验证**:
```matlab
A = [1 2; 3 4];
save('test.mat', 'A');
clear A;
load('test.mat');
disp(A);  % 应输出 [1 2; 3 4]
```

### 3.4 控制系统工具箱补全 [control.py]

| 函数 | 说明 | 状态 |
|------|------|------|
| `tf` | 传递函数模型 | 已有，验证正确性 |
| `ss` | 状态空间模型 | 已有，验证正确性 |
| `zpk` | 零极点增益模型 | 已有，验证正确性 |
| `pid` | PID控制器 | 已有，验证正确性 |
| `bode` | 波特图 | 已有，验证正确性 |
| `step` | 阶跃响应 | 已有，验证正确性 |
| `feedback` | 反馈连接 | 已有，验证正确性 |
| `pole`, `zero` | 极点/零点 | 已有，验证正确性 |
| `impulse` | 脉冲响应 | **新增** |
| `lsim` | 线性仿真 | **新增** |
| `rlocus` | 根轨迹 | **新增** |
| `c2d`, `d2c` | 连续/离散转换 | 已有，验证正确性 |

### 3.5 图像处理工具箱补全 [image.py]

| 函数 | 说明 | 状态 |
|------|------|------|
| `imread`, `imwrite`, `imshow` | I/O | 已有，验证正确性 |
| `rgb2gray` | 颜色转换 | 已有，验证正确性 |
| `imresize`, `imrotate`, `imcrop` | 几何变换 | 已有，验证正确性 |
| `edge` | 边缘检测 | 已有，验证正确性 |
| `imhist`, `histeq`, `imadjust` | 增强 | 已有，验证正确性 |
| `medfilt2`, `imgaussfilt` | 滤波 | 已有，验证正确性 |
| `imerode`, `imdilate`, `imopen`, `imclose` | 形态学 | 已有，验证正确性 |
| `imbinarize` | 二值化 | 已有，验证正确性 |
| `imnoise` | 噪声添加 | 已有，验证正确性 |
| `strel` | 结构元素 | 已有，验证正确性 |

---

## Phase 4: 运行时类型增强 (第12-13周)

### 4.1 table/timetable 数据类型

**任务**:
- [ ] `runtime/types.py` 新增 `Table` 类:
  ```python
  class Table:
      """MATLAB table 数据类型"""
      def __init__(self, data: dict[str, np.ndarray], row_names=None):
          self._data = data  # column_name -> array
          self._row_names = row_names

      @property
      def Variables(self):
          return list(self._data.keys())

      @property
      def Height(self):
          return len(next(iter(self._data.values())))

      @property
      def Width(self):
          return len(self._data)
  ```
- [ ] 支持 `T.colname` 点访问语法 (interpreter.py)
- [ ] 支持 `T(1:3, :)` 行索引
- [ ] 支持 `T{:, 'colname'}` 花括号提取数据
- [ ] 内置函数: `table`, `array2table`, `table2array`, `readtable`, `writetable`
- [ ] 新增测试: table 创建、索引、合并、I/O

### 4.2 containers.Map 类型

**任务**:
- [ ] `runtime/types.py` 新增 `Map` 类:
  ```python
  class Map:
      """MATLAB containers.Map"""
      def __init__(self, keys=None, values=None, key_type='any', value_type='any'):
          self._data = {}
          self._key_type = key_type
          self._value_type = value_type
  ```
- [ ] 支持 `map(key)` 和 `map(key) = value` 语法
- [ ] 内置函数: `keys`, `values`, `isKey`, `remove`
- [ ] 新增测试

### 4.3 datetime/duration 类型

**任务**:
- [ ] `runtime/types.py` 新增 `Datetime` 和 `Duration` 类
- [ ] 内置函数: `datetime`, `duration`, `years`, `days`, `hours`, `minutes`, `seconds`
- [ ] 支持 datetime 算术: 日期差 = duration
- [ ] 支持 `datestr`, `datevec`, `datenum` 转换
- [ ] 新增测试

### 4.4 categorical 类型

**任务**:
- [ ] `runtime/types.py` 新增 `Categorical` 类
- [ ] 内置函数: `categorical`, `categories`, `isundefined`
- [ ] 支持 categorical 比较和分组
- [ ] 新增测试

---

## Phase 5: 工程质量与生态 (第14-15周)

### 5.1 interpreter.py 拆分

**问题**: 840 行单文件，职责过多

**任务**:
- [ ] 拆分为:
  - `interpreter.py` — 主类，`run()` 入口，`_exec_stmt` 分发 (~200行)
  - `evaluator.py` — 表达式求值 `_eval`, `_eval_binop`, `_eval_index` (~300行)
  - `statement_handler.py` — 语句执行 `_exec_if`, `_exec_for`, `_exec_while` (~200行)
  - `func_resolver.py` — 函数调用解析 `_eval_func_call`, `_call_user_func` (~150行)
- [ ] 保持 `Interpreter` 类的公共 API 不变 (向后兼容)
- [ ] 全量测试通过

### 5.2 错误处理标准化

**任务**:
- [ ] 统一错误层次结构:
  ```
  MatPyError
  ├── MatPySyntaxError      (解析阶段)
  ├── MatPyTypeError         (类型错误)
  ├── MatPyIndexError        (索引越界)
  ├── MatPyNameError         (未定义变量)
  ├── MatPyValueError        (值域错误)
  ├── MatPyDimensionError    (维度不匹配)
  ├── MatPyMemoryError       (内存不足)
  └── MatPyNotImplementedError (未实现特性)
  ```
- [ ] 所有错误包含: 文件名、行号、列号、错误标识符
- [ ] 支持 `lasterror` 函数获取最近错误
- [ ] 支持 `warning` 函数和 `warning('off', 'id')` 控制

### 5.3 REPL 增强 [__main__.py]

**任务**:
- [ ] 支持 Tab 补全 (基于变量名和函数名)
- [ ] 支持多行编辑 (不依赖 `end` 关键字判断)
- [ ] 支持 `%` magic 命令: `%time expr`, `%profile on/off`
- [ ] 支持 `dbstop`/`dbstep`/`dbcont`/`dbquit` 基本调试
- [ ] 支持 `doc funcname` 显示函数文档
- [ ] 历史命令搜索 (Ctrl+R)

### 5.4 文档系统

**任务**:
- [ ] 为每个内置函数添加 docstring (MATLAB help 格式):
  ```python
  @register("zeros")
  def mat_zeros(*args):
      """
      ZEROS Zeros array.
          ZEROS(N) is an N-by-N matrix of zeros.
          ZEROS(M,N) is an M-by-N matrix of zeros.
          ...
      """
  ```
- [ ] `help funcname` 在 REPL 中显示文档
- [ ] 生成 API 参考文档 (Sphinx/MkDocs)

### 5.5 包管理 [pyproject.toml]

**任务**:
- [ ] 添加 `scipy` 为必需依赖 (已部分使用)
- [ ] 添加 `h5py` 为可选依赖 (HDF5 支持)
- [ ] 添加 `pandas` 为可选依赖 (table 支持)
- [ ] 更新 `pyproject.toml`:
  ```toml
  dependencies = [
      "numpy>=1.24",
      "matplotlib>=3.7",
      "scipy>=1.10",
  ]

  [project.optional-dependencies]
  hdf5 = ["h5py>=3.8"]
  tables = ["pandas>=2.0"]
  full = ["h5py>=3.8", "pandas>=2.0"]
  dev = ["pytest>=7.0", "pytest-cov", "ruff"]
  ```

---

## Phase 6: 高级特性 (第16-18周)

### 6.1 并行计算基础

**任务**:
- [ ] `parfor` 循环: 编译为 Python `multiprocessing.Pool.map`
- [ ] `parfeval`: 异步函数执行
- [ ] `parallel.pool.Constant`: 跨 worker 共享数据
- [ ] 限制: 不支持分布式，仅单机多核

### 6.2 MEX 接口基础

**任务**:
- [ ] 支持调用 Python 函数作为 MEX 替代:
  ```matlab
  mex_func = py.mymodule.my_function;
  result = mex_func(args);
  ```
- [ ] 支持 `py.importlib.import_module` 导入 Python 模块
- [ ] 数据类型自动转换: Mat ↔ numpy.ndarray, string ↔ str

### 6.3 包/命名空间

**任务**:
- [ ] 支持 `+package` 目录结构:
  ```
  +mypackage/
      myfunc.m
      +subpkg/
          helper.m
  ```
- [ ] 支持 `import mypackage.myfunc` 语法
- [ ] 支持 `mypackage.myfunc()` 调用
- [ ] 函数搜索路径包含 `+` 包

### 6.4 JIT 编译探索

**任务**:
- [ ] 热点检测: 识别频繁执行的代码路径
- [ ] 类型推断: 推断变量类型以优化
- [ ] 评估 numba/Cranelift 集成可行性
- [ ] 编译结果缓存机制

---

## 三、测试策略

### 测试覆盖目标

| 阶段 | 测试数目标 | 覆盖率目标 |
|------|-----------|-----------|
| Phase 0 | 150 | 70% |
| Phase 1 | 300 | 75% |
| Phase 2 | 500 | 80% |
| Phase 3 | 1,000 | 80% |
| Phase 4 | 1,500 | 85% |
| Phase 5 | 1,800 | 85% |
| Phase 6 | 2,000+ | 90% |

### 测试目录结构

```
tests/
├── unit/                  # 单元测试
│   ├── test_lexer.py
│   ├── test_parser.py
│   ├── test_interpreter.py
│   ├── test_bytecode.py
│   ├── test_runtime.py
│   └── test_builtins/
│       ├── test_math.py
│       ├── test_matrix.py
│       ├── test_string.py
│       ├── test_io.py
│       ├── test_plotting.py
│       ├── test_signal.py
│       ├── test_statistics.py
│       ├── test_optimization.py
│       ├── test_control.py
│       └── test_image.py
├── integration/           # 集成测试
│   ├── test_examples.py   # 运行所有 .m 示例
│   ├── test_mat_file.py   # .mat 文件读写
│   └── test_repl.py       # REPL 交互测试
├── benchmark/             # 性能基准
│   ├── bench_fibonacci.py
│   ├── bench_matrix.py
│   └── bench_loop.py
├── conftest.py
└── examples/              # MATLAB 示例脚本
```

### 回归测试策略

每个 Phase 完成后:
1. 运行全量测试: `pytest tests/ -v --tb=short`
2. 运行所有示例脚本: `python -m matpy tests/examples/*.m`
3. 运行基准测试: `python tests/benchmark/`
4. 检查覆盖率: `pytest --cov=matpy --cov-report=html`
5. 确认无性能退化

---

## 四、里程碑与验收标准

### M1: 工程基础完成 (第1周末)

- [ ] 版本号统一为 0.5.0
- [ ] extra.py 清理完成
- [ ] 测试基础设施就绪
- [ ] CI 流水线运行通过
- [ ] 测试数 ≥ 150

### M2: 语言核心补全 (第4周末)

- [ ] N-D 数组完整支持
- [ ] varargin/varargout 增强
- [ ] 命令式语法扩展
- [ ] classdef 高级特性
- [ ] try/catch MException
- [ ] 嵌套函数/局部函数
- [ ] 测试数 ≥ 300

### M3: 字节码 VM 可用 (第7周末)

- [ ] 字节码引擎支持所有语句类型
- [ ] 字节码引擎支持用户函数调用
- [ ] `--bytecode` 命令行参数可用
- [ ] 循环密集场景性能提升 5x+
- [ ] 基准测试套件就绪
- [ ] 测试数 ≥ 500

### M4: 内置函数扩展完成 (第11周末)

- [ ] 内置函数数 ≥ 600 (去重后)
- [ ] 所有 P0-P5 函数有测试覆盖
- [ ] .mat 文件读写可用
- [ ] 控制系统/图像处理工具箱补全
- [ ] 测试数 ≥ 1,000

### M5: 运行时类型增强 (第13周末)

- [ ] table/timetable 可用
- [ ] containers.Map 可用
- [ ] datetime/duration 可用
- [ ] categorical 可用
- [ ] 测试数 ≥ 1,500

### M6: 工程质量达标 (第15周末)

- [ ] interpreter.py 拆分完成
- [ ] 错误处理标准化
- [ ] REPL 增强完成
- [ ] 文档系统就绪
- [ ] 测试数 ≥ 1,800
- [ ] 覆盖率 ≥ 85%

### M7: 生产级发布 (第18周末)

- [ ] 并行计算基础可用
- [ ] MEX 接口可用
- [ ] 包/命名空间支持
- [ ] 内置函数数 ≥ 1,200
- [ ] 测试数 ≥ 2,000
- [ ] 覆盖率 ≥ 90%
- [ ] 性能基准报告发布
- [ ] v1.0.0 正式发布

---

## 五、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 字节码 VM 工作量超预期 | Phase 2 延期 | 优先保证树遍历正确性，字节码作为优化可选 |
| 内置函数数值精度问题 | 测试失败 | 与 MATLAB/Octave 对比，设置合理误差容限 |
| classdef 完整实现复杂 | Phase 1 延期 | 先实现 80% 场景，复杂特性标记为 NotSupported |
| .mat v7.3 HDF5 兼容性 | I/O 功能受限 | 依赖 scipy/h5py，测试多种 MATLAB 版本生成的文件 |
| 性能优化效果不显著 | 用户体验差 | 优先优化热点路径，非热点保持树遍历 |

---

## 六、依赖路线图

```
Phase 0 (工程基础)
    │
    ├─→ Phase 1 (语言核心) ──→ Phase 4 (运行时类型)
    │                              │
    ├─→ Phase 2 (字节码VM)         │
    │       │                      │
    │       └─→ Phase 3 (内置函数) ←┘
    │               │
    │               └─→ Phase 5 (工程质量)
    │                       │
    │                       └─→ Phase 6 (高级特性)
    │
    └─→ 所有 Phase 依赖 Phase 0 完成
```

---

## 七、每日/每周节奏

### 每日
- 早上: 查看前日测试结果，修复失败
- 上午: 实现新功能
- 下午: 编写测试，运行基准
- 晚上: 代码审查，提交

### 每周
- 周一: 规划本周任务，回顾上周进度
- 周三: 中期检查，调整优先级
- 周五: 里程碑验收，发布周报

### 每个 Phase 结束
- 全量回归测试
- 性能基准对比
- 更新 MATPY_GAP_ANALYSIS.md
- 更新 README.md
- Git tag 发布

---

## 八、成功指标

### 功能指标
- [ ] 能运行 80% 的 MATLAB 基础教程代码
- [ ] 能运行 Octave Forge 包中的 50% 示例
- [ ] 内置函数数 ≥ 1,200

### 性能指标
- [ ] 启动时间 < 0.3s
- [ ] 矩阵运算性能 ≥ MATLAB 的 30% (NumPy 后端)
- [ ] 循环密集场景 ≥ MATLAB 的 10% (字节码引擎)
- [ ] 内存使用 < MATLAB 的 2x

### 质量指标
- [ ] 测试覆盖率 ≥ 90%
- [ ] 无已知崩溃 (crash)
- [ ] 错误消息可读、可定位
- [ ] 文档覆盖所有公共 API
