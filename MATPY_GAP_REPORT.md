# MatPy v0.5.0 对标生产级 MATLAB 解释器 — 差距报告与对齐方案

> 生成日期: 2026-06-25
> 基于全量代码审计 + 504 项测试验证

---

## 一、量化总览

| 维度 | MatPy 当前 | MATLAB/Octave 参考 | 差距倍数 | 评级 |
|------|-----------|-------------------|---------|------|
| 内置函数数 | 649 (去重后) | MATLAB 1,500+ / Octave 1,000+ | 2.3x | 🟡 |
| 语言覆盖率 | ~55% | MATLAB ~100% / Octave ~90% | 1.8x | 🟡 |
| 测试用例数 | 504 | Octave 10,000+ | 20x | 🔴 |
| 核心代码行 | 12,183 | Octave 200,000+ | 16x | 🔴 |
| 执行性能 | 纯树遍历 | MATLAB JIT / Octave 解释器 | 50-100x | 🔴 |
| .mat 文件 | v5 读写 (scipy) | 完整 v5/v7.3 | 部分 | 🟡 |
| 运行时类型 | 14 种 | 30+ 种 | 2x | 🟡 |
| 工程质量 | 基础 | 生产级 | — | 🔴 |

---

## 二、逐维度差距详析

### 2.1 语言核心

#### ✅ 已实现 (覆盖约 55%)

| 类别 | 特性 | 状态 |
|------|------|------|
| 基础语法 | 变量赋值、算术/比较/逻辑运算符 | ✅ 完整 |
| 矩阵字面量 | `[1 2; 3 4]`、N-D 构造 | ✅ 完整 |
| 控制流 | if/elseif/else, for, while, switch/case, try/catch | ✅ 完整 |
| 函数 | 单/多返回值、varargin/varargout、嵌套函数、局部函数 | ✅ 完整 |
| OOP | classdef 单继承、属性、方法、构造函数 | ✅ 基础 |
| 索引 | 1-based、`end` 关键字、逻辑索引、线性索引 | ✅ 完整 |
| 运算符 | 全部算术/元素-wise/比较/逻辑运算符 | ✅ 完整 |
| 函数句柄 | `@func`、匿名函数 `@(x) x^2` | ✅ 完整 |
| 动态执行 | `eval`、`feval` | ✅ 基础 |
| 命令式语法 | `disp hello`、`grid on` 等 ~40 个 | ✅ 部分 |
| 全局/持久变量 | `global`、`persistent` 关键字 | ✅ 基础 |
| 转置 | `'` (共轭)、`.'` (非共轭) | ✅ 完整 |

#### ❌ 未实现 (差距约 45%)

| 特性 | MATLAB 行为 | MatPy 现状 | 优先级 | 影响面 |
|------|------------|-----------|--------|--------|
| `arguments` 块 | 函数参数验证 (类型、大小、默认值) | ❌ 未解析 | P1 | 高 — 现代 MATLAB 核心特性 |
| 字符串插值 | `"Hello " + name` 或 `sprintf` 隐式 | ❌ 仅 `sprintf` | P2 | 中 |
| 属性方法 | `properties (Access=private, Constant)` | ❌ 仅解析 `properties` 块 | P1 | 高 — classdef 必需 |
| `methods (Static)` | 静态方法 `ClassName.method()` | ❌ 未实现 | P1 | 高 |
| `enumeration` 块 | `classdef Color {Red, Green, Blue}` | ❌ 未解析 | P2 | 中 |
| `events` 块 | 事件驱动编程 | ❌ 未解析 | P3 | 低 |
| `rethrow` | 重新抛出异常 | ❌ 未实现 | P1 | 中 |
| 表达式索引 `end` | `A(end-1)` 在赋值右侧 | 🟡 部分 | P1 | 高 |
| `switch` 范围 | `case {1:5}` | ❌ 未实现 | P2 | 低 |
| 通用命令式语法 | 任何函数都可用命令式调用 | 🟡 仅 40 个硬编码 | P2 | 中 |
| `parfor` | 并行 for 循环 | 🟡 存根，不可用 | P3 | 低 |
| 包/命名空间 | `+package` 目录 | 🟡 部分加载，不完整 | P2 | 中 |
| `table` 行索引 | `T(1:3, :)`、`T{:, 'col'}` | ❌ 仅列访问 | P1 | 高 |
| 属性验证 | `properties x {mustBeNumeric}` | ❌ 未实现 | P2 | 中 |
| `try/catch` 嵌套 | 多层嵌套 + MException 链 | 🟡 基础可用 | P2 | 低 |
| 函数签名重载 | 同名不同参数数量 | ❌ 未实现 | P3 | 低 |

### 2.2 内置函数

#### 当前统计

- **注册条目**: 649 个
- **覆盖模块**: 16 个
- **去重后实际函数**: ~580 个 (部分跨模块重复注册)

#### 按模块覆盖度

| 模块 | 注册数 | MATLAB 参考 | 覆盖率 | 缺失关键函数 |
|------|--------|-----------|--------|-------------|
| math.py | 50 | ~80 | 63% | cart2pol, pol2cart, nextpow2, airy, besselj |
| matrix_ops.py | 95 | ~120 | 79% | gsvd, balance, planerot, gallery 增强 |
| advanced_math.py | 60 | ~80 | 75% | griddata, pchip 增强, ode113, bvp4c 增强 |
| statistics.py | 75 | ~150 | 50% | fitcsvm, fitctree, fitcensemble, glmval |
| signal.py | 30 | ~80 | 38% | czt, goertzel, firpm, fdesign, dsp 系统 |
| optimization.py | 25 | ~50 | 50% | surrogateopt, paretosearch, gamultiobj |
| control.py | 25 | ~60 | 42% | lqr, kalman, estim, augstate, connect |
| io.py | 30 | ~50 | 60% | inputParser 增强, onCleanup 增强 |
| plotting.py | 35 | ~80 | 44% | heatmap, polarplot, tiledlayout, animatedline |
| data_struct.py | 20 | ~40 | 50% | structfun 增强, cellfun 增强, table 增强 |
| string.py | 30 | ~60 | 50% | string 数组操作增强, compose 增强 |
| file_io.py | 25 | ~50 | 50% | .mat v7.3, xmlread, json 增强 |
| common.py | 90 | ~120 | 75% | datetime 增强, calendarDuration 增强 |
| sparse.py | 25 | ~40 | 63% | spaugment, eigs 增强, ichol 增强 |
| image.py | 30 | ~80 | 38% | imhist3, imsegkmeans, regionprops3, strel 增强 |
| string_array.py | 7 | ~20 | 35% | join, split, matches, replace 增强 |

#### 未覆盖的 MATLAB 工具箱

| 工具箱 | 函数量 | MatPy 状态 | 优先级 |
|--------|--------|-----------|--------|
| Symbolic Math | 200+ | ❌ 完全缺失 | P3 |
| Simulink | — | ❌ 不适用 | — |
| Communications | 100+ | ❌ 完全缺失 | P3 |
| DSP System | 150+ | ❌ 完全缺失 | P2 |
| Robotics | 100+ | ❌ 完全缺失 | P3 |
| Deep Learning | 100+ | ❌ 完全缺失 | P3 |
| Financial | 100+ | ❌ 完全缺失 | P3 |
| Audio | 50+ | ❌ 完全缺失 | P3 |
| Wavelet | 30+ | ❌ 完全缺失 | P3 |
| Mapping | 50+ | ❌ 完全缺失 | P3 |

### 2.3 运行时类型系统

#### ✅ 已实现 (14 种)

| 类型 | 行数 | 完整度 | 说明 |
|------|------|--------|------|
| `Mat` | 120 | 🟢 90% | numpy 包装，1-based 索引，稀疏支持 |
| `CellArray` | 50 | 🟡 70% | 2D 异构容器，缺高维支持 |
| `Struct` | 40 | 🟡 70% | 动态字段，缺嵌套赋值优化 |
| `FuncHandle` | 20 | 🟢 90% | 函数引用 + 闭包 |
| `StringArray` | 40 | 🟡 60% | 双引号字符串，缺完整 string 操作 |
| `ClassDefRuntime` | 30 | 🟡 60% | 类元数据，缺属性验证 |
| `ClassInstance` | 40 | 🟡 60% | 实例，缺 subsref/subsasgn 重载 |
| `MException` | 20 | 🟡 70% | 异常对象，缺 stack trace |
| `Table` | 50 | 🔴 40% | 列式数据，缺行索引、花括号提取 |
| `Map` | 30 | 🟡 70% | 键值容器 |
| `Datetime` | 40 | 🟡 60% | 日期时间，缺格式化、时区 |
| `Duration` | 30 | 🟡 60% | 时间段，缺 calendarDuration 集成 |
| `CalendarDuration` | 15 | 🟡 50% | 日历时间段 |
| `Categorical` | 30 | 🟡 60% | 分类数组，缺 ordinal 支持 |

#### ❌ 未实现

| 类型 | 说明 | 优先级 |
|------|------|--------|
| `timetable` | 时间索引表 | P2 |
| `graph` / `digraph` | 图数据结构 | P3 |
| `dictionary` | R2022b+ 新类型 | P3 |
| `string` (标量) | 独立字符串类型 | P1 |
| `int8/16/32/64` | 显式整数类型 | P1 |
| `uint8/16/32/64` | 无符号整数 | P1 |
| `single` | 单精度浮点 | P2 |
| `logical` (独立) | 独立布尔类型 (非 0/1) | P1 |
| `fi` | 定点数 | P3 |
| `gpuArray` | GPU 数组 | P3 |

### 2.4 执行引擎

#### 树遍历解释器 (当前唯一完整引擎)

| 指标 | 状态 |
|------|------|
| 语句覆盖 | 14/14 类型 ✅ |
| 表达式覆盖 | 16/16 类型 ✅ |
| 函数调用 | 6 级优先级解析 ✅ |
| 闭包支持 | 环境捕获 ✅ |
| 性能 | 循环密集场景慢 50-100x 🔴 |

#### 字节码 VM (部分可用)

| 特性 | 状态 | 说明 |
|------|------|------|
| 基本算术 | ✅ | +, -, *, /, ^ |
| 变量存取 | ✅ | STORE_VAR, LOAD_VAR |
| 控制流 | ✅ | JUMP, JUMP_IF_FALSE |
| 矩阵构建 | ✅ | BUILD_MATRIX |
| 内置调用 | ✅ | CALL opcode |
| 用户函数 | 🔴 | 回退到树遍历 |
| 闭包 | 🔴 | 不支持 |
| classdef | 🔴 | 不支持 |
| switch/case | 🟡 | 编译存在，VM 处理不完整 |
| try/catch | 🟡 | 标记存在，未真正集成 |
| INDEX_ASSIGN | 🟡 | opcode 存在，编译不完整 |
| global/persistent | 🔴 | VM 中未实现 |

### 2.5 工程质量

| 维度 | 当前状态 | 生产级要求 | 差距 |
|------|---------|-----------|------|
| 测试用例 | 504 | 2,000+ | 4x |
| 测试覆盖率 | 未测量 | ≥85% | — |
| 错误消息 | 基础 (行号+列号) | 文件名+行号+列号+标识符 | 🟡 |
| REPL 功能 | 历史、多行、%time/%who | Tab补全、调试器、文档 | 🔴 |
| CI/CD | 无 | GitHub Actions 自动测试 | 🔴 |
| 文档 | README 中文 | API 文档 + 函数帮助 | 🔴 |
| 代码组织 | interpreter.py 1000行 | 模块化拆分 | 🟡 |
| 版本管理 | REPL 横幅 v0.3.0 vs 实际 v0.5.0 | 统一 | 🔴 Bug |
| 依赖管理 | 仅 numpy/matplotlib | scipy/pandas/h5py 声明 | 🟡 |

### 2.6 已知 Bug

| # | 文件 | 行号 | 问题 | 严重度 |
|---|------|------|------|--------|
| 1 | `__main__.py` | 86 | REPL 横幅打印 `v0.3.0`，应为 `v0.5.0` | 低 |
| 2 | `matrix_ops.py` | 328 | `isKindOfClass` 应为 `isinstance` (拼写错误) | 高 |
| 3 | `statistics.py` | — | 依赖 `sklearn` 但未声明在 `pyproject.toml` | 中 |
| 4 | `image.py` | — | 依赖 `skimage` 但未声明在 `pyproject.toml` | 中 |
| 5 | 多模块 | — | 649 个注册中有部分跨模块重复 (后注册覆盖前注册) | 中 |
| 6 | `control.py` | — | step 响应使用脉冲响应积分近似，非状态空间仿真 | 中 |

---

## 三、与同类项目对比

| 维度 | MatPy v0.5.0 | GNU Octave | RunMat | MATLAB |
|------|-------------|-----------|--------|--------|
| 语言 | Python | C++ | Rust | C/C++ |
| 执行 | 树遍历 | 解释器 | JIT+解释器 | JIT |
| 内置函数 | ~580 | 1,000+ | 400+ | 1,500+ |
| 语言覆盖 | ~55% | ~90% | ~95% | 100% |
| .mat 支持 | v5 | 完整 | 部分 | 完整 |
| 工具箱 | 部分 (7个) | Forge 生态 | 无 | 完整生态 |
| 性能 | 慢 50-100x | 慢 5-10x | 接近原生 | 基准 |
| 许可 | MIT | GPL | MIT | 商业 |
| 优势 | Python 生态集成 | 成熟稳定 | 高性能 | 标准 |
| 劣势 | 性能、完整性 | 语法老旧 | 生态缺失 | 价格 |

---

## 四、对齐方案

### 总体路线图

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 0: 工程修复 (1周)                                         │
│  ├─ Bug 修复 (6个已知)                                           │
│  ├─ 依赖声明规范化                                                │
│  └─ CI/CD 流水线                                                 │
├─────────────────────────────────────────────────────────────────┤
│ Phase 1: 语言核心补全 (3周)                                      │
│  ├─ classdef 高级特性 (Static, Access, Constant, Abstract)       │
│  ├─ arguments 参数验证块                                         │
│  ├─ table 行索引/花括号提取                                       │
│  ├─ 通用命令式语法                                                │
│  └─ rethrow / MException 增强                                    │
├─────────────────────────────────────────────────────────────────┤
│ Phase 2: 字节码 VM 完善 (3周)                                    │
│  ├─ 用户函数调用 (原生栈帧)                                       │
│  ├─ 闭包支持                                                     │
│  ├─ switch/try-catch 完整支持                                    │
│  ├─ --bytecode CLI 集成                                          │
│  └─ 性能基准测试                                                  │
├─────────────────────────────────────────────────────────────────┤
│ Phase 3: 内置函数扩展 (4周)                                      │
│  ├─ 去重清理 + 缺失核心函数补齐                                    │
│  ├─ 信号处理/DSP 增强                                             │
│  ├─ 控制系统工具箱补全                                             │
│  ├─ 图像处理增强                                                  │
│  ├─ 统计/ML 函数扩展                                              │
│  └─ .mat v7.3 (HDF5) 支持                                       │
├─────────────────────────────────────────────────────────────────┤
│ Phase 4: 类型系统增强 (2周)                                      │
│  ├─ 显式整数类型 (int8/16/32/64, uint*)                          │
│  ├─ logical 独立类型                                              │
│  ├─ single 单精度                                                 │
│  ├─ timetable                                                     │
│  └─ datetime/duration 增强                                       │
├─────────────────────────────────────────────────────────────────┤
│ Phase 5: 工程质量 (2周)                                          │
│  ├─ interpreter.py 模块化拆分                                     │
│  ├─ 错误处理标准化                                                │
│  ├─ REPL 增强 (Tab补全、调试器)                                   │
│  ├─ 文档系统                                                      │
│  └─ 测试扩展到 1500+                                              │
├─────────────────────────────────────────────────────────────────┤
│ Phase 6: 高级特性 (3周)                                          │
│  ├─ 并行计算 (parfor)                                            │
│  ├─ Python MEX 桥接                                               │
│  ├─ +package 命名空间                                             │
│  └─ JIT 探索 (热点检测 + 类型推断)                                │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 0: 工程修复 (第 1 周)

**目标**: 消除已知缺陷，建立工程基线

| 任务 | 文件 | 验证标准 |
|------|------|---------|
| 修复 REPL 版本号 | `__main__.py:86` | `v0.3.0` → `v0.5.0` |
| 修复 isinstance 拼写 | `matrix_ops.py:328` | `isKindOfClass` → `isinstance` |
| 声明 scipy 依赖 | `pyproject.toml` | `scipy>=1.10` 加入 dependencies |
| 声明可选依赖 | `pyproject.toml` | `[project.optional-dependencies]` 块 |
| 去重内置函数注册 | `builtins/` 各模块 | 无重复 key 警告 |
| GitHub Actions CI | `.github/workflows/test.yml` | push 后自动运行 pytest |
| 测试覆盖率基线 | `pyproject.toml` | `--cov` 报告生成 |

### Phase 1: 语言核心补全 (第 2-4 周)

**目标**: 语言覆盖率从 55% 提升到 75%

#### 1.1 classdef 高级特性

```matlab
classdef (Abstract) Shape < handle
    properties (Constant, Access = private)
        PI = 3.14159
    end
    properties (Access = public)
        Name string
    end
    methods (Abstract)
        a = area(obj)
    end
    methods (Static)
        function s = describe()
            s = 'Shape base class';
        end
    end
end
```

**实现要点**:
- `parser.py`: 解析属性列表 `(Access=private, Constant, Abstract)`
- `interpreter.py`: `_exec_classdef` 处理属性元数据
- `runtime/types.py`: `ClassDefRuntime` 扩展属性验证

#### 1.2 arguments 参数验证块

```matlab
function result = myFunc(x, y, opts)
    arguments
        x double {mustBeNumeric}
        y double {mustBePositive}
        opts.Method char = 'default'
        opts.Tolerance (1,1) double = 1e-6
    end
    ...
end
```

**实现要点**:
- `parser.py`: 新增 `_parse_arguments_block` 方法
- `interpreter.py`: 函数调用前执行参数验证
- 内置验证函数: `mustBeNumeric`, `mustBePositive`, `mustBeNonempty`, `mustBeMember`

#### 1.3 table 增强

```matlab
T = table([1;2;3], ['A';'B';'C'], 'VariableNames', {'ID', 'Name'});
T(1:2, :)           % 行索引
T{:, 'ID'}          % 花括号数据提取
T.ID                % 点访问 (已支持)
```

#### 1.4 通用命令式语法

```python
# parser.py: 任何 identifier 后跟 string/number token 都可作为命令式调用
# 不再硬编码函数名列表
if current_token is IDENTIFIER and next_token in (STRING, NUMBER, IDENTIFIER):
    return CommandCall(name, args)
```

### Phase 2: 字节码 VM 完善 (第 5-7 周)

**目标**: 循环密集场景性能提升 5-10x

| 任务 | 新增 Opcode | 说明 |
|------|------------|------|
| 用户函数调用 | `CALL_USER_FUNC`, `PUSH_FRAME`, `POP_FRAME` | 原生栈帧，不回退树遍历 |
| 闭包支持 | `LOAD_UPVALUE`, `STORE_UPVALUE` | 捕获外层变量 |
| switch 完整 | `SWITCH_START`, `CASE_CMP` | VM 原生处理 |
| try/catch 完整 | `TRY_BEGIN`, `TRY_END`, `CATCH_BEGIN` | VM 原生异常 |
| INDEX_ASSIGN | 已有 opcode | 完善编译逻辑 |
| global/persistent | `LOAD_GLOBAL`, `STORE_GLOBAL` | VM 变量空间 |

**性能目标**:

| 基准测试 | 树遍历 | 字节码目标 | 提升倍数 |
|---------|--------|-----------|---------|
| fibonacci(1000) | ~2s | ~0.2s | 10x |
| 循环 10000 次 sin | ~5s | ~0.5s | 10x |
| 矩阵 100x100 乘法 | ~0.01s | ~0.01s | 1x (numpy) |

### Phase 3: 内置函数扩展 (第 8-11 周)

**目标**: 内置函数从 580 提升到 800+

#### 优先补齐清单

**P0 — 核心数学 (阻塞用户日常使用)**

| 函数 | 模块 | 说明 |
|------|------|------|
| `nextpow2` | math.py | 2 的下一个幂 |
| `cart2pol`/`pol2cart` | math.py | 坐标转换 |
| `airy` | math.py | Airy 函数 |
| `besselj`/`bessely` | math.py | Bessel 函数 |
| `gallery` 增强 | matrix_ops.py | 测试矩阵完整集 |
| `balance` | matrix_ops.py | 矩阵平衡 |
| `gsvd` | matrix_ops.py | 广义 SVD |

**P1 — 工程应用 (信号/控制/优化)**

| 函数 | 模块 | 说明 |
|------|------|------|
| `czt` | signal.py | Chirp Z 变换 |
| `goertzel` | signal.py | Goertzel 算法 |
| `firpm` | signal.py | Parks-McClellan FIR |
| `fdesign` | signal.py | 滤波器设计框架 |
| `lqr` | control.py | LQR 控制器 |
| `kalman` | control.py | Kalman 滤波器 |
| `rlocus` | control.py | 根轨迹 |
| `surrogateopt` | optimization.py | 代理优化 |
| `gamultiobj` | optimization.py | 多目标遗传算法 |

**P2 — 数据科学 (统计/ML)**

| 函数 | 模块 | 说明 |
|------|------|------|
| `fitcsvm` | statistics.py | SVM 分类器 |
| `fitctree` | statistics.py | 决策树 |
| `fitcensemble` | statistics.py | 集成学习 |
| `glmval` | statistics.py | GLM 预测 |
| `confusionmat` | statistics.py | 混淆矩阵 |

#### .mat v7.3 (HDF5) 支持

```python
# file_io.py 增强
import h5py

@register("load")
def mat_load(filename, *args):
    if filename.endswith('.mat'):
        try:
            import scipy.io
            data = scipy.io.loadmat(filename)
        except NotImplementedError:
            # v7.3 HDF5 format
            import h5py
            data = {}
            with h5py.File(filename, 'r') as f:
                for key in f.keys():
                    data[key] = np.array(f[key])
```

### Phase 4: 类型系统增强 (第 12-13 周)

**目标**: 支持 MATLAB 完整数值类型体系

| 类型 | Python 实现 | 说明 |
|------|------------|------|
| `int8/16/32/64` | `numpy.int8/16/32/64` | 显式整数 |
| `uint8/16/32/64` | `numpy.uint8/16/32/64` | 无符号整数 |
| `single` | `numpy.float32` | 单精度 |
| `logical` | `numpy.bool_` (独立包装) | 布尔类型 |
| `timetable` | 扩展 `Table` | 时间索引 |

### Phase 5: 工程质量 (第 14-15 周)

#### interpreter.py 拆分方案

```
matpy/
├── interpreter.py        # 主类 (~200行): run(), _exec_stmt 分发
├── evaluator.py          # 表达式求值 (~300行): _eval, _eval_binop, _eval_index
├── statement_handler.py  # 语句执行 (~200行): _exec_if, _exec_for, _exec_while
├── func_resolver.py      # 函数解析 (~150行): _eval_func_call, _call_user_func
└── class_handler.py      # OOP 处理 (~150行): _exec_classdef, method dispatch
```

#### 错误处理标准化

```
MatPyError
├── MatPySyntaxError       ← 解析阶段
├── MatPyTypeError         ← 类型不匹配
├── MatPyIndexError        ← 索引越界
├── MatPyNameError         ← 未定义变量
├── MatPyValueError        ← 值域错误
├── MatPyDimensionError    ← 维度不匹配
├── MatPyMemoryError       ← 内存不足
├── MatPyNotImplementedError ← 未实现特性
└── MatPyFileNotFoundError ← 文件不存在
```

每个错误包含: `filename`, `line`, `col`, `identifier` (如 `MATLAB:UndefinedFunction`)

#### REPL 增强

| 功能 | 实现方案 |
|------|---------|
| Tab 补全 | `readline.set_completer` + 变量名/函数名索引 |
| 调试器 | `dbstop` 设置断点, `dbstep` 单步, `dbcont` 继续, `dbquit` 退出 |
| 文档 | `doc funcname` 显示完整文档 |
| 历史搜索 | readline Ctrl+R (已有基础) |

### Phase 6: 高级特性 (第 16-18 周)

#### 并行计算

```python
# interpreter.py
def _exec_parfor(self, node):
    """parfor i = 1:n ... end → multiprocessing.Pool.map"""
    from multiprocessing import Pool
    iterable = self._eval(node.iter)
    body_func = self._compile_parfor_body(node.body, node.var)
    with Pool() as pool:
        results = pool.map(body_func, iterable)
    self._assign_parfor_results(node.var, results)
```

#### Python MEX 桥接

```matlab
% 导入 Python 模块
np = py.importlib.import_module('numpy');
result = np.array([1, 2, 3]);

% 调用 Python 函数
py.math.sqrt(2)  % → 1.4142
```

#### JIT 探索

```
热点检测 → 类型推断 → LLVM/Numba 编译 → 缓存
- 识别循环体中频繁执行的代码路径
- 推断变量类型 (double 矩阵为主)
- 评估 numba jit 或自建 IR 编译
- 编译结果缓存到 __pycache__
```

---

## 五、测试扩展计划

| 阶段 | 当前 | 目标 | 新增重点 |
|------|------|------|---------|
| Phase 0 | 504 | 550 | Bug 回归测试 |
| Phase 1 | 550 | 700 | classdef/arguments/table 测试 |
| Phase 2 | 700 | 900 | 字节码 VM 全特性测试 |
| Phase 3 | 900 | 1,200 | 内置函数数值精度测试 |
| Phase 4 | 1,200 | 1,400 | 类型系统测试 |
| Phase 5 | 1,400 | 1,800 | 集成测试 + REPL 测试 |
| Phase 6 | 1,800 | 2,000+ | 并行/MEX/JIT 测试 |

---

## 六、风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 字节码 VM 用户函数调用复杂度超预期 | 高 | Phase 2 延期 | 先保证树遍历正确性，字节码作为可选优化 |
| classdef 完整实现工作量大 | 中 | Phase 1 延期 | 先覆盖 80% 场景 (Access, Constant, Static)，复杂特性标记 NotImplemented |
| 内置函数数值精度与 MATLAB 不一致 | 高 | 测试失败 | 设合理误差容限，与 Octave 对比验证 |
| .mat v7.3 HDF5 兼容性问题 | 中 | I/O 受限 | 依赖 scipy/h5py，测试多种 MATLAB 版本生成的文件 |
| JIT 编译探索无明确方案 | 高 | Phase 6 延期 | 先做好字节码，JIT 作为远期目标 |

---

## 七、成功指标

### 功能指标
- [ ] 能运行 80% 的 MATLAB 基础教程代码
- [ ] 内置函数 ≥ 800 (去重后)
- [ ] 语言覆盖率 ≥ 75%

### 性能指标
- [ ] 启动时间 < 0.5s
- [ ] 矩阵运算 ≈ NumPy 原生性能
- [ ] 循环密集场景 (字节码) ≥ MATLAB 的 10%

### 质量指标
- [ ] 测试 ≥ 2,000 项，覆盖率 ≥ 85%
- [ ] 无已知崩溃
- [ ] 错误消息可定位到文件+行号+列号
- [ ] 所有公共 API 有文档
