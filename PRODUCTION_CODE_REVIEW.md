# MatPy v0.5.0 — 代码级生产对标分析

> 生成日期: 2026-06-27
> 分析方法: 自动化测试 + 代码审查代理 × 2
> 测试状态: 520 passed / 0 failed
> 测试覆盖率: **47%** (目标 ≥80%)

---

## 一、总体评估

| 维度 | 当前 | 目标 | 状态 |
|------|------|------|------|
| 测试覆盖率 | 47% | ≥80% | 🔴 差距大 |
| CRITICAL Bug | 4 | 0 | 🔴 必须修复 |
| HIGH Bug | 23 | 0 | 🔴 必须修复 |
| MEDIUM 问题 | 20+ | — | 🟡 需改进 |
| LOW 问题 | 8+ | — | 🟢 可延后 |

---

## 二、CRITICAL 问题（阻断性，必须立即修复）

### C-1. try/catch 吞掉 return/break/continue
- **文件**: `interpreter.py:327-345`
- **问题**: `except Exception` 捕获了 `ReturnSignal`/`BreakSignal`/`ContinueSignal`
- **影响**: `try` 块中的 `return` 不会退出函数，而是执行 catch 块
- **修复**: 在 except 中优先 re-raise 控制流信号

### C-2. for 循环遍历标量 Mat 崩溃
- **文件**: `interpreter.py:277-280`
- **问题**: 0-d Mat 的 `data.shape[0]` 抛 IndexError
- **影响**: `for i = 5` 直接崩溃
- **修复**: 添加 `elif data.ndim == 0` 分支

### C-3. eval() 始终返回 None
- **文件**: `interpreter.py:969-981`
- **问题**: `result = None` 从未被赋值
- **影响**: `eval('1+2')` 返回 None
- **修复**: 追踪最后一个 ExprStmt 的结果

### C-4. io.py 引用未定义变量
- **文件**: `builtins/io.py:313`
- **问题**: `isinstance(x, Mat)` 中 `x` 不存在，参数名是 `x0`
- **影响**: `py.scipy.optimize.minimize` 必然 NameError
- **修复**: `isinstance(x, Mat)` → `isinstance(x0, Mat)`

---

## 三、HIGH 问题（按模块分组）

### 3.1 解释器 (interpreter.py)

| # | 行号 | 问题 | 影响 |
|---|------|------|------|
| H-1 | 626-628 | `A(end)` 在 2D 数组上返回行数而非元素总数 | 索引错误 |
| H-2 | 535-551 | `_eval_unaryop` 对非数值类型静默返回 None | 无错误提示 |
| H-3 | 1078-1085 | `_is_equal` 可能返回 numpy 数组而非 bool | switch/case 崩溃 |
| H-4 | 470-527 | `&&`/`||` 不短路求值 | `false && error('x')` 抛异常 |
| H-5 | 497 | 矩阵幂静默截断浮点指数 | `A^2.5` → `A^2` |
| H-6 | 521 | 标量 `&`/`|` 返回 Python and/or 结果 | 类型错误 |
| H-7 | 490,494 | 矩阵除法假设方阵可逆 | 非方阵抛原始 LinAlgError |

### 3.2 解析器 (parser.py)

| # | 行号 | 问题 | 影响 |
|---|------|------|------|
| H-8 | 496-521 | 命令式语法对含运算符的参数产生错误 AST | `disp 1+2` 解析失败 |
| H-9 | 496-498 | 关键字参数不触发命令式语法 | `disp end` 解析失败 |
| H-10 | 29-35 | `_peek`/`_advance` 无边界检查 | 越界时抛原始 IndexError |

### 3.3 词法分析器 (lexer.py)

| # | 行号 | 问题 | 影响 |
|---|------|------|------|
| H-11 | 235-256 | 双引号字符串末尾 `\` 不报错 | `"hello\` 被静默接受 |

### 3.4 运行时类型 (runtime/types.py)

| # | 行号 | 问题 | 影响 |
|---|------|------|------|
| H-12 | 467 | Table 切片 stop 多减 1 | `T(1:3)` 只返回 2 行 |
| H-13 | 466 | 切片 start=0 被当 None（falsy-zero） | 边界条件错误 |
| H-14 | 173 | FuncHandle 类型注解用 `callable` 非 `Callable` | mypy 报错 |
| H-15 | 440-444 | Table.set_field if/else 两分支相同 | 死代码 |
| H-16 | 421 | Table.Height 只检查第一列长度 | 列长不一致时错误 |

### 3.5 作用域 (environment.py)

| # | 行号 | 问题 | 影响 |
|---|------|------|------|
| H-17 | 53 | `_get_global` 抛 KeyError 非 NameError | 错误类型不一致 |
| H-18 | 55-59 | `_set_global` 可注入未声明的全局变量 | 静默污染作用域 |

### 3.6 文件 I/O (builtins/file_io.py)

| # | 行号 | 问题 | 影响 |
|---|------|------|------|
| H-19 | 503-507 | tempname 使用不安全的 mktemp() | TOCTOU 竞态 |
| H-20 | 539-546 | path() 替换而非追加 sys.path | 破坏后续 import |
| H-21 | 369 vs 431 | _exist 有两个版本，注册的是桩 | exist('f','func') 不工作 |

### 3.7 I/O (builtins/io.py)

| # | 行号 | 问题 | 影响 |
|---|------|------|------|
| H-22 | 26-29 | nargin 始终返回 0 | MATLAB 语义错误 |
| H-23 | 99-103 | _validateattributes 是空操作桩 | 验证无效 |

---

## 四、MEDIUM 问题汇总

### 解释器
- `_eval_range` step=0 崩溃（应返回空数组）
- `"end_marker"` 哨兵字符串可能与用户变量冲突
- `_load_package` 用 print 非 logging
- `_exec_parfor` 是死代码（lexer/parser 无 parfor）
- 矩阵除法对非方阵不用 lstsq
- 字符串索引无边界检查

### 解析器
- classdef 块中未知 token 被静默跳过
- 匿名函数体仅支持单表达式

### 词法分析器
- 转置 vs 字符串启发式可能误分类
- 块注释不支持嵌套
- `_add` 中 `line or self.line` 对 line=0 有 bug

### 运行时类型
- `Mat.__len__` 对 0-d 返回 1（应抛 TypeError）
- `CellArray([])` 被当 None 处理
- `StringArray.__eq__` 返回 numpy 数组非 bool
- `Duration` 属性可被外部修改导致不同步

### 文件 I/O
- 错误输出到 stdout 非 stderr
- `_file_handles` 非线程安全
- `_fgetl` 不处理 `\r\n`

### I/O
- `py.importlib.import_module` 允许任意模块导入（安全隐患）
- `_OnCleanup` 依赖 `__del__`（不保证执行）
- 多处冗余 `import numpy as np`

---

## 五、测试覆盖率详情

### 整体: 47%

| 模块 | 覆盖率 | 优先级 |
|------|--------|--------|
| ast_nodes.py | 98% | ✅ |
| tokens.py | 99% | ✅ |
| parser.py | 79% | P2 |
| lexer.py | 76% | P2 |
| math.py | 67% | P2 |
| interpreter.py | 60% | P1 |
| common.py | 52% | P1 |
| runtime/types.py | 51% | P1 |
| bytecode.py | 48% | P1 |
| environment.py | 48% | P1 |
| string.py | 48% | P1 |
| io.py | 43% | P1 |
| plotting.py | 42% | P1 |
| advanced_math.py | 42% | P1 |
| string_array.py | 40% | P1 |
| signal.py | 40% | P1 |
| matrix_ops.py | 39% | P1 |
| sparse.py | 37% | P1 |
| statistics.py | 36% | P1 |
| optimization.py | 25% | P0 |
| control.py | 25% | P0 |
| file_io.py | 26% | P0 |
| data_struct.py | 22% | P0 |
| image.py | 21% | P0 |
| runtime/matrix.py | 26% | P0 |

---

## 六、修复路线图

### Phase 1 — 修复阻断性问题（1-2 天）
- [ ] C-1: try/catch 控制流信号
- [ ] C-2: for 标量崩溃
- [ ] C-3: eval() 返回 None
- [ ] C-4: io.py 变量名错误
- [ ] 验证: 所有现有测试仍通过

### Phase 2 — 修复 HIGH 问题（3-5 天）
- [ ] H-1 ~ H-7: 解释器核心语义
- [ ] H-8 ~ H-11: 解析器/词法分析器
- [ ] H-12 ~ H-16: 运行时类型
- [ ] H-17 ~ H-18: 作用域
- [ ] H-19 ~ H-23: 文件 I/O 和 I/O
- [ ] 验证: 为每个修复编写回归测试

### Phase 3 — 提升覆盖率至 80%（1-2 周）
- [ ] 为覆盖率 <30% 的模块补充测试
- [ ] 重点: interpreter.py, types.py, environment.py
- [ ] 验证: `pytest --cov=matpy` 显示 ≥80%

### Phase 4 — 代码质量提升（持续）
- [ ] 移除死代码（_exec_parfor, io.py 重复定义）
- [ ] 补全类型注解
- [ ] 统一错误输出到 stderr
- [ ] 添加 linting 配置
- [ ] 补全 docstring

---

## 七、与旧版差距报告的关系

本报告 (`PRODUCTION_CODE_REVIEW.md`) 聚焦于**代码级 bug 和正确性问题**，是对 `MATPY_GAP_REPORT.md`（功能级差距分析）的补充：
- `MATPY_GAP_REPORT.md` — 功能覆盖、语言特性、与 MATLAB 的功能对比
- `PRODUCTION_CODE_REVIEW.md`（本文件） — 代码质量、bug、测试覆盖率、修复优先级
