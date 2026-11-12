# Contributing

仓库约束以 [AGENTS.md](AGENTS.md) 为准；代码组织参考
[代码规范](docs/code_conventions.md)，闭环方法映射见
[VERDI 方法工程映射](docs/VERDI_METHOD_ENGINEERING.md)。

Contributions should preserve VerdiWM's claim boundary.

1. Open an issue describing the failure mechanism, intended intervention, and
   falsifiable expected effect.
2. Keep changes narrow and add an offline harness or unit test.
3. Do not modify frozen evaluators, held-out splits, acceptance thresholds, or
   existing evidence to make a candidate pass.
4. For a new primitive, declare its semantic descriptor, required capability,
   hook, dose, invariants, and intent-to-code receipt.
5. For a new backbone, add capability, goal, split, evaluator, hook, receipt,
   and regression-harness surfaces before claiming closed-loop support.
6. Run `bash scripts/ci/check_control_plane.sh` before submitting a change.

Generated code is reviewed under the same standard as handwritten code. If an
implementation cannot preserve the declared method intent, it must fail closed
or return to proposal rather than silently substitute a weaker method.

Do not commit model weights, datasets, private logs, credentials, machine-local
paths, or generated `results/` and `runs/` trees.



发布候选和依赖维护还需遵守 [发布规范](docs/RELEASING.md)、
[本地认证](docs/LOCAL_HTTP_AUTH.md) 与 [远端设置](docs/GITHUB_SETTINGS.md)。
控制面当前支持 Python 3.12；不要把外部模型运行时的 Python 要求混入控制面。
安全公告命中应先核对适用版本与修复版本，再升级锁文件并完成离线验证；
不得通过删除查询结果、弱化测试或静默豁免来制造通过。
