# 发布检查清单

本清单与 `AGENTS.md` 第 7 节一致，详细约束见 [RELEASING.md](RELEASING.md)。

1. 审查全部新增文件，只纳入公开源码、配置、测试和正式文档；运行产物、
   凭据、模型和私有数据不得入库。
2. 正式发布在干净、已提交的候选版本执行；开发阶段可用 `--allow-dirty`
   生成完整审阅快照。快照有文件摘要，但不能冒充已提交的发布版本。
3. 首次安装运行 `uv sync --locked --group dev`，再运行：

```bash
bash scripts/ci/release_preflight.sh --output-dir artifacts/release/delivery
```

此入口复用控制面 CI：扫描源码，构建和扫描 wheel/sdist，从 sdist 重建并
比对 wheel 内容，在新环境验证实际 API/资源，在导出源码的新环境运行
完整 CPU 测试。通过后输出 `repository/`、发行包、`pytest.xml` 和
`release_report.json`。报告包括文件摘要、测试数量与验证边界。

4. `repository/` 不带旧 `.git`，仅用作源码审阅。现有 GitHub 仓库已经公开，
   不以此快照重建或强推覆盖已有历史。按 RELEASING.md 扫描全部准备推送的
   refs；旧仓库中的问题不能通过删除当前文件或新增 `.gitignore` 消除。
5. 确认第三方许可、平台范围和真实 GPU 验证边界。启用 GitHub 私密漏洞
   报告入口。自动脚本不执行任何推送或远端发布。
6. 发布后在新目录克隆公开仓库，再执行同一预检，验证公开提交完整性。

GitHub 源码只来自 `repository/`；发行包和验证报告作为独立发布附件。
不要将开发目录的 `.git`、旧 UI 备份、模型、数据或历史实验包复制进去。
