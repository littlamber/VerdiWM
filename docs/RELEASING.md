# 开源发布与验证

## 发布对象与支持范围

当前支持 Linux（Windows 使用 WSL2）和 Python 3.12，项目处于 Beta 阶段。
控制面测试与真实模型/GPU 验证分别记录，未经验证的平台或效果不作支持承诺。
主实现放在 `wmloop/`；`scripts/` 仅作入口和运维工具，库代码不得导入它。
工作台直接使用 `wmloop/web/` 中的 HTML/CSS/JavaScript，不需要前端构建。
授权说明见 `THIRD_PARTY_NOTICES.md`，修改约束以 `AGENTS.md` 为准。

## 安装与验证

首次联网安装使用固定 uv 版本（当前 CI 为 0.12.10），执行：

```bash
uv sync --locked --group dev
bash scripts/ci/check_control_plane.sh
```

`--locked` 要求锁文件与声明一致。构建系统的依赖也固定在 `pyproject.toml`。
本地已有缓存时使用 `uv sync --offline --frozen`；验证脚本默认
`UV_OFFLINE=1`。控制面不安装可选 `torch-tests` 组，Torch/GPU 验证另行执行。

验证脚本先导出源码，再构建 wheel 与 sdist，扫描 wheel 文件内容，并在
仓库之外的新虚拟环境中安装锁定且校验摘要的依赖及 wheel。安装验证包括
CLI、schema 的正负例、导出服务和工作台资源。缓存不足或缺资源的检查
必须报告失败/跳过，不能把原环境的可导入结果冒充干净安装。

## 源码快照与 Git 历史

发布脚本默认只接受没有遗漏公开新文件的工作树：

```bash
uv run --offline --frozen python -m scripts.export.build_github_staging \
  --source-root . --output-root artifacts/release/source
```

未跟踪的公开源码会导致明确报错；已跟踪的凭据或非公开文件同样阻断发布，
不能靠导出过滤掩盖它们已经进入 Git 的事实。开发期可添加 `--include-untracked` 生成
完整审阅快照，不修改真实 Git index。该选项不会把私有目录、凭据文件或
生成产物加入发布。最终发布仍需在包含所有预期源码的提交上重跑检查。
`git commit -am` 不会纳入新文件。导出目录必须不存在，禁止使用符号链接。

导出的目录不带 `.git`，不继承或批准开发仓库的历史。现有 GitHub 仓库已
公开，此快照用于审阅文件差异，不能作为重建或强推覆盖历史的授权。
保留源码中的版权、许可证和上游署名；不要将开发仓库的 refs、tags 或
整个 `.git` 复制到交付目录。

发布到现有仓库之前，必须先检查所有准备推送的 refs：

```bash
uv run --offline --frozen python -m scripts.ci.check_release_history \
  --repo-root . --report artifacts/release/history_audit.json
```

扫描覆盖本地可达对象和历史文件名，只报告类型、对象 ID/文件名，不输出
凭据值；过大或二进制对象会标为未扫描并阻断。浅克隆和未获取的远程 refs
不在扫描范围内。模式扫描只能发现已知形式，不能证明不存在秘密。
若历史含真实凭据，先撤销/轮换，再处理历史；删除文件不会撤销已泄露凭据。

## 服务与执行边界

Campaign API 和 Workbench 只允许本地回环绑定，校验 Host/Origin，并限制
JSON 请求大小与读取时间。所有 API 使用每次启动的访问令牌，不提供多租户授权，
不支持通过反向代理或
端口转发向不可信用户开放。远程服务必须单独设计身份验证与授权。
候选执行的进程隔离要求见 `SECURITY.md`；Git worktree 不是操作系统沙箱。

## 发布门禁

- 源码扫描、目录依赖规则、完整 CPU 测试和干净安装全部通过。
- wheel 与 sdist 来自同一审阅快照，且不含私有产物和凭据文件。
- 明确准备发布的历史检查范围；未解决的历史问题不得随源码推送。
- 许可证、平台范围、跳过项与真实模型验证边界在发布说明中可查。
- 发布由维护者显式执行；构建/审计脚本不提交、不推送、不创建远程仓库。

## 第二轮发布门禁补充

- 本地服务的跨用户认证见 `LOCAL_HTTP_AUTH.md`；HTTP 测试必须同时覆盖无令牌拒绝、
  有令牌成功、重启失效与令牌文件权限。
- `pytest.xml` 的公开版本只含测试数量汇总，原始测试名称、stdout/stderr、属性、
  主机名、时间戳和路径都不发布。原始诊断日志只保留在本地 artifacts。
- 安装验证分别记录 locked 与 resolved 两种依赖模式。离线 resolved 检查只代表
  当前缓存可解析的组合；GitHub 的 wheel-install 工作流另行联网验证当前解析结果，
  不使用仓库锁约束 wheel 的依赖选择。
- GitHub 的控制面 CI 在联网准备阶段通过 `check_installed_wheel.py --prepare-locked-cache`
  安装带摘要的锁定依赖，再切换离线运行发布检查。仅执行项目 `uv sync` 不代表
  已准备完整的独立 `uv pip install --require-hashes` 缓存；准备步骤不算 wheel 验证。
- dependency-security 每周及 main 更新后查询 OSV，查询失败不是通过。安全报告
  包含可选测试依赖；命中公告时不能用 CPU 测试通过替代修复或经过审查的适用性分析。
- 远端设置见 `GITHUB_SETTINGS.md`。已有公开仓库不自动重写历史或强推；本地报告
  不代表远端 required checks、私密漏洞报告或 secret scanning 已启用。
- 未发布候选使用 1.0.4rc1，不覆盖现有 v1.0.3。版本、CHANGELOG 与 CITATION.cff 一起维护。

在线安全查询只发送依赖名称和版本，不上传源码、模型、数据或凭据。报告绑定
锁文件 SHA256 和查询时间；锁文件变化后必须重新检查。测试与安装产物只证明
记录的版本组合，不能承诺未来索引解析得到的任何依赖版本都兼容。
