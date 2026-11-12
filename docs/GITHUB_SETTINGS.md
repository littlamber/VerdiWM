# GitHub 远端发布设置

这些设置属于 GitHub 仓库状态，提交工作流文件不会自动启用它们。
检查日期：2026-11-11，目标仓库 `littlamber/VerdiWM` 已公开，默认分支为 `main`。
公开 API 显示私密漏洞报告未启用、rulesets 列表为空；经典分支保护状态无法
通过本次匿名请求确认。当前执行环境没有 GitHub 管理令牌，未修改远端设置。

## 维护者需要落实的设置

1. Settings → Code security：启用 Private vulnerability reporting、Dependabot
   alerts/security updates，以及账户计划允许的 secret scanning/push protection。
   不要将令牌贴到聊天、issue、工作流文件或命令行参数中。
2. Settings → Rules → Rulesets：为 `main` 建立 active 分支规则，要求通过 PR
   合并、至少一位审阅者、解决讨论并通过检查；限制强推与分支删除。
3. 在工作流实际运行后选择 required checks：`control-plane`、`resolved-install`、
   `locked-advisories`。不要将尚不存在的检查配置成无法满足的门禁。
4. Settings → Actions：维持只读默认 GITHUB_TOKEN 权限；外部贡献只在隔离的
   GitHub-hosted runner 上运行，不把来自 fork 的代码放到有模型、凭据或
   私有数据的自托管 GPU runner。禁止用 pull_request_target 执行 PR 中的代码。
5. 已有公开标签不可重写。候选版本使用新的版本号；正式 tag、发布附件与
   SHA256 应对应同一审核提交。新建公开历史前先检查已有用户和分支的迁移影响，
   不对现有公开 main 自动强推。

已通过认证且有管理员权限时，可用 GitHub CLI 启用私密报告：

```bash
gh api --method PUT repos/littlamber/VerdiWM/private-vulnerability-reporting
```

其他设置优先在仓库页面核对适用计划和现有规则后配置。完成后重新读取远端
状态；不能把本文件或准备好的命令当成已经生效的证明。
