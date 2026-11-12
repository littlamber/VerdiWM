# 安全政策

## 私密报告

在目标 GitHub 仓库的 Security → Advisories → Report a vulnerability 提交
私密报告。维护者首次公开仓库时应先启用 GitHub Private vulnerability reporting。
若该入口尚未启用，只在 issue 请求开启私密报告入口，不公开漏洞细节或凭据。
报告应包含受影响 revision、复现步骤、影响范围和最小复现，不要附真实密钥。
目前仅对最新发布版本和主分支接受修复；历史版本不承诺回补。

## 本地服务

Workbench 和 Campaign API 仅服务可信的本机用户，只绑定回环地址。
服务校验 Host/Origin、JSON framing，并限制请求体为 1 MiB、读取超时为 10 秒。
所有 API 还必须验证每次启动生成的随机访问令牌，令牌文件目录为 0700、文件为 0600。
浏览器通过同源 Authorization header 调用接口，不使用 Cookie 或 URL 传递令牌。
不要通过反向代理、容器端口映射或端口转发向不可信网络开放服务。
没有令牌的同机其他普通用户访问 API 返回 401。root 和同 UID 的不可信进程仍可能
获取凭据；共享机器还需独立账户和输出目录。详见 `docs/LOCAL_HTTP_AUTH.md`。

## 候选代码执行

VerdiWM 可以准备并执行代理生成的训练代码。Git worktree 是审计边界，
不是操作系统沙箱。不可信候选应在独立容器或等效隔离环境中运行：

- 默认无网络；按任务明确开放必要连接。
- 源码和 evaluator 只读挂载，只开放当前 trial 的输出目录写入。
- 移除 Linux capabilities，不传入宿主凭据。
- 限制 CPU、内存、进程数、运行时间、磁盘和 GPU。
- 对命令与产物收集建立明确白名单。

不要向 proposal provider 暴露生产密钥或个人数据。冻结摘要和 receipt 用于
审计，不能替代进程隔离；路径预检查也不是抵抗恶意并发修改目录的沙箱。

## 发布检查

参见 `docs/RELEASING.md`。当前源码扫描不能替代历史审计，也不能证明不存在
未知格式的凭据。发生真实凭据泄露时先撤销/轮换，再处理文件与 Git 历史。
