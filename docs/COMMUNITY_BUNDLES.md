# 社区共享 Bundle

VerdiWM 的社区发布物是一个可验证的、无本地路径的知识包。它用于把已经整理好的模型画像、Capability IR、IRG、干预结果、证据边界和知识图谱投影发布到魔塔社区等 registry，供其他研究者检索和生成 transfer hypothesis。

Bundle 是传输和发布边界，不是实验权威。模型源码、checkpoint、数据集、GPU 命令、campaign 目录、budget ledger 和本地数据库都不会被复制进去。真正的实验结论仍由本地 Archive/CAS、冻结 evaluator、target-side verification 和 settlement receipt 支撑；bundle 只携带这些结论的路径无关语义摘要以及内容地址引用。

## 最小发布流程

先在本地完成模型接入、IRG 探针和受控实验，并从已有的 semantic documents 生成 JSON 文件。文件可以包含一个文档，也可以是文档数组：

```bash
verdiwm community publish \
  --document model-portrait.json \
  --document evidence-records.json \
  --output-root ./community-bundle \
  --publisher-id community/example \
  --signing-key ./publisher-private.pem \
  --trust-state locally_validated
```

发布命令只接受已经通过 VerdiWM portable knowledge graph 校验的语义文档。它会重建 deterministic graph，生成 quality audit，为每个 record 写入 SHA-256，并用 Ed25519 私钥签名 `bundle.json`。同一 bundle 目录可以安全重复发布；如果目录中的内容已经被改动，命令会报冲突而不会覆盖它。

如果语义记录分散在批量实验、画像和证据工具生成的多个目录，可以先用只读
导出器发现并整理它们，再把导出目录下的 `records/` 交给发布器：

```bash
verdiwm community export \
  --source-root ./local-artifacts \
  --source-root ./.verdiwm/semantic-records \
  --execution ./.verdiwm/batches/my-batch/execution.json \
  --output-root ./.verdiwm/community-export

verdiwm community publish \
  --documents-dir ./.verdiwm/community-export/records \
  --execution ./.verdiwm/batches/my-batch/execution.json \
  --output-root ./community-bundle \
  --publisher-id community/example \
  --signing-key ./publisher-private.pem
```

`community export` 只递归读取 JSON，不导入模型、不启动 GPU、不修改源目录，也
不会把 `execution.json`、Archive、CAS 数据库或本地运行路径复制到输出。它接受
单个语义对象和 JSON 数组，按 canonical SHA-256 去重，并对已识别的语义类型运行
完整 graph 和 quality audit；无关的运行时 manifest 会在 `export.json` 中计数并
忽略，已识别但不合约的文档会直接阻断。导出目录使用临时目录和原子替换写入，
同一输入可重复执行；若已有目录被篡改或混入额外文件，命令会报冲突。

导出目录包含 `export.json`、`graph.json`、`quality-audit.json` 和
`records/<sha256>.json`。发布时只把 `records/` 作为 `--documents-dir`；三个
导出元文件是 staging 和审计结果，不应作为语义记录再次发布。

如果语义记录已经集中保存在一个目录，可以用 `--documents-dir` 读取该目录第一层按文件名排序的 `*.json` 文件：

```bash
verdiwm community publish \
  --documents-dir ./semantic-records \
  --output-root ./community-bundle \
  --publisher-id community/example \
  --signing-key ./publisher-private.pem
```

在上传 registry 前或 registry 收到 bundle 后，可以独立验证：

```bash
verdiwm community verify \
  --bundle-root ./community-bundle \
  --public-key ./publisher-public.pem
```

`--public-key` 是可选的。省略时，验证器使用 `signature.json` 中的公钥；如果同时提供，两个公钥必须逐字节相同。验证器会检查 manifest schema、发布者和 key ID 绑定、Ed25519 签名、所有 member hash、目录中是否有未声明文件、quality audit、record 数量、record 文件名、图摘要以及从 records 重建出的 graph。验证成功只代表发布物完整且由该 key 签署，不代表目标模型已经获得同样的效果。

批量实验完成后，可以额外传入 `--execution` 绑定批次：

```bash
verdiwm community publish \
  --documents-dir ./semantic-records \
  --execution ./.verdiwm/batches/my-batch/execution.json \
  --output-root ./community-bundle \
  --publisher-id community/example \
  --signing-key ./publisher-private.pem
```

系统会校验 `execution.json` 的 batch execution schema 和摘要，然后只把 `batch_id`、`plan_sha256`、`execution_sha256`、状态和排序后的模型 ID 写入 `bundle.json`。批次里的 campaign、budget、模型目录、数据路径和运行时命令不会进入 bundle。这个绑定用于追踪来源，不会把编排状态提升成模型质量结论。

## 密钥和发布者身份

发布者本地生成并保管 Ed25519 私钥。当前实现接受 PEM 编码的 PKCS8 私钥和 SubjectPublicKeyInfo 公钥；私钥永远不会写进 bundle。`signing_key_id` 是公钥原始 32 字节的 SHA-256 前 16 位，并以 `key-` 开头。建议把私钥放在受保护的密钥目录中，把公钥和发布者 ID 放进社区 registry 的身份记录。

可以用项目环境生成一对密钥：

```bash
uv run python - <<'PY'
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

private = Ed25519PrivateKey.generate()
Path("publisher-private.pem").write_bytes(private.private_bytes(
    serialization.Encoding.PEM,
    serialization.PrivateFormat.PKCS8,
    serialization.NoEncryption(),
))
Path("publisher-public.pem").write_bytes(private.public_key().public_bytes(
    serialization.Encoding.PEM,
    serialization.PublicFormat.SubjectPublicKeyInfo,
))
PY
chmod 600 publisher-private.pem
```

签名覆盖 canonical JSON 格式的 `bundle.json`。manifest 又覆盖 `graph.json`、`quality-audit.json` 和全部 `records/*.json` 的 SHA-256，所以修改任意成员都会在验证阶段失败。签名只解决完整性和签署者绑定；社区对发布者的信任、审核和撤回仍需要 registry 的治理记录。

## Bundle 目录

```text
community-bundle/
├── bundle.json          # bundle 元数据、trust state、成员 hash 和 claim boundary
├── signature.json       # Ed25519 算法、公钥、manifest digest 和签名
├── graph.json           # 从 records 确定性重建的 path-free knowledge graph
├── quality-audit.json   # 图的 path-free、license、冻结证据审计
└── records/
    ├── <record-digest>.json
    └── ...
```

发布包只允许上述文件。`records/` 文件名由文档 canonical digest 的前 32 位构成；验证器会重新计算并检查这一绑定。这样 registry 可以只保存和分发声明过的内容，也不会把临时说明文件误当成知识记录。

## Trust state 和社区审核

`trust_state` 表示证据生命周期，而不是一个不可解释的总分：

| 状态 | 含义 |
| --- | --- |
| `unverified` | 已形成语义记录，但尚未完成本地验证 |
| `locally_validated` | 在发布者本地通过既定 verifier 和 settlement 流程 |
| `source_reproducible` | 具备可复现源侧实验条件 |
| `target_confirmed` | 在目标模型上通过冻结 evaluator 的确认实验 |
| `transfer_licensed` | 已满足可用于 transfer ranking 的证据和授权边界 |
| `community_reviewed` | 经过社区审核流程 |
| `revoked` | 发布者或社区已撤回，不能作为可信 transfer 知识使用 |

`community_review_state` 独立表示社区审核：`unreviewed`、`reviewed` 或 `withdrawn`。签名完整的 `revoked` bundle 仍然可以被验证，以便审计历史；验证结果会明确返回 `state: revoked`，调用方必须将它从可信检索结果中排除。若需要记录撤回原因、替代 bundle 或证据，使用已有的 append-only `verdiwm-knowledge-lifecycle` 文档，并把 authority 和 evidence 绑定到 CAS 或其他内容地址。

可以用 CLI 生成生命周期记录：

```bash
verdiwm community lifecycle \
  --action revocation \
  --subject-kind community_bundle \
  --subject-id verdiwm-bundle-0123456789abcdef01234567 \
  --reason "target-side verifier found an invalid claim" \
  --authority-ref cas://sha256/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --evidence-ref cas://sha256/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb \
  --output ./lifecycle/revocation.json
```

`supersession` 还需要同时提供 `--replacement-kind` 和 `--replacement-id`。输出文件是 path-free、幂等且原子写入的；同一路径已有不同内容时会阻断。该命令生成的是治理记录，Bundle 是否真正撤回仍由社区 registry 的 append-only 状态和检索策略决定。

## 从实验到发布

批量模型实验产生的 `execution.json` 是编排状态，不能直接打包为社区知识。它通常包含 campaign 路径、budget 路径和运行时细节。正确的导出边界是：

```text
batch execution
  + settled verifier receipts
  + model portrait / Capability IR
  + IRG / optimization fingerprint
  + portable experience and evidence graph projection
  -> path-free semantic records
  -> portable graph + quality audit
  -> signed community bundle
```

导出时只保留模型家族、能力和画像标识、探针响应摘要、方法或 intervention、effect label、uncertainty、validity boundary、evaluator digest、protocol digest、CAS/URN/SHA-256 evidence refs 和 lifecycle 状态。不要把本地源文件名、checkpoint 路径、数据集路径、GPU worker 命令或未经 settlement 的中间结果写入 records。

检索到的 record 只能产生候选 hypothesis。只有在目标侧使用冻结 verifier 完成 paired trials，并将 positive、null、harmful、interaction、abstention 和 validity boundary 写入新的 evidence record 后，才能把结果提升为 target-confirmed 或 transfer-licensed 知识。相邻 IRG 模型之间的矛盾也应作为新的诊断证据进入本地知识演化流程，而不是在 registry 中静默覆盖旧结论。

## 第一版社区部署边界

第一版采用“本地运行、社区共享 bundle”的模式：用户在自己的环境中接入模型并运行实验，社区 registry 接收签名后的 path-free bundle，提供校验、检索、审核和版本治理。陌生用户提交的 bundle 不会触发共享 GPU worker 执行任意代码；社区服务只处理 JSON、签名和内容地址引用。后续如果要提供托管实验，需要单独设计沙箱、资源配额、镜像白名单、人工审核和 receipt settlement 协议，不能把 registry 上传直接连接到执行器。
