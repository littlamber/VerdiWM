# 第三方组件与授权边界

VerdiWM 自有源码按根目录 `LICENSE` 中的 Apache-2.0 分发。该许可不改变
第三方代码、模型、数据和可执行文件的原始许可。

## 当前发布内容

- 工作台的可编辑源码是 `wmloop/web/workbench.*` 与 `wmloop/web/setup.*`。
  页面使用浏览器原生 API，不捆绑 React、路由器、图标库、字体或 CDN 脚本，
  因此无需 Node/npm 构建或前端依赖锁文件。
- 历史 React 编译产物缺少对应源码、依赖锁及完整授权清单，已从源码导出、
  wheel 和 sdist 的发布范围移除，不以可重建前端的名义分发。
- 模型权重、外部模型 checkout、数据集与第三方 vendor 树不随本包分发。
  用户单独获取时应遵循对应上游许可、访问条款和数据使用限制。

## Python 依赖

运行依赖由 `pyproject.toml` 声明，具体版本与文件摘要由 `uv.lock` 记录。
它们由包管理器独立安装，不合并进 VerdiWM 的 Python 源文件。
以下是直接依赖的上游许可概览；精确版本的许可证以相应发行包的
`*.dist-info/licenses/` 或上游 LICENSE 文件为准。

| 组件 | 上游许可概览 | 用途 |
| --- | --- | --- |
| imageio | BSD-2-Clause | 媒体读写 |
| jsonschema | MIT | 契约验证 |
| NumPy | BSD-3-Clause，附带组件另有声明 | 数值计算 |
| Pillow | MIT-CMU，附带编解码库另有声明 | 图像处理 |
| Pydantic | MIT | 数据契约 |
| PyYAML | MIT | 配置解析 |
| cryptography | Apache-2.0 或 BSD-3-Clause，附带组件另有声明 | 签名验证 |
| tenacity | Apache-2.0 | 重试控制 |
| tomli | MIT | 锁文件 TOML 解析 |

`imageio[ffmpeg]` 会引入 `imageio-ffmpeg`；其 Python 包和所提供 FFmpeg
二进制有不同授权边界，FFmpeg 的构建选项可能涉及 LGPL/GPL。若另行打包
容器、完整虚拟环境或可执行文件，必须保留对应发行物的许可证并检查二进制
分发义务，不能直接沿用本项目 Apache-2.0 的声明。

开发工具、构建工具和可选 Torch 测试环境也各自保留上游许可。
新增第三方源码、复制代码或前端依赖时，必须同时提交来源、版本、完整许可
及可重建步骤；不得删除已有上游版权头或用本文件替代上游许可证。
