# Sub-Agent Visualization

> AstrBot 子代理可视化面板 —— 实时甘特图 + 嵌套代理树 + 完整对话记录

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

为 AstrBot 提供 Hermes 风格的子代理运行可视化仪表盘。

---

## ✨ 功能

| 功能 | 说明 |
|:---|:---|
| 📊 **甘特图时间线** | 实时展示子代理执行时间轴与并发情况 |
| 🌳 **嵌套代理树** | 层级化展示子代理树形结构 |
| 💬 **完整对话** | 每个代理的工具调用、结果、推理过程全记录 |
| 📈 **Token 统计** | 输入/输出 token 实时累计 |
| 🔄 **实时刷新** | 面板自动轮询更新 |

---

## 📦 安装

### 方式一：WebUI 安装（推荐）

1. AstrBot 面板 → **插件管理**
2. 从插件市场搜索 `subagent-viz`
3. 点击安装

### 方式二：手动安装

```bash
# 克隆到 AstrBot 插件目录
cd <AstrBot>/data/plugins/
git clone https://github.com/Eason-Mai-bit/astrbot_plugin_subagent_viz.git

# 重启 AstrBot
```

### 方式三：离线安装

下载 Release 中的 `subagent-viz-0.2.0.zip`，解压到 `<AstrBot>/data/plugins/subagent-viz/`。

---

## ⚙️ 配置

在 AstrBot WebUI → **插件管理** → `subagent-viz` → 配置：

| 字段 | 说明 | 默认 |
|:---|:---|:---:|
| `port` | 面板服务端口 | `8642` |
| `host` | 监听地址 | `127.0.0.1` |

安装后在面板菜单中会出现 **「子代理可视化」** 入口。

---

## 📁 目录结构

```
astrbot_plugin_subagent_viz/
├── main.py                 # 插件主程序（仪表盘服务 + 事件监听）
├── metadata.yaml           # 插件元信息
├── _conf_schema.json       # 配置声明
├── LICENSE                 # MIT
├── README.md               # 本文档
└── pages/
    └── dashboard/
        └── index.html     # 可视化面板前端（单文件）
```

---

## 🔌 兼容性

| 项目 | 要求 |
|:---|:---|
| AstrBot | `>= 0.1.0` |
| Python | 3.9+ |
| 依赖 | 见 `requirements`（无额外三方依赖）|

---

## 📄 License

MIT License — 详见 [LICENSE](LICENSE)

---

## ⚠️ 免责声明

1. 本项目仅供学习交流
2. 面板展示的对话内容可能包含敏感信息，**部署到非本机环境前请评估风险**
3. 默认仅监听 `127.0.0.1`（仅本机可访问），如需开放请自行评估安全风险
