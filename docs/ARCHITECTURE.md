# SyncTalk 数字人直播软件 — 技术方案

## 一、产品定位

**一句话描述**：用户在自己的电脑上运行数字人，通过虚拟摄像头接入任何直播/会议软件，实现实时口型同步的数字人直播。

**核心成本模型**：

| 项目 | 传统云端方案 | 本方案（本地推理） |
|------|------------|-----------------|
| GPU 服务器 | ¥5-20/小时/用户 | ¥0（用户自有显卡） |
| 带宽 | 视频流上下行 | 仅 API 元数据（KB 级） |
| 云端存储 | 视频/模型存储 | 模型存 CDN，视频在本地 |
| 月运营成本 | ¥数万起 | ¥数百（一台轻量 API 服务器） |

**关键假设**：用户有 NVIDIA GTX 1060+ 以上显卡（实测 UNet 328px 推理 ~15ms/帧，满足 25fps 实时需求）。

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        用户本地 (Windows/Mac)                        │
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌────────────────┐  │
│  │ 桌面 GUI │──→│ 实时管线 │──→│ 虚拟摄像头 │──→│ OBS/Zoom/抖音  │  │
│  │ (Electron│   │ (Python) │   │(pyvirtualcam│   │ 钉钉/腾讯会议  │  │
│  │  /PyQt)  │   │          │   │ /v4l2)     │   │                │  │
│  └─────┬────┘   └────┬─────┘   └───────────┘   └────────────────┘  │
│        │             │                                              │
│        │        ┌────┴─────┐                                        │
│        │        │ 本地模型 │  ← 首次从 CDN 下载                      │
│        │        │ 本地角色 │  ← 训练后的数据                         │
│        │        └──────────┘                                        │
│        │                                                            │
└────────┼────────────────────────────────────────────────────────────┘
         │ HTTPS (仅元数据, KB级)
         │
┌────────┴────────────────────────────────────────────────────────────┐
│                     云端 (轻量级 API 服务器)                         │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐            │
│  │ Auth Service  │  │ Meta Store   │  │ Model Registry │            │
│  │ (JWT 登录)    │  │ (PostgreSQL) │  │ (CDN/OSS)      │            │
│  └──────────────┘  └──────────────┘  └────────────────┘            │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐                                │
│  │ License Mgmt │  │ Analytics    │                                │
│  │ (授权/激活)   │  │ (使用统计)   │                                │
│  └──────────────┘  └──────────────┘                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三、模块设计

### 3.1 云端服务（轻量 API，一台 2C4G 服务器即可）

#### 3.1.1 数据模型

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐
│   users      │     │   characters     │     │   licenses       │
├─────────────┤     ├─────────────────┤     ├──────────────────┤
│ id (PK)     │────<│ id (PK)         │     │ id (PK)          │
│ email       │     │ user_id (FK)    │     │ user_id (FK)     │
│ password_hash│     │ name            │     │ license_key      │
│ display_name│     │ resolution      │     │ plan (free/pro/biz)│
│ plan        │     │ asr_mode        │     │ activated_at     │
│ created_at  │     │ checkpoint_hash │     │ expires_at       │
│ last_login  │     │ frame_count     │     │ machine_id       │
│ machine_id  │     │ thumbnail_url   │     │ status           │
└─────────────┘     │ model_url (CDN) │     └──────────────────┘
                    │ status          │
                    │ created_at      │     ┌──────────────────┐
                    │ file_size_mb    │     │ usage_logs       │
                    └─────────────────┘     ├──────────────────┤
                                            │ id (PK)          │
                                            │ user_id (FK)     │
                                            │ character_id (FK)│
                                            │ session_start    │
                                            │ session_end      │
                                            │ duration_minutes │
                                            │ fps_avg          │
                                            │ gpu_model        │
                                            └──────────────────┘
```

#### 3.1.2 API 端点

```
# 认证
POST   /api/v1/auth/register          注册
POST   /api/v1/auth/login             登录 → JWT
POST   /api/v1/auth/refresh           刷新 Token
GET    /api/v1/auth/me                当前用户信息

# 角色管理
GET    /api/v1/characters              我的角色列表
POST   /api/v1/characters              创建角色（上传训练数据后）
GET    /api/v1/characters/{id}         角色详情
DELETE /api/v1/characters/{id}         删除角色
GET    /api/v1/characters/{id}/download 获取模型下载链接（签名 URL）

# 授权
POST   /api/v1/license/activate        激活设备
GET    /api/v1/license/status           授权状态
POST   /api/v1/license/deactivate      取消设备绑定

# 模型市场（可选）
GET    /api/v1/marketplace              公开角色列表
GET    /api/v1/marketplace/{id}/download 下载公开角色

# 使用统计
POST   /api/v1/usage/heartbeat          心跳上报（每5分钟）
GET    /api/v1/usage/summary            使用汇总
```

#### 3.1.3 传输数据量估算

| 操作 | 数据量 | 频率 |
|------|--------|------|
| 登录 | ~1 KB | 每次启动 |
| 角色列表 | ~5 KB | 启动时 |
| 模型下载 | 50-100 MB | 首次/更新 |
| 心跳上报 | ~0.5 KB | 每 5 分钟 |
| **直播期间** | **~6 KB/小时** | **极低** |

### 3.2 本地客户端

#### 3.2.1 启动流程

```
App 启动
  │
  ├─ 1. 检查本地授权缓存 (license.json)
  │     ├─ 有效 → 继续
  │     └─ 无效/过期 → 显示登录界面
  │
  ├─ 2. 登录/注册 → 获取 JWT Token
  │     └─ 验证授权 → 获取 License 状态
  │
  ├─ 3. 同步角色列表 (本地 ↔ 云端)
  │     ├─ 新角色 → 从 CDN 下载模型文件
  │     └─ 已有角色 → 校验 checksum
  │
  ├─ 4. 检测 GPU → 显示性能信息
  │
  └─ 5. 进入主界面
        ├─ 角色选择
        ├─ 音频源选择（麦克风/系统音频）
        ├─ 虚拟摄像头配置
        └─ [开始直播] 按钮
```

#### 3.2.2 目录结构

```
SyncTalk/                          # 安装目录
├── SyncTalk.exe                   # 主程序
├── config.yaml                    # 用户配置
├── license.json                   # 本地授权缓存
│
├── models/                        # 预训练基础模型
│   └── audio_visual_encoder.pth
│
├── characters/                    # 角色数据（按用户下载）
│   ├── character_001/
│   │   ├── meta.json              # 角色元数据
│   │   ├── model.pth              # UNet 权重
│   │   ├── frames/                # 预处理帧
│   │   └── landmarks/             # 关键点数据
│   └── character_002/
│       └── ...
│
└── logs/                          # 运行日志
```

#### 3.2.3 核心模块映射（基于现有工程）

| 功能 | 现有模块 | 改造点 |
|------|---------|--------|
| 实时推理 | `synctalk/realtime/pipeline.py` | ✅ 已实现 |
| 音频捕获 | `synctalk/realtime/audio_stream.py` | ✅ 已实现 |
| 虚拟摄像头 | `synctalk/realtime/virtual_camera.py` | ✅ 已实现 |
| 角色加载 | `synctalk/realtime/character.py` | 需增加云端同步 |
| 模型推理 | `synctalk/models/unet.py` | ✅ 已实现 |
| 配置管理 | `synctalk/configs/base.py` | 需增加用户配置 |
| 认证系统 | 新增 | `synctalk/auth/` |
| 桌面 GUI | 新增 | `synctalk/gui/` |
| 云端同步 | 部分 (`synctalk/sdk.py`) | 需扩展 |

---

## 四、需要新增的模块

### 4.1 新增模块清单

```
synctalk/
├── auth/                          # 【新增】认证和授权
│   ├── __init__.py
│   ├── client.py                  # 云端认证客户端
│   ├── license.py                 # 本地授权管理
│   └── models.py                  # 用户/授权数据模型
│
├── cloud/                         # 【新增】云端同步
│   ├── __init__.py
│   ├── sync.py                    # 角色同步（本地↔云端）
│   ├── downloader.py              # 模型下载（断点续传）
│   └── heartbeat.py               # 使用心跳上报
│
├── gui/                           # 【新增】桌面 GUI
│   ├── __init__.py
│   ├── app.py                     # 主窗口
│   ├── login_view.py              # 登录/注册界面
│   ├── main_view.py               # 主控制面板
│   ├── character_view.py          # 角色管理界面
│   ├── settings_view.py           # 设置界面
│   └── assets/                    # 图标/样式
│
├── server/                        # 【新增】云端 API（独立部署）
│   ├── __init__.py
│   ├── app.py                     # FastAPI 服务端
│   ├── auth.py                    # JWT 认证
│   ├── database.py                # PostgreSQL 连接
│   ├── models.py                  # SQLAlchemy ORM
│   ├── routes/
│   │   ├── auth.py                # 认证路由
│   │   ├── characters.py          # 角色管理路由
│   │   ├── license.py             # 授权路由
│   │   └── usage.py               # 使用统计路由
│   └── alembic/                   # 数据库迁移
│
└── packaging/                     # 【新增】打包分发
    ├── build_windows.py           # PyInstaller Windows 打包
    ├── build_macos.py             # macOS .app 打包
    ├── installer.nsi              # NSIS Windows 安装程序
    └── icon.ico                   # 应用图标
```

### 4.2 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| **桌面 GUI** | PyQt6 | 跨平台，Python 原生，GPU 预览可嵌入 |
| **云端 API** | FastAPI + PostgreSQL | 已有 FastAPI 基础，轻量高效 |
| **ORM** | SQLAlchemy 2.0 + Alembic | Python 标准 ORM，迁移管理 |
| **认证** | JWT (PyJWT) | 无状态，客户端友好 |
| **模型存储** | 阿里云 OSS / AWS S3 | CDN 加速下载 |
| **客户端打包** | PyInstaller | Python → 单文件 exe |
| **安装程序** | NSIS (Win) / DMG (Mac) | 行业标准 |
| **虚拟摄像头** | pyvirtualcam | 已集成，跨平台 |
| **GPU 加速** | CUDA + TensorRT (可选) | 最大化推理速度 |

---

## 五、实施路线图

### Phase 1：云端 API + 认证（1-2 周）

```
目标：用户可以注册、登录、管理角色元数据

交付物：
- PostgreSQL 数据模型（users, characters, licenses, usage_logs）
- FastAPI 服务端（auth, characters, license 路由）
- JWT 认证
- 本地授权缓存
- 客户端 SDK 扩展
```

### Phase 2：桌面 GUI（2-3 周）

```
目标：用户通过 GUI 完成完整使用流程

交付物：
- PyQt6 登录界面
- 主控制面板（角色选择、音频源、虚拟摄像头）
- 实时预览窗口
- 性能监控面板（FPS、GPU 温度、延迟）
- 系统托盘
```

### Phase 3：模型分发 + 同步（1-2 周）

```
目标：角色模型通过 CDN 分发，支持断点续传

交付物：
- OSS/S3 模型上传工具
- 客户端断点续传下载器
- 角色数据本地↔云端同步
- 模型版本管理
```

### Phase 4：打包分发（1-2 周）

```
目标：用户可以下载安装包一键安装

交付物：
- PyInstaller Windows 打包
- NSIS 安装程序
- macOS .app 打包
- 自动更新机制
```

---

## 六、成本估算

### 云端运营成本（月）

| 项目 | 规格 | 月费 |
|------|------|------|
| API 服务器 | 2C4G (阿里云 ECS) | ¥150 |
| PostgreSQL | RDS 基础版 | ¥100 |
| OSS 存储 | 100GB | ¥10 |
| CDN 流量 | 100GB/月 | ¥20 |
| 域名 + SSL | — | ¥10 |
| **合计** | | **¥290/月** |

### 对比

| 方案 | 100 用户/月 | 1000 用户/月 | 10000 用户/月 |
|------|-----------|------------|-------------|
| **本方案（本地推理）** | ¥290 | ¥500 | ¥2,000 |
| 云端 GPU 推理 | ¥15,000 | ¥150,000 | ¥1,500,000 |

> **成本降低 50-750 倍**，且随用户增长几乎只增加 CDN 带宽成本。

---

## 七、商业模式建议

| 方案 | Free | Pro (¥99/月) | Business (¥499/月) |
|------|------|-------------|-------------------|
| 角色数量 | 1 个 | 5 个 | 无限 |
| 分辨率 | 160px | 328px | 328px + 自定义 |
| 直播时长 | 2 小时/天 | 无限 | 无限 |
| 水印 | 有 | 无 | 无 |
| 虚拟摄像头 | ✓ | ✓ | ✓ |
| 模型市场 | 下载 | 下载+上传 | 下载+上传+销售 |
| 技术支持 | 社区 | 邮件 | 专属 |
