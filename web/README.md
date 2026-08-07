# 血流麻将 Web 客户端

动作栏直接使用引擎提供的 `legalActionIds`。弃牌响应处于 `HuResponse` 时，动作栏必须同时显示 legal mask 中的胡、碰、直杠和过；客户端不能自行按动作优先级过滤。引擎负责胡的全局优先级和一炮多响结算。

## Bot 策略

Worker 启动时会加载 `model/latest.onnx`，并使用 WASM `RuleNn` 校验 ONNX metadata 和张量契约。校验成功时，设置界面和提示策略会自动提供 `rule-nn`；校验失败时只提供 `rule-fast` 和 `rule-ev`。重新训练后只需替换 `model/latest.onnx` 并重新构建 Web 资源，不需要修改代码。

## 运行客户端

```bash
cd web
npm install
npm run dev
```

生产构建自包含，可由任意静态 Web 服务器托管：

```bash
npm run build
npm run preview
```

构建同时会生成离线 service worker。

## 客户端结构

- `src/engine` 负责 Worker 协议和 WASM 游戏实例。
- `src/scene` 渲染桌面、牌、牌墙、河牌和副露。
- `src/components` 包含 DOM HUD、动作栏、暂停/设置视图以及回放/结算浮层。
- `scripts/extract_assets.py` 将 OpenRiichi 数据目录转换为 `public/assets` 下可复用的浏览器资源。

`scripts/extract_assets.py` 将 OpenRiichi `bin/Data` 目录转换为：

- 仅含几何信息的 GLB 模型；
- WebP 牌面图集和 WebP 界面纹理；
- 中性的 Ogg Opus 音效；
- 界面字体 WOFF2 子集；
- `public/assets/manifest.json`，记录尺寸、牌面映射、来源和许可证信息。

## 依赖要求

- Python 3.10 或更高版本；
- Pillow；
- 支持 WOFF2 的 fonttools；
- 带 `libopus` 编码器的 FFmpeg；
- OpenRiichi 代码检出。

安装 Python 依赖：

```bash
python -m pip install -r web/scripts/requirements.txt
```

从仓库根目录运行转换：

```bash
python web/scripts/extract_assets.py \
  --source /path/to/OpenRiichi/bin/Data \
  --out web/public/assets
```

字体子集包含来自 `GAME_RULES.md` 及 `web/src` 树的 CJK 文本。可使用 `--charset-source PATH`（可多次指定）将其他文件或目录中的文本纳入子集。

生成的资源派生自 OpenRiichi，仍须遵守 `public/assets/manifest.json` 中记录的相关声明。
