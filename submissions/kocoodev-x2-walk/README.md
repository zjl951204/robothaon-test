# X2 Humanoid Walking Demo

## 项目名称

X2 Humanoid Walking Demo

## 使用的机器人本体

Agibot X2（`assets/x2/scene.xml`）

## 任务目标

在 MuJoCo 中加载 X2 人形机器人，通过周期性双足步态与骨盆前移实现向前行走动画，并导出 demo 视频。

## 技术方案

- 使用仓库自带的 X2 MJCF 场景
- 左右腿交替摆动 + 手臂 counter-swing
- 骨盆沿 X 轴平滑前移，形成连续行走效果

## 核心功能

- 双足交替步态与 forward locomotion
- 轨迹采样与 JSON 导出
- MP4 demo 视频导出

## 运行方式

在仓库根目录执行：

```bash
python3 -m pip install -r requirements.txt
python submissions/kocoodev-x2-walk/run_walk.py
```

可选参数：

```bash
python submissions/kocoodev-x2-walk/run_walk.py --duration 8 --fps 30 --walk-speed 0.95
```

输出：

- `submissions/kocoodev-x2-walk/demo.mp4`
- `submissions/kocoodev-x2-walk/trajectory.json`

## 项目亮点

- 基于仓库现有 X2 模型，无需额外下载资源
- 代码单文件可运行，便于评审复现
- 包含 demo 视频与轨迹数据

## 当前限制

- 步态为开环周期控制，骨盆前移用于展示连续行走轨迹

## 未来改进方向

- 加入 IMU / 接触传感器闭环平衡控制
- 支持键盘或手柄实时遥操
- 增加地形与障碍物场景

## Demo 视频

见同目录 `demo.mp4`。
