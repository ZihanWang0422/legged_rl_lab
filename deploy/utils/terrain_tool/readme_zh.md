# Terrain Tool 中文说明

这个工具把一条 AME parkour 课程写进 MuJoCo XML。默认机器人是 **G1**，输出到：

- `deploy/g1_deploy/assets/scene_terrain.xml`
- `--robot go2` 时输出到 `deploy/go2_deploy/assets/scene_terrain.xml`

默认课程沿世界坐标 `+x` 方向排布，整体结构是：

```text
地面 -> 主上楼梯 -> deck 高度障碍课程 -> 终点平台
```

当前默认课程宽度是 `3.2 m`。主入口楼梯单级高度是 `0.30 m`，课程内 F/G 上下楼梯单级高度是 `0.20 m`。主楼梯 10 级，所以主 deck 顶面高度是 `3.0 m`。

---

## 1. 生成和查看

```bash
conda activate legged_rl_lab
python deploy/utils/terrain_tool/terrain_generator.py --robot g1
python -m mujoco.viewer --mjcf=deploy/g1_deploy/assets/scene_terrain.xml
```

命令行参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--robot {g1,go2}` | `g1` | 选择模板和输出目录 |
| `--input_scene PATH` | 自动选择 | 输入 MuJoCo XML 模板 |
| `--output_scene PATH` | 自动选择 | 输出 XML 路径 |
| `--seed INT` | `7` | 随机种子，只影响 rough / heightfield demo |
| `--side_demos` | 关闭 | 额外生成旧 demo 地形，放在主课程旁边 |

---

## 2. 默认课程与地形参数

默认拓扑如下。每个 `ame_connector_*` 都是完整宽度过渡平台，默认长度 `0.8 m`。

```text
ame_stair_00..09
ame_stair_top
ame_connector_after_stairs
E  concentric_gaps
ame_connector_E_B
B  double_column_stakes
ame_connector_B_D
D  stone_bridge
ame_connector_D_C
C  stepping_stones
ame_connector_C_F
F  stairs_up
ame_connector_F_G
G  stairs_down
ame_connector_G_H
H  radial_plank_bridge
ame_final_platform
```

### 主上楼梯

函数：`AddStairsUpWithPlatform`

从地面连续爬升到主 deck。默认 10 级，每级高 `0.30 m`，顶部接一个平台。

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `init_pos` | `[1.5, 0.0, 0.0]` | 主课程起点 |
| `yaw` | `0.0` | 课程朝向，`0` 表示沿 `+x` |
| `width` | `0.30` | 每级踏面沿 x 深度 |
| `height` | `0.30` | 每级高度 |
| `length` | `3.2` | 楼梯横向宽度，也作为后续 `lane_width` |
| `stair_nums` | `10` | 楼梯级数 |
| `top_width` | `1.2` | 顶平台沿 x 长度 |

### 过渡平台

函数：`_add_top_box`

每两个地形之间的完整宽度平台，用来让机器人从一个障碍稳定过渡到下一个障碍。

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `connector_length` | `0.8` | 过渡平台沿 x 长度 |
| `lane_width` | `3.2` | 平台横向宽度 |
| `thickness` | `0.18` | box 厚度 |
| `deck_height` | `3.0` | 主 deck 顶面高度，由主楼梯返回 |

### E gap

函数：`AddAMEConcentricGaps`

整宽实心平台块和空档交替。当前每段实地 `1.5 m`，gap `0.8 m`，E 段整体加长到 `8.0 m`。

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `segment_length` | `8.0` | E 段总长度 |
| `lane_width` | `3.2` | 地块横向宽度 |
| `ground_width` | `1.50` | 每个实心地块沿 x 长度 |
| `gap_width` | `0.80` | 普通 gap 沿 x 宽度 |
| `second_gap_width` | `0.80` | 第二个 gap 沿 x 宽度；当前和普通 gap 一样 |
| `thickness` | `0.18` | 地块厚度 |

### B 双列石桩

函数：`AddAMEDoubleColumnStakes`

左右两列正方形石桩同步出现，中间留通道。石桩保持正方形。

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `segment_length` | `4.0` | B 段总长度 |
| `lane_width` | `3.2` | 段落横向总宽度 |
| `stake_side` | `0.40` | 单个石桩边长 |
| `stake_gap` | `0.30` | 相邻石桩沿 x 间隔 |
| `column_gap` | `0.60` | 左右两列之间的间距 |
| `thickness` | `0.18` | 石桩厚度 |

### D 石桥

函数：`AddAMEStoneBridge`

沿中心线排布的正方形石块，石块之间有短 gap。这里石块不是长条，保持正方形。

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `segment_length` | `4.0` | D 段总长度 |
| `lane_width` | `3.2` | 段落横向总宽度 |
| `stone_width` | `0.70` | 石块横向宽度 |
| `stone_length` | `0.70` | 石块沿 x 长度，和宽度一致 |
| `stone_distance` | `0.22` | 相邻石块之间的 gap |
| `thickness` | `0.18` | 石块厚度 |

### C 踏石

函数：`AddAMESteppingStones`

正方形踏石左右轻微交替摆放，机器人需要按节奏踩过去。

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `segment_length` | `4.0` | C 段总长度 |
| `lane_width` | `3.2` | 段落横向总宽度 |
| `stone_width` | `0.70` | 踏石边长 |
| `stone_gap` | `0.18` | 相邻踏石沿 x 间隔 |
| `y_center` | `+/-0.18` | 左右交替偏移量，写在函数内部 |
| `thickness` | `0.18` | 踏石厚度 |

### F 上楼梯

函数：`AddAMEStairsUpSegment`

在主 deck 上继续向上爬 6 级，进入更高的平台。每级高度是 `0.20 m`。

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `stair_nums` | `6` | 楼梯级数 |
| `step_width` | `0.30` | 每级踏面沿 x 深度 |
| `step_height` | `0.20` | 每级高度 |
| `lane_width` | `3.2` | 楼梯横向宽度 |
| `thickness` | `0.18` | 每级 box 厚度 |

### G 下楼梯

函数：`AddAMEStairsDownSegment`

和 F 对称下降，回到主 deck 高度。下楼梯是实心连接结构，不是悬空薄板；`stair_nums` 和 `step_height` 要和 F 保持一致，否则高度闭环会失败。

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `stair_nums` | `6` | 楼梯级数 |
| `step_width` | `0.30` | 每级踏面沿 x 深度 |
| `step_height` | `0.20` | 每级下降高度 |
| `solid_height` | 自动计算 | 每一级从低 deck 下方填到当前踏面，保证不镂空 |
| `lane_width` | `3.2` | 楼梯横向宽度 |
| `thickness` | `0.18` | 每级 box 厚度 |

### H 窄板桥

函数：`AddAMERadialPlankBridge`

先给一个正方形中心平台，后面接一条窄长板。

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `segment_length` | `4.0` | H 段总长度 |
| `lane_width` | `3.2` | 段落横向总宽度 |
| `platform_len` | `0.75` | 中心平台边长，写在函数内部 |
| `plank_width` | `0.38` | 窄板横向宽度 |
| `thickness` | `0.18` | 平台和窄板厚度 |

### 终点平台

函数：`_add_top_box`

完整宽度平台，放在所有障碍最后。

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `x_size` | `1.2` | 终点平台沿 x 长度 |
| `y_size` | `3.2` | 终点平台横向宽度 |
| `top_height` | `3.0` | 顶面对齐主 deck |
| `thickness` | `0.18` | 平台厚度 |

### A 交错石桩（备用）

函数：`AddAMEAlternateColumnStakes`

这个函数还保留在脚本里，但已经不在默认课程中调用。它是一排左、一排右交替出现的正方形石桩。

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `segment_length` | `4.0` | A 段总长度 |
| `lane_width` | `3.2` | 段落横向总宽度 |
| `stake_side` | `0.40` | 单个石桩边长 |
| `stake_gap` | `0.30` | 相邻石桩沿 x 间隔 |
| `column_gap` | `0.60` | 左右列之间的间距 |
| `thickness` | `0.18` | 石桩厚度 |
