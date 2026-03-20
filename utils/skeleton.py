# -*- coding: utf-8 -*-
"""
HADD 论文表1 骨骼定义（严格对齐原文）
Table 1: Joint hierarchy division rule
Paper: HADD: A Hierarchy-Aware Disentangled Diffusion Framework...
"""
import torch

# ====================== 核心配置（论文表1+原文补充） ======================
NUM_JOINTS = 17    # 论文固定：17个核心关节（最小必要集）
NUM_BONES = 16     # 骨骼数 = 关节数 - 1（根关节无父骨骼）
HIERARCHY_LEVELS = 6  # 论文核心：6层骨骼层级

# ====================== 17关节名称（对应表1+解剖学标准） ======================
# 索引0~16，每个关节严格对应表1的层级划分
JOINT_NAMES = [
    "Pelvis_root",   # 0 - 层级0：根节点（骨盆）
    "Right_hip",     # 1 - 层级1：躯干近端层（右髋）
    "Right_knee",    # 2 - 层级2：躯干&肢体中层（右膝）
    "Right_ankle",   # 3 - 层级3：肢体近端&颈部层（右踝）
    "Left_hip",      # 4 - 层级1：躯干近端层（左髋）
    "Left_knee",     # 5 - 层级2：躯干&肢体中层（左膝）
    "Left_ankle",    # 6 - 层级3：肢体近端&颈部层（左踝）
    "Spine1",        # 7 - 层级1：躯干近端层（脊柱1）
    "Spine2",        # 8 - 层级2：躯干&肢体中层（脊柱2/胸部）
    "Neck",          # 9 - 层级3：肢体近端&颈部层（颈部）
    "Head",          # 10 - 层级4：肢体中层&头部层（头部）
    "Right_shoulder",# 11 - 层级3：肢体近端&颈部层（右肩）
    "Right_elbow",   # 12 - 层级4：肢体中层&头部层（右肘）
    "Right_hand",    # 13 - 层级5：末端肢体层（右手）
    "Left_shoulder", # 14 - 层级3：肢体近端&颈部层（左肩）
    "Left_elbow",    # 15 - 层级4：肢体中层&头部层（左肘）
    "Left_hand"      # 16 - 层级5：末端肢体层（左手）
]

# ====================== 表1核心：6层关节层级划分 ======================
# 索引对应关节0~16，值对应层级0~5（严格遵循表1）
SKELETON_HIERARCHY = [
    0,  # 0: Pelvis_root → 层级0（Root Node Layer）
    1,  # 1: Right_hip → 层级1（Trunk Proximal Layer）
    2,  # 2: Right_knee → 层级2（Trunk & Limb Mid Layer）
    3,  # 3: Right_ankle → 层级3（Limb Proximal & Neck Layer）
    1,  # 4: Left_hip → 层级1（Trunk Proximal Layer）
    2,  # 5: Left_knee → 层级2（Trunk & Limb Mid Layer）
    3,  # 6: Left_ankle → 层级3（Limb Proximal & Neck Layer）
    1,  # 7: Spine1 → 层级1（Trunk Proximal Layer）
    2,  # 8: Spine2 → 层级2（Trunk & Limb Mid Layer）
    3,  # 9: Neck → 层级3（Limb Proximal & Neck Layer）
    4,  # 10: Head → 层级4（Limb Mid & Head Layer）
    3,  # 11: Right_shoulder → 层级3（Limb Proximal & Neck Layer）
    4,  # 12: Right_elbow → 层级4（Limb Mid & Head Layer）
    5,  # 13: Right_hand → 层级5（Terminal Limb Layer）
    3,  # 14: Left_shoulder → 层级3（Limb Proximal & Neck Layer）
    4,  # 15: Left_elbow → 层级4（Limb Mid & Head Layer）
    5   # 16: Left_hand → 层级5（Terminal Limb Layer）
]

# ====================== 运动学树：父关节索引（论文表1隐含结构） ======================
# PARENT[i]：第i根骨骼的父关节索引（骨骼i对应子关节i+1）
# 严格遵循人体正向运动学（根节点→躯干→肢体末端）
PARENT = [
    0,   # 骨骼0：子关节1（Right_hip）→ 父关节0（Pelvis_root）
    1,   # 骨骼1：子关节2（Right_knee）→ 父关节1（Right_hip）
    2,   # 骨骼2：子关节3（Right_ankle）→ 父关节2（Right_knee）
    0,   # 骨骼3：子关节4（Left_hip）→ 父关节0（Pelvis_root）
    4,   # 骨骼4：子关节5（Left_knee）→ 父关节4（Left_hip）
    5,   # 骨骼5：子关节6（Left_ankle）→ 父关节5（Left_knee）
    0,   # 骨骼6：子关节7（Spine1）→ 父关节0（Pelvis_root）
    7,   # 骨骼7：子关节8（Spine2）→ 父关节7（Spine1）
    8,   # 骨骼8：子关节9（Neck）→ 父关节8（Spine2）
    9,   # 骨骼9：子关节10（Head）→ 父关节9（Neck）
    8,   # 骨骼10：子关节11（Right_shoulder）→ 父关节8（Spine2）
    11,  # 骨骼11：子关节12（Right_elbow）→ 父关节11（Right_shoulder）
    12,  # 骨骼12：子关节13（Right_hand）→ 父关节12（Right_elbow）
    8,   # 骨骼13：子关节14（Left_shoulder）→ 父关节8（Spine2）
    14,  # 骨骼14：子关节15（Left_elbow）→ 父关节14（Left_shoulder）
    15   # 骨骼15：子关节16（Left_hand）→ 父关节15（Left_elbow）
]

# ====================== 每个关节的直接子关节（用于HTDM模块） ======================
CHILD_JOINTS = [
    [1, 4, 7],      # 0: Pelvis_root → 右髋、左髋、脊柱1
    [2],            # 1: Right_hip → 右膝
    [3],            # 2: Right_knee → 右踝
    [],             # 3: Right_ankle → 无（末端关节）
    [5],            # 4: Left_hip → 左膝
    [6],            # 5: Left_knee → 左踝
    [],             # 6: Left_ankle → 无（末端关节）
    [8],            # 7: Spine1 → 脊柱2
    [9, 11, 14],    # 8: Spine2 → 颈部、右肩、左肩
    [10],           # 9: Neck → 头部
    [],             # 10: Head → 无（末端关节）
    [12],           # 11: Right_shoulder → 右肘
    [13],           # 12: Right_elbow → 右手
    [],             # 13: Right_hand → 无（末端关节）
    [15],           # 14: Left_shoulder → 左肘
    [16],           # 15: Left_elbow → 左手
    []              # 16: Left_hand → 无（末端关节）
]

# ====================== 层级相关关节三元组（用于HSDM模块） ======================
# 格式：(祖父关节, 父关节, 子关节) → 强化层级空间注意力传播
PARENT_CHILD_TRIPLETS = [
    # 右腿链（层级0→1→2→3→5）
    (0, 1, 2), (1, 2, 3),
    # 左腿链（层级0→1→2→3→5）
    (0, 4, 5), (4, 5, 6),
    # 躯干-头部链（层级0→1→2→3→4）
    (0, 7, 8), (7, 8, 9), (8, 9, 10),
    # 右臂链（层级0→1→2→3→4→5）
    (8, 11, 12), (11, 12, 13),
    # 左臂链（层级0→1→2→3→4→5）
    (8, 14, 15), (14, 15, 16)
]

# ====================== 统一骨骼结构字典（主模型直接调用） ======================
SKELETON_HUMAN36M = {
    "num_joints": NUM_JOINTS,
    "num_bones": NUM_BONES,
    "hierarchy_levels": HIERARCHY_LEVELS,
    "joint_names": JOINT_NAMES,
    "hierarchy": SKELETON_HIERARCHY,
    "parent": PARENT,
    "child_joints": CHILD_JOINTS,
    "triplets": PARENT_CHILD_TRIPLETS
}

# 兼容模型嵌入层的关节-层级映射张量（无需修改）
joint2hier = torch.tensor(SKELETON_HIERARCHY, dtype=torch.long)