import numpy as np
import os
from scipy.io import savemat


def safe_homography():
    """生成安全的轻微透视变换矩阵"""
    # 基础仿射参数（严格限制范围）
    theta = np.random.uniform(-5, 5)  # 旋转角度±5度
    tx = np.random.uniform(-15, 15)  # 平移量±15像素
    ty = np.random.uniform(-15, 15)
    scale = np.random.uniform(0.95, 1.05)  # 各向同性缩放

    # 构建基础矩阵（旋转->缩放->平移）
    H = np.eye(3)
    H[:2, :2] = np.array([
        [np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))],
        [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))]
    ]) * scale

    H[0, 2] = tx
    H[1, 2] = ty

    # 添加安全透视分量（限制在1e-4量级）
    H[2, 0] = np.random.uniform(-0.0002, 0.0002)  # h31
    H[2, 1] = np.random.uniform(-0.0002, 0.0002)  # h32
    H[2, 2] = np.random.uniform(0.995, 1.005)  # h33

    # 矩阵归一化与校验
    H /= H[2, 2]
    assert np.abs(np.linalg.det(H)) > 0.1, "奇异矩阵警告"
    return H


def generate_homography_matrices(num_pairs, save_dir):
    """生成安全变换矩阵"""
    os.makedirs(save_dir, exist_ok=True)

    for pair_id in range(1, num_pairs + 1):
        # 随机选择变换方向
        direction = np.random.choice(['12', '21'])

        # 生成基础矩阵
        H = safe_homography()
        if direction == '21':
            H = np.linalg.inv(H)
            H /= H[2, 2]  # 保持归一化

        # 保存并打印关键参数
        info = f"det={np.linalg.det(H):.2f} tx={H[0, 2]:.1f} ty={H[1, 2]:.1f}"
        savemat(f"{save_dir}/{pair_id}.{direction}.mat", {'H': H})
        print(f"生成：{pair_id}.{direction}.mat ({info})")


# 示例使用（生成50对）
generate_homography_matrices(424, 'safe_homographies')