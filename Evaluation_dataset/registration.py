import sys
import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import scipy.io as scio
import argparse

# 设置中文字体支持
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]


def checkboard(im1, im2, d=150):
    """生成棋盘格融合图像，用于可视化配准结果"""
    if im1.shape != im2.shape:
        # 确保输入图像尺寸一致
        h = min(im1.shape[0], im2.shape[0])
        w = min(im1.shape[1], im2.shape[1])
        im1 = im1[:h, :w]
        im2 = im2[:h, :w]

    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)
    mask = np.zeros_like(im1)

    # 创建棋盘格掩码
    for i in range(mask.shape[0] // d + 1):
        for j in range(mask.shape[1] // d + 1):
            if (i + j) % 2 == 0:
                y_start, y_end = i * d, min((i + 1) * d, mask.shape[0])
                x_start, x_end = j * d, min((j + 1) * d, mask.shape[1])
                mask[y_start:y_end, x_start:x_end, :] = 1

    # 融合图像
    result = im1 * mask + im2 * (1 - mask)
    return result.astype(np.uint8)


def calculate_rmse(points1, points2):
    """计算两组点之间的均方根误差(RMSE)"""
    if len(points1) != len(points2) or len(points1) == 0:
        return 0.0

    # 计算每个点的欧氏距离
    distances = np.sum((points1 - points2) ** 2, axis=1)
    # 计算均方根误差
    rmse = np.sqrt(np.mean(distances))
    return rmse


def main():
    parser = argparse.ArgumentParser(description='图像配准程序')
    parser.add_argument("--feature_name", type=str, default='SPEM-DSRN', help='特征名称')
    parser.add_argument("--subsets", type=str, default='VIS_SAR',
                        help='模态类型: VIS_NIR, VIS_IR, VIS_SAR, +表示所有')
    parser.add_argument("--nums_kp", type=int, default=-1, help="用于评估的特征点数量")
    parser.add_argument("--vis_flag", type=bool, default=True, help="是否可视化结果")
    args = parser.parse_args()

    # 初始化BF匹配器
    bf = cv2.BFMatcher(crossCheck=True)

    # 配置参数
    MIN_MATCH_COUNT = 5
    REPROJ_THRESHOLD = 3.0  # 重投影误差阈值，用于判断配准成功与否
    num_black_list = 0

    # 处理数据集
    if args.subsets == '+':
        subsets = ['VIS_IR', 'VIS_NIR', 'VIS_SAR']
    else:
        subsets = [args.subsets]

    # 处理关键点数量
    if args.nums_kp < 0:
        nums_kp = [4096]
    else:
        nums_kp = [args.nums_kp]

    feature_name = args.feature_name
    vis_flag = args.vis_flag
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    for subset in subsets:
        subset_path = os.path.join(SCRIPT_DIR, subset)
        if not os.path.exists(subset_path):
            print(f"警告: 数据集路径 {subset_path} 不存在，跳过该子集")
            continue

        dirlist = os.listdir(subset_path)
        if 'test' not in dirlist:
            print(f"警告: 子集中未找到test文件夹，跳过该子集 {subset}")
            continue

        # 构建图像路径
        modal1, modal2 = subset.split('_')
        filepath1 = os.path.join(subset_path, 'test', modal1)
        filepath2 = os.path.join(subset_path, 'test', modal2)

        if not os.path.exists(filepath1) or not os.path.exists(filepath2):
            print(f"警告: 图像路径不存在，跳过该子集 {subset}")
            continue

        print(f"\n处理子集: {subset}")
        print(f"模态1路径: {filepath1}")
        print(f"模态2路径: {filepath2}")

        for num in nums_kp:
            print(f"\n使用关键点数量: {num}")
            errs = []
            rmses = []  # 存储RMSE结果
            inlier_ratios = []  # 存储内点比例
            failed_id = []
            image_list = sorted(os.listdir(filepath1))
            img_list_whitelist = []

            # 创建结果保存目录
            result_dir = os.path.join(SCRIPT_DIR, 'results', subset, feature_name)
            reproj_dir = os.path.join(result_dir, 'regist', str(num))
            os.makedirs(reproj_dir, exist_ok=True)

            progress_bar = tqdm(range(len(image_list)), desc=f"处理图像")
            for id in progress_bar:
                img_name = image_list[id]
                imgpath1 = os.path.join(filepath1, img_name)
                imgpath2 = os.path.join(filepath2, img_name)

                # 检查图像文件是否存在
                if not os.path.exists(imgpath1) or not os.path.exists(imgpath2):
                    print(f"警告: 图像文件不存在 {img_name}，跳过")
                    continue

                # 加载图像
                try:
                    image1 = np.array(Image.open(imgpath1).convert('RGB'))
                    image2 = np.array(Image.open(imgpath2).convert('RGB'))
                except Exception as e:
                    print(f"加载图像 {img_name} 失败: {str(e)}，跳过")
                    continue

                # 加载特征文件
                feat_file = img_name.replace('.png', '.features.mat')
                feat_path = os.path.join(SCRIPT_DIR, 'features', subset, feature_name, feat_file)
                if not os.path.exists(feat_path):
                    print(f"特征文件 {feat_path} 不存在，跳过")
                    continue

                try:
                    feats = scio.loadmat(feat_path)
                    desc1 = np.array(feats['desc1'], dtype=np.float32)[:num]
                    desc2 = np.array(feats['desc2'], dtype=np.float32)[:num]
                    kp1 = np.array(feats['kp1'][:, 0:2], dtype=np.float32)[:num]
                    kp2 = np.array(feats['kp2'][:, 0:2], dtype=np.float32)[:num]
                except Exception as e:
                    print(f"加载特征 {feat_file} 失败: {str(e)}，跳过")
                    continue

                # 加载地标点（如果存在）
                landmarks = None
                lm_path = os.path.join(subset_path, 'test', 'landmarks', img_name.replace('.png', '.lms.mat'))
                if os.path.exists(lm_path):
                    try:
                        landmarks = scio.loadmat(lm_path)
                        vis_lm = np.array(landmarks['vis_points'])
                        ir_lm = np.array(landmarks['ir_points'])
                        if len(ir_lm) < 5:
                            num_black_list += 1
                            continue
                    except Exception as e:
                        print(f"加载地标点 {lm_path} 失败: {str(e)}")
                        landmarks = None
                else:
                    vis_lm = None
                    ir_lm = None

                img_list_whitelist.append(img_name)

                # 加载单应性矩阵
                H = None
                suffix = None
                try:
                    # 尝试两种可能的后缀
                    for s in ['.12', '.21']:
                        H_path = os.path.join(subset_path, 'test', 'transforms',
                                              img_name.replace('.png', s + '.mat'))
                        if os.path.exists(H_path):
                            H = scio.loadmat(H_path)['H']
                            suffix = s
                            break

                    if H is None:
                        print(f"未找到单应性矩阵文件，跳过 {img_name}")
                        continue
                except Exception as e:
                    print(f"加载单应性矩阵失败: {str(e)}，跳过 {img_name}")
                    continue

                # 特征匹配
                try:
                    if suffix == '.21':
                        matches = bf.match(desc1, desc2)
                        src_pts = np.float32([kp1[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp2[m.trainIdx] for m in matches]).reshape(-1, 1, 2)
                        src_im = image2
                        gt_im = image1

                        # 处理地标点
                        if landmarks is not None:
                            lm_gt = cv2.perspectiveTransform(ir_lm.reshape(-1, 1, 2), H)
                            lm_src = vis_lm.reshape(-1, 1, 2)
                    else:  # .12
                        matches = bf.match(desc2, desc1)
                        src_pts = np.float32([kp2[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp1[m.trainIdx] for m in matches]).reshape(-1, 1, 2)
                        src_im = image1
                        gt_im = image2

                        # 处理地标点
                        if landmarks is not None:
                            lm_gt = cv2.perspectiveTransform(vis_lm.reshape(-1, 1, 2), H)
                            lm_src = ir_lm.reshape(-1, 1, 2)
                except Exception as e:
                    print(f"特征匹配失败: {str(e)}，跳过 {img_name}")
                    continue

                # 估计单应性矩阵并计算误差
                current_rmse = 1000.0  # 初始化一个较大的RMSE值
                current_err = 1000.0  # 用于失败判断的误差值
                inlier_ratio = 0.0  # 内点比例

                if len(matches) > MIN_MATCH_COUNT:
                    try:
                        # 使用RANSAC估计单应性矩阵并获取内点掩码
                        M, mask = cv2.findHomography(
                            src_pts, dst_pts,
                            cv2.RANSAC,
                            ransacReprojThreshold=3.0,
                            maxIters=100000
                        )

                        if M is not None:
                            # 筛选出内点
                            inlier_mask = mask.ravel() == 1
                            src_pts_inliers = src_pts[inlier_mask]
                            dst_pts_inliers = dst_pts[inlier_mask]

                            # 计算内点比例
                            inlier_ratio = np.sum(inlier_mask) / len(inlier_mask)

                            # 使用所有内点重新计算单应性矩阵（最小二乘拟合）
                            if len(src_pts_inliers) > MIN_MATCH_COUNT:
                                M_refined, _ = cv2.findHomography(
                                    src_pts_inliers, dst_pts_inliers,
                                    method=0  # 使用所有点进行最小二乘拟合
                                )
                                if M_refined is not None:
                                    M = M_refined

                            # 计算配准后的图像
                            warp_im = cv2.warpPerspective(gt_im, M, (gt_im.shape[1], gt_im.shape[0]))

                            # 计算重投影误差（无论是否有地标点，都计算）
                            reproj_pts = cv2.perspectiveTransform(src_pts_inliers, M)
                            current_rmse = calculate_rmse(reproj_pts.reshape(-1, 2), dst_pts_inliers.reshape(-1, 2))
                            current_err = current_rmse  # 使用RMSE作为误差指标

                            # 可视化棋盘格结果
                            if vis_flag:
                                im_cb = checkboard(warp_im, src_im)
                                save_path = os.path.join(reproj_dir, img_name)
                                Image.fromarray(im_cb).save(save_path)
                        else:
                            current_rmse = 1000.0
                            current_err = 1000.0
                    except Exception as e:
                        print(f"配准过程出错: {str(e)}，跳过 {img_name}")
                        current_rmse = 1000.0
                        current_err = 1000.0
                else:
                    current_rmse = 1000.0
                    current_err = 1000.0

                # 记录结果
                errs.append(current_err)
                rmses.append(current_rmse)
                inlier_ratios.append(inlier_ratio)

                # 更新进度条信息
                progress_bar.set_postfix(
                    {"当前图像": img_name, "RMSE": f"{current_rmse:.2f}", "内点比例": f"{inlier_ratio:.2f}"})

                # 记录失败案例
                if current_err >= REPROJ_THRESHOLD:
                    failed_id.append(img_name)

            # 处理结果
            errs = np.array(errs)
            rmses = np.array(rmses)
            inlier_ratios = np.array(inlier_ratios)

            # 使用重投影误差阈值和内点比例判断成功
            success_mask = (rmses < REPROJ_THRESHOLD) & (inlier_ratios > 0.25)  # 内点比例大于25%
            success_count = np.sum(success_mask)

            print(f"\n配准结果统计 - 关键点数量: {num}")
            print(f"总处理图像数: {len(image_list)}")
            print(f"成功配准图像数: {success_count}")
            print(f"失败配准图像数: {len(image_list) - success_count}")

            # 计算平均RMSE（仅考虑成功的配准）
            if success_count > 0:
                avg_rmse = np.mean(rmses[success_mask])
                avg_inlier_ratio = np.mean(inlier_ratios[success_mask])
                print(f"平均配准RMSE: {avg_rmse:.2f}")
                print(f"平均内点比例: {avg_inlier_ratio:.2f}")
            else:
                avg_rmse = 0.0
                avg_inlier_ratio = 0.0
                print("所有图像配准失败，无法计算平均RMSE和内点比例")

            # 保存结果
            scio.savemat(
                os.path.join(result_dir, f'regist_result_{num}.mat'),
                {
                    'image_names': img_list_whitelist,
                    'errors': errs,
                    'rmse_values': rmses,
                    'inlier_ratios': inlier_ratios,
                    'average_rmse': avg_rmse,
                    'average_inlier_ratio': avg_inlier_ratio
                }
            )

            # 写入日志
            log_file_path = os.path.join(result_dir, 'regist_log.txt')
            with open(log_file_path, 'a+', encoding='utf-8') as log_file:
                log_file.write(f"\n{'=' * 50}\n")
                log_file.write(f"子集: {subset}\n")
                log_file.write(f"特征名称: {feature_name}\n")
                log_file.write(f"关键点数量: {num}\n")
                log_file.write(f"处理日期: {np.datetime64('now')}\n")
                log_file.write(f"总处理图像数: {len(image_list)}\n")
                log_file.write(f"成功配准图像数: {success_count}\n")
                log_file.write(f"失败配准图像数: {len(image_list) - success_count}\n")
                log_file.write(f"平均配准RMSE: {avg_rmse:.2f}\n")
                log_file.write(f"平均内点比例: {avg_inlier_ratio:.2f}\n")
                log_file.write(f"失败图像ID: {', '.join(failed_id)}\n")
                log_file.write(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()