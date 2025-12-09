import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from scipy.io import savemat
import numpy as np
import cv2
import os
from PIL import Image
from tqdm import tqdm
from skimage.feature import match_descriptors
import scipy.io as scio
import argparse

blacklist_SAR = ['89.png', '87.png', '105.png', '129.png']

parser = argparse.ArgumentParser()
parser.add_argument("--feature_name", type=str, default='SPEM-DSRN', help='Name of feature')
parser.add_argument("--subsets", type=str, default='VIS_SAR',
                    help="Type of modal: VIS_NIR, VIS_IR, VIS_SAR, '+' for all")
parser.add_argument("--nums_kp", type=int, default=-1, help="Number of feature for evaluation")
parser.add_argument("--vis_flag", type=bool, default=True, help="Visualization flag")
args = parser.parse_args()

bf = cv2.BFMatcher(crossCheck=True)

lm_counter = 0
MIN_MATCH_COUNT = 5
num_black_list = 0

if args.subsets == '+':
    subsets = ['VIS_IR', 'VIS_NIR', 'VIS_SAR']
else:
    subsets = [args.subsets]

if args.nums_kp < 0:
    nums_kp = [1024,2048,4096]
else:
    nums_kp = [args.nums_kp]

feature_name = args.feature_name
vis_flag = args.vis_flag

for subset in subsets:
    subset_path = os.path.join(SCRIPT_DIR, subset)
    dirlist = os.listdir(subset_path)
    if 'test' in dirlist:
        imgs = os.listdir(os.path.join(subset_path, 'test', 'VIS'))
    else:
        continue
    print(subset)
    filepath1 = os.path.join(subset_path, 'test', subset.split('_')[0])
    filepath2 = os.path.join(subset_path, 'test', subset.split('_')[1])

    for num in [1024,2048,4096]:
        N_k1 = []
        N_k2 = []
        N_corr = []
        N_corretmatches = []
        N_in_corretmatches = []
        N_k1_ol = []
        N_k2_ol = []
        N_corr_thres = []
        N_corretmatches_thres = []
        image_list = sorted(os.listdir(filepath1))
        img_list_whitelist = []
        progress_bar = tqdm(range(len(image_list)))

        for id in progress_bar:
            if subset == 'SAR' and image_list[id] in blacklist_SAR:
                continue
            else:
                img_list_whitelist.append(image_list[id])

            imgpath1 = os.path.join(filepath1, image_list[id])
            imgpath2 = os.path.join(filepath2, image_list[id])
            image1 = np.array(Image.open(imgpath1).convert('RGB'))
            image2 = np.array(Image.open(imgpath2).convert('RGB'))

            ff = image_list[id].replace('.png', '.features.mat')
            feats = scio.loadmat(os.path.join(SCRIPT_DIR, 'features', subset, feature_name, ff))
            desc1 = feats['desc1']
            desc2 = feats['desc2']
            kp1 = feats['kp1'][:, 0:2]
            kp2 = feats['kp2'][:, 0:2]

            # 尝试加载图像间的单应性矩阵H，计算掩码和关键点变换坐标
            try:
                suffix = '.12'
                H_path = os.path.join(subset_path, 'test', 'transforms',
                                      image_list[id].replace('.png', suffix + '.mat'))
                H = scio.loadmat(H_path)['H']
                ones = np.ones_like(image1)
                mask = cv2.warpPerspective(ones, H, [ones.shape[1], ones.shape[0]])
                mask_1 = mask > 0.5
                mask_1 = mask_1 * 1.0
                mask = cv2.warpPerspective(mask_1, np.linalg.inv(H), [mask.shape[1], mask.shape[0]])
                mask_2 = mask > 0.5
                ones = np.ones([np.size(kp2, 0), 1])
                kp_2_warped = np.hstack([kp2, ones])
                kp_2_warped = H @ kp_2_warped.transpose()
                kp_2_warped = kp_2_warped / kp_2_warped[2, :]
                kp_2_warped = kp_2_warped[0:2, :].transpose()
                kp_1_warped = kp1
            except:
                suffix = '.21'
                H_path = os.path.join(subset_path, 'test', 'transforms',
                                      image_list[id].replace('.png', suffix + '.mat'))
                H = scio.loadmat(H_path)['H']
                ones = np.ones_like(image1)
                mask = cv2.warpPerspective(ones, H, [ones.shape[1], ones.shape[0]])
                mask_2 = mask > 0.5
                mask_2 = mask_2 * 1.0
                mask = cv2.warpPerspective(mask_2, np.linalg.inv(H), [mask.shape[1], mask.shape[0]])
                mask_1 = mask > 0.5
                ones = np.ones([np.size(kp1, 0), 1])
                kp_1_warped = np.hstack([kp1, ones])
                kp_1_warped = H @ kp_1_warped.transpose()
                kp_1_warped = kp_1_warped / kp_1_warped[2, :]
                kp_1_warped = kp_1_warped[0:2, :].transpose()
                kp_2_warped = kp2

            N_k1.append(kp1[0:num].shape[0])
            N_k2.append(kp2[0:num].shape[0])

            # 计算重叠区域内的关键点数量
            overlap1 = 0
            for kp in kp1[0:num]:
                x = int(kp[0] + 0.5)
                y = int(kp[1] + 0.5)
                if mask_1[(y, x)].sum(axis=-1) > 0.5:
                    overlap1 += 1
            N_k1_ol.append(overlap1)

            overlap2 = 0
            for kp in kp2[0:num]:
                x = int(kp[0] + 0.5)
                y = int(kp[1] + 0.5)
                if mask_2[((y, x))].sum(axis=-1) > 0.5:
                    overlap2 += 1
            N_k2_ol.append(overlap2)

            kp_1_warped_ = kp_1_warped[0:num][:, :2].reshape(-1, 1, 2)
            kp_2_warped_ = kp_2_warped[0:num][:, :2].reshape(1, -1, 2)
            dist_k = ((kp_1_warped_ - kp_2_warped_) ** 2).sum(axis=2)

            # 双向匹配特征描述符
            matches = match_descriptors(desc1[0:num], desc2[0:num], cross_check=True)
            keypoints_left = kp_1_warped[0:num][matches[:, 0], :2]
            keypoints_right = kp_2_warped[0:num][matches[:, 1], :2]
            dif = (keypoints_left - keypoints_right)
            dist_m = dif[:, 0] ** 2 + dif[:, 1] ** 2

            # 处理不同距离阈值
            for thres in range(1, 11):
                # 统计对应点对（NC）
                n_corr = ((dist_k <= thres ** 2).sum(axis=1) > 0.9).sum()
                N_corr_thres.append(n_corr.item())
                if thres == 5:
                    N_corr.append(n_corr.item())

                inds = dist_m <= thres ** 2
                N_corretmatches_thres.append(inds.sum())
                if thres == 5:
                    N_corretmatches.append(inds.sum())

            # 可视化匹配结果（保持原有逻辑）
            if vis_flag and num == 1024:
                h1, w1 = image1.shape[:2]
                h2, w2 = image2.shape[:2]
                vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
                vis[:h1, :w1] = image1
                vis[:h2, w1:w1 + w2] = image2

                # 绘制所有匹配点的关键点
                for match in matches:
                    idx1, idx2 = match[0], match[1]
                    pt1 = tuple(map(int, kp1[idx1]))
                    pt2 = tuple(map(int, kp2[idx2]))
                    pt2 = (pt2[0] + w1, pt2[1])
                    cv2.circle(vis, pt1, 3, (0, 0, 255), 1)  # 红色：图像1关键点
                    cv2.circle(vis, pt2, 3, (0, 255, 0), 1)  # 绿色：图像2关键点

                # 绘制正确匹配点的连接线
                correct_matches = matches[inds]
                for match in correct_matches:
                    idx1, idx2 = match[0], match[1]
                    pt1 = tuple(map(int, kp1[idx1]))
                    pt2 = tuple(map(int, kp2[idx2]))
                    pt2 = (pt2[0] + w1, pt2[1])
                    cv2.line(vis, pt1, pt2, (0, 225, 255), 1)  # 黄色：正确匹配连线

                # 保存可视化结果
                save_dir = os.path.join(SCRIPT_DIR, 'results', subset, feature_name, 'match_vis')
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'match_{image_list[id]}')
                cv2.imwrite(save_path, vis)

        # 转换为numpy数组便于计算
        N_corr_thres = np.array(N_corr_thres)
        N_corr = np.array(N_corr) * 1.0
        N_k1 = np.array(N_k1) * 1.0
        N_k2 = np.array(N_k2) * 1.0
        N_k1_ol = np.array(N_k1_ol)
        N_k2_ol = np.array(N_k2_ol)
        N_corretmatches = np.array(N_corretmatches) * 1.0
        N_corretmatches_thres = np.array(N_corretmatches_thres)

        # 计算评估指标
        CMR = np.divide(N_corretmatches, N_corr, out=np.zeros_like(N_corretmatches), where=N_corr != 0) * 100
        average_CMR = CMR.mean()
        RR = N_corr * 1.0 / np.array([N_k1, N_k2]).min(axis=0)

        # 处理MS计算中的0值避免除以0
        N_k1_ol_temp = N_k1_ol.copy()
        N_k1_ol_temp[N_k1_ol_temp < 0.1] = 1
        N_k2_ol_temp = N_k2_ol.copy()
        N_k2_ol_temp[N_k2_ol_temp < 0.1] = 1
        MS = (N_corretmatches / N_k1_ol_temp + N_corretmatches / N_k2_ol_temp) / 2

        # 保存结果到mat文件
        save_dir_mat = os.path.join(SCRIPT_DIR, 'results', subset, feature_name)
        os.makedirs(save_dir_mat, exist_ok=True)
        savemat(
            os.path.join(save_dir_mat, f'match_result_{num}.mat'),
            {
                'N_corr': N_corr, 'N_k1': N_k1, 'N_k2': N_k2,
                'N_correctmatches': N_corretmatches, 'N_k1_ol': N_k1_ol,
                'N_k2_ol': N_k2_ol, 'N_correctmatches_thres': N_corretmatches_thres,
                'N_corr_thres': N_corr_thres, 'CMR': CMR,
            }
        )

        # 打印结果
        print('=' * 50)
        print(f'Subset: {subset}, Feature: {feature_name}, Keypoints: {num}')
        print('Number of sar keypoints: {:.2f}.'.format(np.mean(N_k1)))
        print('Number of visible keypoints: {:.2f}.'.format(np.mean(N_k2)))
        print('Number of correspondence (NC): {:.2f}.'.format(N_corr.mean()))
        print('Number of correct matches (NCM): {:.2f}.'.format(np.mean(N_corretmatches)))
        print('Correct match rate (CMR): {:.2f}%.'.format(average_CMR))
        print('Repeatability rate (RR): {:.4f}.'.format(RR.mean()))
        print('Matching score (MS): {:.4f}.'.format(MS.mean()))
        print('=' * 50)

        # 写入日志文件
        log_file_path = os.path.join(save_dir_mat, 'match_log.txt')
        with open(log_file_path, 'a+', encoding='utf-8') as log_file:  # 使用with语句自动关闭文件
            log_file.write('=' * 50 + '\n')
            log_file.write(f'Subset: {subset}, Feature: {feature_name}, Keypoints: {num}\n')
            log_file.write('Number of sar keypoints: {:.2f}.\n'.format(np.mean(N_k1)))
            log_file.write('Number of visible keypoints: {:.2f}.\n'.format(np.mean(N_k2)))
            log_file.write('Number of correspondence (NC): {:.2f}.\n'.format(N_corr.mean()))
            log_file.write('Number of correct matches (NCM): {:.2f}.\n'.format(np.mean(N_corretmatches)))
            log_file.write('Correct match rate (CMR): {:.2f}%.\n'.format(average_CMR))
            log_file.write('Repeatability rate (RR): {:.4f}.\n'.format(RR.mean()))
            log_file.write('Matching score (MS): {:.4f}.\n'.format(MS.mean()))
            log_file.write('=' * 50 + '\n\n')