import cv2
import numpy as np


# 获取旋转后的坐标
def get_dir(src, rot_rad):
    sin, cos = np.sin(rot_rad), np.cos(rot_rad)
    return [src[0] * cos - src[1] * sin, src[0] * sin + src[1] * cos]


# 获取仿射变换的第三个点
def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


# 获取仿射变换
def get_affine_transform(center, bw, bh, rot, dst_w, dst_h, is_train=True):
    # 三个点构建仿射变换实现 旋转、裁剪
    src, dst = np.zeros((3, 2), dtype=np.float32), np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + get_dir([0, bh * -0.5], np.pi * rot / 180)
    src[2, :] = center + get_dir([bw * -0.5, bh * -0.5], np.pi * rot / 180)
    # center仿射变换后 即 新图片的中心 [w / 2, h / 2]
    dst[0, :] = np.array([dst_w * 0.5, dst_h * 0.5])
    dst[1, :] = np.array([dst_w * 0.5, 0])
    dst[2, :] = np.array([0, 0])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    if not is_train:
        trans_v = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans_v = np.zeros((2, 3))
    return trans, trans_v


# 获取欧式变换(旋转 + 平移)
def get_similarity_transform(center, rot, tx, ty):
    trans = cv2.getRotationMatrix2D(center, rot, 1)
    trans[0, 2] += tx
    trans[1, 2] += ty
    return trans


# 应用仿射变换
def affine_transform(joints, t):
    new_joints = np.array([joints[0], joints[1], 1.]).T
    new_joints = np.dot(t, new_joints)
    return new_joints[:2]


# 复合变换
def mul_transform(t1, t2):
    t1 = np.concatenate([t1, np.array([[0, 0, 1]])], axis=0)
    t2 = np.concatenate([t2, np.array([[0, 0, 1]])], axis=0)
    return np.dot(t1, t2)[:2, :]


# 对图像和关节点同时应用变换
def perform_transform(img, joints, trans, dst_w, dst_h):
    # 应用变换
    input = cv2.warpAffine(img, trans, (int(dst_w), int(dst_h)), flags=cv2.INTER_LINEAR)
    # 坐标真值随仿射变换修改
    for i in range(joints.shape[0]):
        joints[i] = affine_transform(joints[i], trans)
    return input, joints


# 生成高斯概率热图
def generate_target(joints, joints_vis, sigma, image_size, heatmap_size):
    num_joints = joints.shape[0]
    target_weight = joints_vis.copy()

    target = np.zeros((num_joints,
                       heatmap_size,
                       heatmap_size),
                      dtype=np.float32)

    for joint_id in range(num_joints):
        feat_stride = image_size / heatmap_size
        mu_x = int(joints[joint_id][0] / feat_stride + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride + 0.5)
        # 关节点的3 * sigma热点区域必须与图像有交集, 否则视为不可见
        ul = [int(mu_x - 3 * sigma), int(mu_y - 3 * sigma)]
        br = [int(mu_x + 3 * sigma + 1), int(mu_y + 3 * sigma + 1)]
        if ul[0] >= heatmap_size or ul[1] >= heatmap_size \
                or br[0] < 0 or br[1] < 0:
            target_weight[joint_id] = 0
            continue

        # 生成高斯概率
        size = 6 * sigma + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size)
        img_y = max(0, ul[1]), min(br[1], heatmap_size)

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target, target_weight


# 水平翻转
def flip_joints(joints, joints_vis, img_width, dataset='mpii'):
    flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
    if dataset.lower() == 'mpii':
        flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
    joints[:, 0] = img_width - joints[:, 0] - 1

    for a, b in flip_pairs:
        joints[a, :], joints[b, :] = joints[b, :], joints[a, :].copy()
        joints_vis[a], joints_vis[b] = joints_vis[b], joints_vis[a]

    return joints, joints_vis


# 用于flip-test
def flip_back(output, dataset='mpii'):
    flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
    if dataset.lower() == 'mpii':
        flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]

    output = output[:, :, :, ::-1]
    for a, b in flip_pairs:
        output[:, a, :, :], output[:, b, :, :] = output[:, b, :, :], output[:, a, :, :].copy()
    return output
