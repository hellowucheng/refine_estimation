import torch
import torch.nn as nn
from models.base import Base


class JointsFocalLoss(Base):

    def __init__(self, beta=4, alpha=2):
        super(JointsFocalLoss, self).__init__()

        self.beta, self.alpha = beta, alpha

    def forward(self, preds, target, target_weight, joints_vis):
        # N x C x H x W, 正类像素点掩码
        pos_inds = target.ge(0.8).float()
        # N x C x H x W, 负类像素点掩码
        neg_inds = target.lt(0.8).float()

        neg_weights = torch.pow(1 - target, self.beta)

        loss = 0
        for i, pred in enumerate(preds):
            pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)

            # 0.27620
            # 在正负关节点个数差异方面做均衡
            heatmap_weight = (target_weight[i] + 0.27620 * (1 - target_weight[i]))

            # 交叉熵均衡难易预测
            pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds[i]
            neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights[i] * neg_inds[i]

            # 带遮挡点的热图和可见点的热图之间的均衡
            # assert heatmap_weight.dim() == 1
            pos_loss = pos_loss * heatmap_weight.view(heatmap_weight.size(0), 1, 1)
            neg_loss = neg_loss * heatmap_weight.view(heatmap_weight.size(0), 1, 1)

            # 有多少个正类像素点
            num_pos = pos_inds[i].float().sum()
            # neg_inds是除遮挡点位置外的所有像素点
            # 正负像素点之间的均衡
            pos_loss = pos_loss.sum() / pos_inds[i].sum()
            neg_loss = neg_loss.sum() / (neg_inds[i] * heatmap_weight.view(heatmap_weight.size(0), 1, 1)).sum()

            # 全是负类热图, 即这是一个没有遮挡的实例
            if num_pos == 0:
                loss = loss - neg_loss
            else:
                loss = loss - (pos_loss + neg_loss)
        return loss / len(preds)


class JointsMSELoss(Base):

    def __init__(self, use_target_weight=True):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight, joints_vis=None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()

            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx].unsqueeze(1)),
                    heatmap_gt.mul(target_weight[:, idx].unsqueeze(1))
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointsMSEBalancedLoss(Base):

    def __init__(self, alpha=0.005):
        super(JointsMSEBalancedLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.alpha = alpha

    def forward(self, output, target, target_weight, joints_vis):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            vis_filter = (joints_vis[:, idx] == 2).int()
            re_weight = (vis_filter + self.alpha * (1 - vis_filter))
            loss += 0.5 * self.criterion(
                heatmap_pred.mul(re_weight.unsqueeze(1)),
                heatmap_gt.mul(re_weight.unsqueeze(1))
            )

        return loss / num_joints
