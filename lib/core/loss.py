import torch.nn as nn
from models.base import Base

import matplotlib.pyplot as plt


class JointsMSELoss(Base):

    def __init__(self, use_target_weight=True):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()

            # plt.imshow(heatmap_pred[0, :].reshape(64, 64).cpu().detach().numpy())
            # plt.show()
            # print('')
            # plt.imshow(heatmap_gt[0, :].reshape(64, 64).cpu().detach().numpy())
            # plt.show()
            # a = heatmap_pred.mul(target_weight[:, idx].unsqueeze(1))
            # b = heatmap_gt.mul(target_weight[:, idx].unsqueeze(1))

            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx].unsqueeze(1)),
                    heatmap_gt.mul(target_weight[:, idx].unsqueeze(1))
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints
