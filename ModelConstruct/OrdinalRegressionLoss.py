import torch
import torch.nn as nn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class OrdinalRegressionLoss(nn.Module):

    def __init__(self, num_class, train_cutpoints=False, scale=20.0):
        super().__init__()
        self.num_classes = num_class
        num_cutpoints = self.num_classes - 1
        self.cutpoints = torch.arange(num_cutpoints).float() * scale / (num_class - 2) - scale / 2
        self.cutpoints = nn.Parameter(self.cutpoints)
        if not train_cutpoints:
            self.cutpoints.requires_grad_(False)

    def forward(self, pred, label):
        pred = pred.unsqueeze(-1)
        sigmoids = torch.sigmoid(self.cutpoints.to(device) - pred)
        link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
        link_mat = torch.cat((
            sigmoids[:, [0]],
            link_mat,
            (1 - sigmoids[:, [-1]])
        ),
            dim=1
        )

        eps = 1e-15
        likelihoods = torch.clamp(link_mat, eps, 1 - eps)

        neg_log_likelihood = torch.log(likelihoods)
        if label is None:
            loss = 0
        else:
            label = label.unsqueeze(-1)
            loss = -torch.gather(neg_log_likelihood, 1, label.to(torch.long)).mean()

        predicted_classes = torch.argmax(link_mat, dim=1)

        return loss, likelihoods, predicted_classes


''' 使用样例
ord_loss = OrdinalRegressionLoss(5)
pred = torch.rand((5, 1))
label = torch.LongTensor([0, 1, 2, 3, 4]).unsqueeze(-1)
loss, likelihoods = ord_loss(pred, label)
print(loss)'''
