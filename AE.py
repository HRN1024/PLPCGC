import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import PointNet, SAPP
from pytorch3d.loss import chamfer_distance


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.sa = SAPP(in_channel=3, feature_region=8, mlp=[64, 64, 128, 128], norm=False, sample_rate=1, res=False)
        self.pn = PointNet(in_channel=3 + 128, mlp=[128, 128, 64, 64], norm=False, res=False)
        self.output_layer = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, args.d),
        )

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2)
        _, feature = self.sa(xyz)

        feature = self.pn(torch.cat((xyz, feature), dim=1))
        feature = self.output_layer(feature)
        return feature


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.d = args.d
        self.patch_point_num = args.patch_point_num
        self.MLP_layers = nn.Sequential(
            nn.Linear(self.d, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.patch_point_num * 32),
            nn.ReLU()
        )
        self.local_coord_predict = nn.Sequential(
            nn.Conv2d(self.d + 32, 128, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1)
        )

    def forward(self, latent_quantized_trans):
        mlp_output = self.MLP_layers(latent_quantized_trans)
        mlp_output = mlp_output.view(latent_quantized_trans.shape[0], -1, self.patch_point_num)  
        latent_quantized = latent_quantized_trans.unsqueeze(-1).repeat((1, 1, self.patch_point_num))
        smlp_input = torch.cat((mlp_output, latent_quantized), dim=1)
        smlp_input = smlp_input.unsqueeze(-1)  # [B, C, N, 1]
        new_xyz = self.local_coord_predict(smlp_input)
        new_xyz = new_xyz.squeeze(-1)  # [B, D, N]
        new_xyz = new_xyz.transpose(2, 1)
        return new_xyz


class STEQuantize(torch.autograd.Function):
    """Straight-Through Estimator for Quantization.
    Forward pass implements quantization by rounding to integers,
    backward pass is set to gradients of the identity function.
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.round()

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs


class ConditionalProbabilityModel(nn.Module):
    def __init__(self, L, d):
        super(ConditionalProbabilityModel, self).__init__()
        self.L = L
        self.d = d
        self.model_pn = PointNet(in_channel=3, mlp=[64, 64, 128, 128], norm=False, res=False)
        self.model_mlp = nn.Sequential(
            nn.Conv2d(3 + 128, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, d * self.L, 1),
        )

    def forward(self, sampled_xyz):
        B, S, C = sampled_xyz.shape
        feature = self.model_pn(sampled_xyz.transpose(1, 2))
        mlp_input = torch.cat((sampled_xyz, feature.repeat((1, S)).view(B, S, -1)), dim=2)
        mlp_input = mlp_input.unsqueeze(-1).transpose(1, 2)
        output = self.model_mlp(mlp_input)
        output = output.transpose(1, 2).view(B, S, self.d, self.L)
        pmf = F.softmax(output, dim=3)
        return pmf


class get_model(nn.Module):
    def __init__(self, args=None):
        super(get_model, self).__init__()
        self.patch_num = args.patch_num
        self.patch_point_num = args.patch_point_num
        self.d = args.d
        self.L = args.L
        self.quantize = STEQuantize.apply
        self.encoder = Encoder(args=args)
        self.decoder = Decoder(args=args)

    def forward(self, xyz):
        feature = self.encoder(xyz)
        spread = self.L - 0.2
        latent = torch.sigmoid(feature) * spread - spread / 2  # (-L/2,L/2)
        quantizated_feature = self.quantize(latent)
        new_xyz = self.decoder(quantizated_feature)
        return new_xyz, quantizated_feature


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.criterion_mse = torch.nn.MSELoss()

    def forward(self, pc_pred, pc_target, bpp, lambda_):
        d, _ = chamfer_distance(pc_pred, pc_target)
        loss_pc = d + lambda_ * bpp
        return loss_pc
