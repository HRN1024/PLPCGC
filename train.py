import numpy as np
import torch
import torch.utils.data as Data
from utils import knn_points, estimate_bits_from_pmf
import AE as AE
from dataset import CompressDataset
import os
import itertools
from args_file import args

if __name__ == '__main__':
    dataset = CompressDataset(args.data_path, seq=args.train_seq, patch_num=args.patch_num,
                              patch_point_num=args.patch_point_num)
    loader = Data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)

    ae = AE.get_model(args=args).to(args.device).train()
    criterion = AE.get_loss().to(args.device)
    prob = AE.ConditionalProbabilityModel(args.L, args.d).to(args.device).train()
    optimizer = torch.optim.Adam(itertools.chain(ae.parameters(), prob.parameters()), lr=args.lr)
    global_step = 0
    losses = []
    bpps = []
    for epoch in range(args.epoches):
        for step, (batch_x, total_point_num, sample_point) in enumerate(loader):
            print('epoch：{}，step：{}'.format(epoch, global_step))
            if torch.sum(total_point_num) < 110000:
                print('point num too less.')
                continue
            B = np.shape(batch_x)[0]
            P = np.shape(batch_x)[1]
            target = batch_x
            batch_x = batch_x.float()
            sample_point = sample_point.float()
            if args.NN:
                dists, idx, sample_point = knn_points(sample_point, batch_x.view(1, -1, 3), K=1,
                                                      return_nn=True)
            else:
                sample_point = sample_point.unsqueeze(2)

            batch_x = batch_x.to(args.device)
            target = target.to(args.device).float()
            total_point_num = total_point_num.to(args.device)
            sample_point = sample_point.to(args.device)
            sample_point = sample_point.half()
            sampled_bits = sample_point.shape[0] * sample_point.shape[1] * 16 * 3

            # Local coordinate
            batch_x = batch_x - sample_point.float()
            batch_x = batch_x.view(B * P, np.shape(batch_x)[2], np.shape(batch_x)[3])

            new_xyz, latent_quantized = ae(batch_x.float())
            # global coordinate
            new_xyz = new_xyz + sample_point.squeeze(0).view(-1, 1, 3)

            pmf = prob(sample_point.reshape(1, -1, 3).float())
            feature_bits = estimate_bits_from_pmf(pmf=pmf,
                                                  sym=(latent_quantized.view(B, args.patch_num,
                                                                             args.d) + args.L // 2).long())
            bpp = (sampled_bits + feature_bits.detach().cpu().numpy()) / torch.sum(total_point_num)
            bpp_feature = feature_bits / torch.sum(total_point_num)

            pc_pred = new_xyz.reshape(B, -1, 3)
            pc_target = target.reshape(B, -1, 3)

            if global_step < args.rate_loss_enable_step:  # Optimize distortion only
                loss = criterion(pc_pred, pc_target, bpp_feature, lambda_=0)
            else:  # Optimize reta-distortion
                loss = criterion(pc_pred, pc_target, bpp_feature, lambda_=args.lamda)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            losses.append(loss.item())
            bpps.append(bpp.item())

            if global_step % 2000 == 0:
                print(
                    f'Epoch:{epoch} | Step:{global_step} | bpp:{round(np.array(bpps).mean(), 5)} | Loss:{round(np.array(losses).mean(), 5)}')
                torch.save(ae.state_dict(), os.path.join(args.save_model_path, 'ae{}.pkl'.format(global_step)))
                torch.save(prob.state_dict(), os.path.join(args.save_model_path, 'prob{}.pkl'.format(global_step)))

            if global_step > args.max_steps:
                break
        if global_step > args.max_steps:
            break
