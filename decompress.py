import os
import numpy as np
import torch
import torchac
from tqdm import tqdm
from glob import glob
import utils
import AE as AE
from args_file import args


if not os.path.exists(args.decompressed_path):
    os.makedirs(args.decompressed_path)

files = glob(os.path.join(args.compressed_path, '*.p.bin'))
filenames = np.array([os.path.split(x)[1][:-6] for x in files])


ae = AE.get_model(args=args).to(args.device)
ae.load_state_dict(torch.load(args.trained_ae))
ae.eval()
prob = AE.ConditionalProbabilityModel(args.L, args.d).cuda()
prob.load_state_dict(torch.load(args.trained_prob))
prob.eval()

for i in tqdm(range(len(filenames))):
    latent_code_path = os.path.join(args.compressed_path, filenames[i] + '.p.bin')
    patch_num = np.fromfile(os.path.join(args.compressed_path, filenames[i] + '.h.bin'),
                            dtype=np.uint16)[0]
    sample_num = patch_num
    sample_point = np.fromfile(os.path.join(args.compressed_path, filenames[i] + '.xyz.bin'), dtype=np.float16).reshape(
        -1, 3)
    sample_point = torch.tensor(sample_point, dtype=torch.float32).to(args.device)

    # Estimate probability distribution and performing entropy decoding
    pmf = prob(sample_point.unsqueeze(0).float().cuda())
    with open(latent_code_path, 'rb') as fin:
        byte_stream = fin.read()
    cdf = utils.pmf_to_cdf(pmf).cpu()

    latent = ((torchac.decode_float_cdf(cdf, byte_stream) - ae.L // 2).float()).view(patch_num, -1)

    quantizated_feature = latent.float().to(args.device)

    # predict local coordinate
    patches_pred = ae.decoder(quantizated_feature)

    # calculate global coordinate
    patches_pred = patches_pred + sample_point.unsqueeze(1)
    pc_pred = patches_pred.reshape(1, -1, 3)
    pc_save = pc_pred.reshape(-1, 3).detach().cpu().numpy()

    utils.save_point_cloud(pc_save, filenames[i] + '.bin.ply', path=args.decompressed_path)
