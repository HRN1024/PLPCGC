import argparse

parser = argparse.ArgumentParser(
    prog='PLPCGC',
    description='Patch-wise Lidar Point Cloud Geometric Compression',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--data_path', help='* The path of SemanticKITTI', default='.../SemanticKiTTI/dataset/sequences')
parser.add_argument('--save_model_path', help=' ', default='./trained_model')
parser.add_argument('--train_seq', help=' ', default=('00', '01', '02', '03', '04', '05', '06', '07', '09', '10'))
parser.add_argument('--epoches', type=int, help=' ', default=20)
parser.add_argument('--patch_num', type=int, help=' ', default=2048)
parser.add_argument('--patch_point_num', type=int, help='Point num of each patch', default=62)
parser.add_argument('--d', type=int, help='Bottleneck size.', default=8)
parser.add_argument('--lamda', type=float, help='Lambda for rate-distortion tradeoff.', default=0.001)
parser.add_argument('--lr', type=float, help='Learning rate.', default=0.001)

parser.add_argument('--L', type=float, help='Quantization length', default=25)

parser.add_argument('--rate_loss_enable_step', type=int, help='Apply rate-distortion optimization at this steps.',
                    default=4000)
parser.add_argument('--NN', type=bool, help='Whether using nearest neighbor points as local center points.',
                    default=False)
parser.add_argument('--max_steps', type=int, help='Train up to this number of steps.',
                    default=40002)
# for compress.py and decompress.py
parser.add_argument('--test_data', help='The path of test data.',
                    default='.../SemanticKiTTI/dataset/sequences/08/velodyne_ply_mini/*.ply')
parser.add_argument('--trained_ae', help='The trained entropy autoencoder.', default='./trained_model/ae1000.pkl')
parser.add_argument('--trained_prob', help='The trained entropy model.', default='./trained_model/prob1000.pkl')
parser.add_argument('--compressed_path', help='Path to save the comressed .bin files.', default='.../compress_path')
parser.add_argument('--decompressed_path', help='Path to save the decompressed .ply files.',
                    default='.../decompress_path')
parser.add_argument('--device', help='AE Model Device (cpu or cuda)', default='cuda')

args = parser.parse_args()