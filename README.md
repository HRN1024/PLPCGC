# Patch-wise LiDAR Point Cloud Geometry Compression based on Autoencoder

This is the implementation of the paper "Patch-wise LiDAR Point Cloud Geometry Compression based on Autoencoder"(ICIG 2023).

## Train
`$ python train.py `

Set the following argument in args_file.py:

`parser.add_argument('--data_path', help='Path to SemanticKITTI dataset', default='.../SemanticKiTTI/dataset/sequences')
`
*  Adjust the following arguments to achieve a different compression ratio

`parser.add_argument('--patch_num', type=int, help=' ', default=2048)`

`parser.add_argument('--patch_point_num', type=int, help='point num of each patch', default=62)`

`parser.add_argument('--d', type=int, help='Bottleneck size.', default=8)`

`parser.add_argument('--lamda', type=float, help='Lambda for rate-distortion tradeoff.', default=0.001)`

## Compress
`$ python compress.py `

Set the following arguments in args_file.py:


`parser.add_argument('--test_data', help='Point clouds glob pattern for compression.',
                    default='.../SemanticKiTTI/dataset/sequences/08/velodyne_ply_mini/*.ply')
`

`parser.add_argument('--trained_ae', help='Path to the trained entropy autoencoder.', default='./trained_model/ae40000.pkl')
`

`parser.add_argument('--trained_prob', help='Path to the trained entropy model.', default='./trained_model/prob40000.pkl')
`

`parser.add_argument('--compressed_path', help='Path to save the compressed .bin files.', default='.../compress_path')
`

## Decompress

`$ python decompress.py `


Set the following argument in args_file.py:


`parser.add_argument('--decompressed_path', help='Path to save the decompressed .ply files.',
                    default='.../decompress_path')`

## Acknowledgment
The code is built based on the repositories of [PCC_Patch](https://github.com/I2-Multimedia-Lab/PCC_Patch) and  [IPDAE](https://github.com/I2-Multimedia-Lab/IPDAE). 
Thanks for their great work.


