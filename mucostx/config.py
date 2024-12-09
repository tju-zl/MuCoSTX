import argparse

def config():
    parser = argparse.ArgumentParser('Configuration File of MuCoSTX')
    
    # system configuration
    parser.add_argument('--project_name', type=str, default='MoCoSTX')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--version', type=str, default='dev-1.0', help='if dev on, warnings will appear.')

    # data preparation
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--srt_resolution', default=150, type=int)
    parser.add_argument('--clusters', default=0, type=int)
    parser.add_argument('--max_neighbors', default=6, type=int)
    parser.add_argument('--n_spot', default=0, type=int, help='update when read data.')
    parser.add_argument('--hvgs', default=3000, type=int)
    parser.add_argument('--mode_rknn', type=str, default='rknn')

    # model parameters
    parser.add_argument('--latent_dim', type=int, default=64, help='dim of latent space')
    parser.add_argument('--flow', type=str, default='source_to_target')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--elastic_corr', default=False, action='store_true')

    # training control
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--wegiht_decay', type=float, default=1e-4)
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--tau', type=float, default=0.05)
    parser.add_argument('--amp', default=True, action='store_true', help='Mixed precision')
    
    # output configuration
    parser.add_argument('--log_file', type=str, default='../Log')
    parser.add_argument('--out_file', type=str, default='../Output')

    # analysis configuration
    parser.add_argument('--visual', default=True, action='store_true')

    return parser
