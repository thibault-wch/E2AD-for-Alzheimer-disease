import argparse
import os


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # ---------------------------------------------------------
        # Core & Hardware Parameters
        # ---------------------------------------------------------
        parser.add_argument('--group', type=str, default='smci+pmci', help='group type/classification target')
        parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')

        parser.add_argument('--drop_ratio', type=float, default=0.0,
                            help='Probability to drop a cropped area if the label is empty. 0 = drop all, 1 = accept all')

        parser.add_argument('--gpu_ids', type=str, default='7',
                            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        # ---------------------------------------------------------
        # Environment & Directory Setup
        # ---------------------------------------------------------
        parser.add_argument('--name', type=str, default='experiment_name',
                            help='name of the experiment. Dictates where to store samples and models')
        parser.add_argument('--root', type=int, default=1,
                            help='Environment flag (1 for local data, other for remote/mnt)')
        parser.add_argument('--checkpoints_dir',type=str,required=True, help='directory used to save checkpoints and metrics')
        # ---------------------------------------------------------
        # Model & Architecture Parameters
        # ---------------------------------------------------------
        parser.add_argument('--class_num', type=int, default=4, help='number of output classes')
        parser.add_argument('--cls_type', type=str, default='resnet3d', help='backbone classification architecture')
        parser.add_argument('--lambda_init', type=float, default=0.6, help='lambda_init for the difference transformer')

        # Initialization
        parser.add_argument('--init_type', type=str, default='kaiming',
                            help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal')

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        opt, _ = parser.parse_known_args()
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = '\n' + '-' * 17 + ' Options ' + '-' * 17 + '\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = f'\t[default: {default}]'
            message += f'{str(k):>25}: {str(v):<30}{comment}\n'
        message += '-' * 17 + ' End ' + '-' * 19 + '\n'

        print(message)

        # Save options to disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(expr_dir, exist_ok=True)  # Replaced custom utils with native os.makedirs

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)

    def parse(self):
        opt = self.gather_options()

        # Ensure isTrain flag exists (defined in subclass like TrainOptions)
        opt.isTrain = getattr(self, 'isTrain', False)

        # ---------------------------------------------------------
        # Safe GPU ID Parsing (Critical for cudanet compatibility)
        # ---------------------------------------------------------
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            parsed_id = int(str_id.strip())
            if parsed_id >= 0:
                opt.gpu_ids.append(parsed_id)

        # Set environment variable only if valid GPUs are provided
        if opt.gpu_ids:
            print(f'Using visible logical GPUs: {opt.gpu_ids}')
        else:
            print('Running on CPU')

        # Print and save final structured options
        self.print_options(opt)

        self.opt = opt
        return self.opt