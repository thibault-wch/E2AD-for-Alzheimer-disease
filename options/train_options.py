from options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # ---------------------------------------------------------
        # Core Training & Logging
        # ---------------------------------------------------------
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc.')
        parser.add_argument('--epoch_count', type=int, default=0, help='starting epoch count or total epochs')
        parser.add_argument('--accum_iter', type=int, default=2, help='gradient accumulation steps')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--print_freq', type=int, default=1,
                            help='frequency of showing training results on console')
        parser.add_argument('--save_epoch_freq', type=int, default=20,
                            help='frequency of saving checkpoints at the end of epochs')

        # ---------------------------------------------------------
        # Optimizer & Learning Rate Scheduling
        # ---------------------------------------------------------
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr_policy', type=str, default='exp',
                            help='learning rate policy: lambda|step|plateau|cosine|exp')
        parser.add_argument('--lr_decay', type=float, default=0.95, help='learning rate decay factor')
        parser.add_argument('--min_lr', type=float, default=0.0, metavar='LR',
                            help='lower lr bound for cyclic schedulers that hit 0')
        parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N', help='epochs to warmup LR')

        # ---------------------------------------------------------
        # Pretrained Checkpoints
        # ---------------------------------------------------------
        parser.add_argument('--pretrained_multi', type=str, default=None, help='model weights path for multi-modal')
        parser.add_argument('--pretrained_single', type=str, default=None, help='model weights path for single-modal')

        # ---------------------------------------------------------
        # Loss Configuration & Balancing Weights
        # ---------------------------------------------------------
        parser.add_argument('--use_atlas', action='store_true', help='use the atlases for training')
        # Changed from int to float to allow fractional tuning (e.g., 0.5)
        parser.add_argument('--lda_sh', type=float, default=1.0, help='loss weight for shared consensus')
        parser.add_argument('--lda_dis', type=float, default=1.0, help='loss weight for disentanglement')
        parser.add_argument('--lda_rec', type=float, default=1.0, help='loss weight for reconstruction')

        # Distillation specific weights
        parser.add_argument('--lda_soft', type=float, default=1.0, help='loss weight for soft label distillation')
        parser.add_argument('--lda_attn', type=float, default=1.0, help='loss weight for attention distillation')
        parser.add_argument('--lda_feat', type=float, default=1.0, help='loss weight for feature distillation')

        self.isTrain = True
        return parser