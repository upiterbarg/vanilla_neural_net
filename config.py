import argparse

# initialize parser
parser = argparse.ArgumentParser()


# enforce determinism
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--rng', type=int, default=-1, help='JAX random key --> reset with random seed during config')

# model 
parser.add_argument('--inp_dim', type=int, default=256, help='dimension of input')
parser.add_argument('--hidden_dim', type=int, default=1024, help='hidden dim')
parser.add_argument('--out_dim', type=int, default=16, help='dimension of output')
parser.add_argument('--nlayers', type=int, default=4, help='number of layers in MLP')

# optimization and training
parser.add_argument('--lr', type=float, default=0.003, help='learning rate during training')
parser.add_argument('--b1', type=float, default=0.9, help='set b1 for adam optimization')
parser.add_argument('--b2', type=float, default=0.999, help='set b2 for adam optimization')
parser.add_argument('--eps', type=float, default=1e-8, help='set eps for adam optimization')
parser.add_argument('--loss', type=str, default='mse')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--n_epochs', type=int, default=10)


# evaluation
parser.add_argument('--eval', type=int, default=0, help='evaluate model only?')

# logging
parser.add_argument('--debug', type=int, default=1, help='if True, logs are NOT saved')
parser.add_argument('--outf', type=str, default=osp.join(os.getcwd(), 'dump'), help='log parent directory')
parser.add_argument('--log_per_iter', type=int, default=200, help='logging frequency during training')



def gen_args():
	args = parser.parse_args()
	args.modeloutf = os.path.join(args.outf, 'trained_model')
	return args