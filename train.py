import haiku as hk
import jax
import jax.numpy as np
import optax

from config import *
from data import MyDataset
from haiku.nets import MLP
from torch.utils.data import DataLoader

# load arguments
args = gen_args()


# if not in 'debug' mode, setup output directory and logger.
if not args.debug:
	os.system('mkdir -p ' + args.outf)
	og = Log(osp.join(args.outf, 'train.log'), 'w')


# load training data
phases = ['train', 'valid'] if args.eval == 0 else ['valid']
datasets = {phase: MyDataset(phase) for phase in phases}
dataloaders = {phase: DataLoader(datasets[phase], batch_size=args.batch_size, shuffle=True if phase == 'train' else False, collate_fn=my_collate, num_workers=args.num_workers) for phase in phases}


# setup + initialize model
def model_fn(x):
	return MLP([args.inp_dim] + [args.hidden_dim] * (args.nlayers - 1) + [args.out_dim])(x)

model = hk.transform(model_fn)
model_params = model.init(args.rng, datasets['train'].__getitem__(0))


# setup training updates
opt = optax.adam(args.lr)
opt_state = opt.init(model_params)


# set up loss function
def mse(params, x, y):
	return jnp.mean((model.apply(params, x) - y) ** 2)


# initialize optimizer
opt_init, opt_update = optax.chain(optax.scale_by_adam(b1=args.b1, b2=args.b2, eps=args.eps), optax.scale(-args.lr))


# log args
print('-'*50)
print(args)
print('-'*50)

# start training
for epoch in range(args.n_epochs):
	for phase in phases:
    	for i, data in enumerate(dataloaders[phase]):
    	# unpack data
    	inputs, labels = data

		# compute gradient and loss.
		loss, grad = jax.value_and_grad(mse)(model_params, xs, bottom_flux, top_flux, ys)

		if phase == 'train':
			# transform the gradients using the optimiser.
			updates, opt_state = opt_update(grad, opt_state, model_params)

			# update parameters.
			model_params = optax.apply_updates(model_params, updates)

		if i % args.log_per_iter == 0:
			print('%s epoch[%d/%d] iter[%d/%d] loss: %.12f' % (phase, epoch, args.n_epochs, i, len(dataloaders[phase]), loss))


print('-'*50)
jnp.save(args.modeloutf, model_params, allow_pickle=True)
print('Model parameters were saved at the following path: '+args.modeloutf)