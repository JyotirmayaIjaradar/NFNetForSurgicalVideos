import torch
from dataloader import prepare_dataset
from model import AnticipationModel
from options import parser
import os

opts = parser.parse_args()

if not os.path.exists(opts.test_folder):
	os.mkdir(opts.test_folder)

model_file = os.path.join(opts.model_folder, 'model{:3d}.pkl'.format(opts.model_epoch))

_, test_set = prepare_dataset(opts)
model = AnticipationModel(opts,train=False,pretrain=model_file)

with torch.no_grad():

	model.net.eval()
	model.net.set_mode('VARIATIONAL')
					
	for ID,op in test_set:
		
		samples = model.sample_op_predictions(op,opts.num_samples)
		model.save_samples(samples,ID=ID,epoch=opts.model_epoch,result_folder=opts.test_folder)
		print('saved samples for OP #' + ID)