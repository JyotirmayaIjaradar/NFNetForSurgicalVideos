import datetime
import os
from shutil import copy2

def prepare_output_folders(opts):
	output_folder = "{}{}_horizon{}_{}/".format(opts.output_folder, datetime.datetime.now().strftime("%Y%m%d-%H%M"), opts.horizon, opts.trial_name)
	print('Output directory: ' + output_folder)
	result_folder = os.path.join(output_folder,'results')
	script_folder = os.path.join(output_folder,'scripts')
	model_folder = os.path.join(output_folder,'models')
	log_path = os.path.join(output_folder,'log.txt')

	os.makedirs(output_folder)
	os.makedirs(result_folder)
	os.makedirs(script_folder)
	os.makedirs(model_folder)

	for f in os.listdir():
		if '.py' in f:
			copy2(f,script_folder)

	return result_folder, model_folder, log_path