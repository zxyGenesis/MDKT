from torch.autograd import Variable
import torch.optim
import h5py
import pandas as pd
import argparse
import torch.utils.data.sampler
import os
import shutil
import random
from models import FSLModel
from utils import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 1118
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


class SimpleHDF5Dataset:
	def __init__(self, file_handle):
		self.f = file_handle
		self.all_feats_dset = self.f['all_feats']
		self.all_labels = self.f['all_labels']
		self.total = self.f['count']

	def __getitem__(self, i):
		return torch.Tensor(self.all_feats_dset[i,:]), int(self.all_labels[i])

	def __len__(self):
		return self.total


# a dataset to allow for category-uniform sampling of base and novel classes.
# also incorporates hallucination
class LowShotDataset:
	def __init__(self, base_feats, novel_feats, base_classes, novel_classes):
		self.f = base_feats
		self.all_base_feats_dset = self.f['all_feats'][...]
		self.all_base_labels_dset = self.f['all_labels'][...]

		self.novel_feats = novel_feats['all_feats']
		self.novel_labels = novel_feats['all_labels']

		self.base_classes = base_classes
		self.novel_classes = novel_classes

		self.frac = 0.5
		self.all_classes = np.concatenate((base_classes, novel_classes))

	def sample_base_class_examples(self, num):
		sampled_idx = np.sort(np.random.choice(len(self.all_base_labels_dset), num, replace=False))
		return torch.Tensor(self.all_base_feats_dset[sampled_idx,:]), torch.LongTensor(self.all_base_labels_dset[sampled_idx].astype(int))

	def sample_novel_class_examples(self, num):
		sampled_idx = np.random.choice(len(novel_feats['all_labels']), num)
		return torch.Tensor(self.novel_feats[sampled_idx,:]), torch.LongTensor(self.novel_labels[sampled_idx].astype(int))

	def get_sample(self, batchsize):
		num_base = round(self.frac*batchsize)
		num_novel = batchsize - num_base
		base_feats, base_labels = self.sample_base_class_examples(int(num_base))
		novel_feats, novel_labels = self.sample_novel_class_examples(int(num_novel))
		return torch.cat((base_feats, novel_feats)), torch.cat((base_labels, novel_labels))

	def featdim(self):
		return self.novel_feats.shape[1]


# simple data loader for test
def get_test_loader(file_handle, batch_size=1000):
	testset = SimpleHDF5Dataset(file_handle)
	data_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
	return data_loader


def training_loop(lowshot_dataset,novel_test_feats, params, batchsize=1000, maxiters=1000, reload=False):
	if os.path.exists('FSLModel/tmp/' + str(params.nshot)) == False:
		os.makedirs('FSLModel/tmp/' + str(params.nshot))

	model = FSLModel(reload).cuda()
	if params.nshot == 1:
		model.a = 2
	test_loader = get_test_loader(novel_test_feats)
	loss_function = nn.CrossEntropyLoss().cuda()

	best_ACC = 0.0
	tmp_epoach = 0
	tmp_count = 0
	tmp_rate = params.lr
	max_tmp_count = 8
	optimizer = torch.optim.Adam(model.parameters(), tmp_rate, weight_decay=params.wd)
	time = getTime()
	for epoch in range(maxiters):
		
		(x,y) = lowshot_dataset.get_sample(batchsize)
		optimizer.zero_grad()

		x = Variable(x.cuda())
		y = Variable(y.cuda())

		scores_v, scores_t, scores_all, t_final, v_final = model(x)

		loss = 0.0
		loss += 100 * nn.MSELoss().cuda()(t_final, v_final)

		loss += loss_function(scores_v, y)
		loss += loss_function(scores_t, y)
		loss += loss_function(scores_all, y)

		loss.backward()
		optimizer.step()

		if (epoch%10==0):
			accs = eval_loop(test_loader, model, novel_classes)
			tmp_count += 1
			if  accs[1] > best_ACC :
				print(accs[1])
				best_ACC = accs[1]
				save_file_path = 'FSLModel/tmp/' + str(params.nshot) +'/' + time + str(params.nshot) +'_save_' + str(epoch)+ '_' + str(round(best_ACC, 4)) + '.pth'
				states = {
					'state_dict': model.state_dict(),
					'test_w': model.test_w,
				}
				torch.save(states, save_file_path)
				print('save: ' , epoch)

	return model

def perelement_accuracy(scores, labels):
	topk_scores, topk_labels = scores.topk(5, 1, True, True)
	label_ind = labels.cpu().numpy()
	topk_ind = topk_labels.cpu().numpy()
	top1_correct = topk_ind[:,0] == label_ind
	top5_correct = np.sum(topk_ind == label_ind.reshape((-1,1)), axis=1)
	return top1_correct.astype(float), top5_correct.astype(float)


def eval_loop(data_loader, model, novel_classes):
	model = model.eval()
	top1 = None
	top5 = None
	no_novel_class = list(set(range(1360)).difference(set(novel_classes)))
	all_labels = None
	for i, (x,y) in enumerate(data_loader):
		x = Variable(x.cuda())
		scores = model.test(x)

		pre_scores = 0.0
		pre_scores += scores.softmax(-1)

		pre_scores[:,no_novel_class] = -0.0
		top1_this, top5_this = perelement_accuracy(pre_scores.data, y)
		top1 = top1_this if top1 is None else np.concatenate((top1, top1_this))
		top5 = top5_this if top5 is None else np.concatenate((top5, top5_this))
		all_labels = y.numpy() if all_labels is None else np.concatenate((all_labels, y.numpy()))

	is_novel = np.in1d(all_labels, novel_classes)
	top1_novel = np.mean(top1[is_novel])
	top5_novel = np.mean(top5[is_novel])
	return np.array([top1_novel, top5_novel])

def parse_args():
	parser = argparse.ArgumentParser(description='Low shot benchmark')
	parser.add_argument('--numclasses', default=1360, type=int)
	parser.add_argument('--lr', default=0.001, type=float)
	parser.add_argument('--wd', default=0.001, type=float)
	parser.add_argument('--maxiters', default=1000000, type=int)
	parser.add_argument('--batchsize', default=1000, type=int)
	parser.add_argument('--nshot', default=5, type=int)
	parser.add_argument('--nsplit', default=1, type=int)

	return parser.parse_args()


if __name__ == '__main__':

	params = parse_args()

	with open('ExperimentSplit/Json/train.json','r') as f:
		exp = json.load(f)
		base_classes = list(set(exp['image_labels']))

	with open('ExperimentSplit/Json/test.json','r') as f:
		exp = json.load(f)
		novel_classes = list(set(exp['image_labels']))

	train_feats = h5py.File('Features/train.hdf5', 'r')
	novel_feats = h5py.File('Features/test.hdf5', 'r')
	novel_feats_train_feats = novel_feats['all_feats'][...]
	novel_feats_train_labels = novel_feats['all_labels'][...]

	all_novel_train_id = pd.read_csv('ExperimentSplit/Split_Idx/novel_trainid.csv', sep = ' ', header = None)
	K = params.nshot
	novel_train_id = None
	for i in range(K):
		novel_train_id = np.concatenate((novel_train_id,all_novel_train_id[i].values),0) if novel_train_id is not None else all_novel_train_id[i].values

	novel_test_id = []
	with open('ExperimentSplit/Split_Idx/novel_testid.csv') as f :
		for idx in f.readlines()[0].split(' '):
			novel_test_id.append(int(idx))
	novel_test_id = np.array(novel_test_id)

	novel_train_feats = novel_feats_train_feats[novel_train_id]
	novel_train_labels = novel_feats_train_labels[novel_train_id]
	novel_test_feats = novel_feats_train_feats[novel_test_id]
	novel_test_labels = novel_feats_train_labels[novel_test_id]


	novel_feats = {}
	novel_feats['all_feats'] = novel_train_feats
	novel_feats['all_labels'] = novel_train_labels
	novel_feats['count'] = len(novel_train_labels)

	novel_val_feats = {}
	novel_val_feats['all_feats'] = novel_test_feats
	novel_val_feats['all_labels'] = novel_test_labels
	novel_val_feats['count'] = len(novel_test_labels)


	lowshot_dataset = LowShotDataset(train_feats, novel_feats, base_classes, novel_classes)
	ori_graph = graph_generate(np.load("imagenet1360wordvec.npy"), 5)
	np.save("ori_graph", ori_graph)
	model = training_loop(lowshot_dataset,novel_val_feats, params, params.batchsize, params.maxiters)
	print("-------retraining----------")
	model = training_loop(lowshot_dataset, novel_val_feats, params, params.batchsize,params.maxiters, reload=True)

	print('trained')
