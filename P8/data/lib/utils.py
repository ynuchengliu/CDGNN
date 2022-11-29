# -*- coding:utf-8 -*-
# pylint: disable=no-member

import csv
import numpy as np
from scipy.sparse.linalg import eigs
from .metrics import mean_absolute_error, mean_squared_error, masked_mape_np
import torch


def search_data(sequence_length, num_of_batches, label_start_idx,
				num_for_predict, units, points_per_hour):
	'''
	Parameters
	----------
	sequence_length: int, length of all history data
	num_of_batches: int, the number of batches will be used for training
	label_start_idx: int, the first index of predicting target
	num_for_predict: int,
					 the number of points will be predicted for each sample
	units: int, week: 7 * 24, day: 24, recent(hour): 1
	points_per_hour: int, number of points per hour, depends on data
	Returns
	----------
	list[(start_idx, end_idx)]
	'''
	
	if points_per_hour < 0:
		raise ValueError("points_per_hour should be greater than 0!")
	
	if label_start_idx + num_for_predict > sequence_length:
		return None
	
	x_idx = []
	for i in range(1, num_of_batches + 1):
		start_idx = label_start_idx - points_per_hour * units * i
		end_idx = start_idx + num_for_predict
		if start_idx >= 0:
			x_idx.append((start_idx, end_idx))
		else:
			return None
	
	if len(x_idx) != num_of_batches:
		return None
	
	return x_idx[::-1]  # 倒叙输出,符合时间的 顺序输出,这里不占用多少空间


def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
					   label_start_idx, num_for_predict, points_per_hour=12):
	'''
	Parameters
	----------
	data_sequence: np.ndarray
				   shape is (sequence_length, num_of_vertices, num_of_features)
	num_of_weeks, num_of_days, num_of_hours: int
	label_start_idx: int, the first index of predicting target
	num_for_predict: int,
					 the number of points will be predicted for each sample
	points_per_hour: int, default 12, number of points per hour
	Returns
	----------
	week_sample: np.ndarray
				 shape is (num_of_weeks * points_per_hour,
						   num_of_vertices, num_of_features)
	day_sample: np.ndarray
				 shape is (num_of_days * points_per_hour,
						   num_of_vertices, num_of_features)
	hour_sample: np.ndarray
				 shape is (num_of_hours * points_per_hour,
						   num_of_vertices, num_of_features)
	target: np.ndarray
			shape is (num_for_predict, num_of_vertices, num_of_features)
	'''
	week_indices = search_data(data_sequence.shape[0], num_of_weeks,
							   label_start_idx, num_for_predict,
							   7 * 24, points_per_hour)
	if not week_indices:
		return None
	
	day_indices = search_data(data_sequence.shape[0], num_of_days,
							  label_start_idx, num_for_predict,
							  24, points_per_hour)
	if not day_indices:
		return None
	
	hour_indices = search_data(data_sequence.shape[0], num_of_hours,
							   label_start_idx, num_for_predict,
							   1, points_per_hour)
	if not hour_indices:
		return None
	
	week_sample = np.concatenate([data_sequence[i: j]
								  for i, j in week_indices], axis=0)
	day_sample = np.concatenate([data_sequence[i: j]
								 for i, j in day_indices], axis=0)
	hour_sample = np.concatenate([data_sequence[i: j]
								  for i, j in hour_indices], axis=0)
	target = data_sequence[label_start_idx: label_start_idx + num_for_predict]
	
	return week_sample, day_sample, hour_sample, target


def compute_val_loss(net, val_loader, loss_function, supports, device, epoch):
	net.eval()
	with torch.no_grad():
		temp = 1
		temp2 = 0  #
		tmp = []
		for index, (val_r, val_t) in enumerate(val_loader):
			if temp % 2 == 0:
				val_r = val_r.to(device)
				val_t = val_t.to(device)
				output, _, _ = net(val_r, supports[temp2])
				l = loss_function(output, val_t)
				tmp.append(l.item())
				temp2 += 1
			temp += 1
		validation_loss = sum(tmp) / len(tmp)
		print('\nEpoch: %s, validation loss: %.4f' % (epoch, validation_loss))
		return validation_loss


def predict(net, test_loader, supports, device):
	'''
	predict
	Parameters
	----------
	net: model
	test_loader: gluon.data.DataLoader
	Returns
	----------
	prediction: np.ndarray,
				shape is (num_of_samples, num_of_vertices, num_for_predict)
	'''
	net.eval()
	temp = 1
	temp2 = 0
	with torch.no_grad():
		prediction = []
		for index, (test_r, test_t) in enumerate(test_loader):
			if temp % 2 == 0:
				test_r = test_r.to(device)
				test_t = test_t.to(device)
				output, _, _ = net(test_r, supports[temp2])
				prediction.append(output.cpu().detach().numpy())
				temp2 += 1

			temp += 1

		prediction = np.array(prediction)
		return prediction#, spatial_at, temporal_at


def evaluate(net, test_loader, true_value, supports, device, epoch_):
	'''
	compute MAE, RMSE, MAPE scores of the prediction
	for 3, 6, 12 points on testing set
	Parameters
	----------
	net: model
	test_loader: gluon.data.DataLoader
	true_value: np.ndarray, all ground truth of testing set
				shape is (num_of_samples, num_for_predict, num_of_vertices)
	num_of_vertices: int, number of vertices
	epoch: int, current epoch
	'''
	net.eval()
	with torch.no_grad():
		prediction = predict(net, test_loader, supports, device)
		P8_ture = true_value[:2752]
		prediction = prediction.reshape(-1,170,12)
		temp_ture = P8_ture.reshape(-1, 16, 170, 12)
		ture = []
		for i in temp_ture[1::2]:
			ture.append(i)

		ture = np.array(ture)
		true_value = ture.reshape(-1,170,12)
		for i in range(1,13):
			print('\t current epoch: %s, predict %s points' % (epoch_, i), end='')
			
			mae = mean_absolute_error(true_value[:, :, 0:i], prediction[:, :, 0:i])
			rmse = mean_squared_error(true_value[:, :, 0:i], prediction[:, :, 0:i]) ** 0.5
			mape = masked_mape_np(true_value[:, :, 0:i], prediction[:, :, 0:i], 0)
			
			print('MAE: %.2f \t' % (mae), 'RMSE: %.2f\t' % (rmse), 'MAPE: %.2f' % (mape))

		_MAE = mean_absolute_error(true_value[:, :, 0:3], prediction[:, :, 0: 3])
		_RMSE = mean_squared_error(true_value[:, :, 0:3], prediction[:, :, 0:3]) ** 0.5
		_MAPE = masked_mape_np(true_value[:, :, 0:3], prediction[:, :, 0:3], 0)

	return _MAE, _RMSE, _MAPE
