import os
import shutil
from time import time
from datetime import datetime
import argparse
import dgl
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from P8.model.predict_A import get_ture_A_and_last_A, predict_A
from P8.model.hawkes import get_u_begin, get_event_list, get_event_train_and_event_ture, discriminator, generator
from data.lib.utils import compute_val_loss, evaluate, predict
from data.lib.preprocess import read_and_generate_dataset
from model.evolution_KL_ST import setup_features_tuple, setup_Adj_matrix
from model.core import ActivateGraphSahe
from model.optimize import Lookahead
np.seterr(divide='ignore', invalid='ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:1', help='')
parser.add_argument('--max_epoch', type=int, default=40, help='Epoch to run [default: 40]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Initial learning rate [default: 0.0005]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adamW', help='adam or momentum [default: adam]')
parser.add_argument('--length', type=int, default=24, help='Size of temporal : 12')
parser.add_argument("--force", type=str, default=True, help="remove params dir")
parser.add_argument("--data_name", type=str, default=8, help="the number of data documents [8/4]", required=False)
parser.add_argument('--num_point', type=int, default=170, help='road Point Number [170/307] ', required=False)
parser.add_argument('--seed', type=int, default=31240, help='', required=False)
parser.add_argument('--decay', type=float, default=0.99, help='decay rate of learning rate [0.97/0.92]')
FLAGS = parser.parse_args()
decay = FLAGS.decay
dataname = FLAGS.data_name
graph_signal_matrix_filename = 'data/PEMS0%s/pems0%s.npz' % (dataname, dataname)
Length = FLAGS.length
num_nodes = FLAGS.num_point
epochs = FLAGS.max_epoch
optimizer = FLAGS.optimizer
num_of_vertices = FLAGS.num_point
seed = FLAGS.seed
num_of_features = 3
points_per_hour = 12
num_for_predict = 12
num_of_weeks = 2
num_of_days = 1
num_of_hours = 2
merge = False
model_name = 'CDGNN_cnt_params_%s' % dataname
params_dir = 'CDGNN_cnt_params'
prediction_path = 'CDGNN_cnt_params_0%s' % dataname
device = torch.device(FLAGS.device)
wdecay = 0.001
learning_rate = 0.001
batch_size = FLAGS.batch_size
mt_mem_adj_value = 0.000001
lt_mem_adj_value = 0.000001
eq_mem_adj_value = 0.0001
is_axis_mean_max_norm = True
scd = -1
method = 'KL'
load_matrix = False
KMD = 0.000001
add_A_and_Diag = False
mat_A_and_Diag = False
writedown = f'/home/user/liucheng/CDGNN_%s_%s.txt' % (dataname, datetime.now(), )
print("mat_A_and_Diag : ", mat_A_and_Diag)
print("batch_size : ", batch_size)
print("mt_mem_adj_value : ", mt_mem_adj_value)
print("lt_mem_adj_value : ", lt_mem_adj_value)
print("eq_mem_adj_value : ", eq_mem_adj_value)
print("Symmetric Correlation Degree : ", scd)
print("is_axis_mean_max_norm : ", is_axis_mean_max_norm)
print("Learning rate : ", learning_rate)
print('Model is %s' % (model_name,))
timestamp_s = datetime.now()
print("\nWorking start at ", timestamp_s, '\n')

if params_dir != "None":
    params_path = os.path.join(params_dir, model_name)
else:
    params_path = 'params/%s_%s/' % (model_name, timestamp_s)

if os.path.exists(params_path) and not FLAGS.force:
    raise SystemExit("Params folder exists! Select a new params path please!")
else:
    if os.path.exists(params_path):
        shutil.rmtree(params_path)
    os.makedirs(params_path)
    print('Create params directory %s, reading data...' % (params_path,))


def generate_all_data(batch_size_):
    all_data = read_and_generate_dataset(graph_signal_matrix_filename,
                                         num_of_weeks,
                                         num_of_days,
                                         num_of_hours,
                                         num_for_predict,
                                         points_per_hour,
                                         merge)

    # test set ground truth
    true_value = all_data['test']['target']
    print(true_value.shape)

    # training set data loader
    train_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['train']['recent']),
            torch.Tensor(all_data['train']['target'])
        ),
        batch_size=batch_size_,
        shuffle=True
    )

    # validation set data loader
    val_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['val']['recent']),
            torch.Tensor(all_data['val']['target'])
        ),
        batch_size=batch_size_,
        shuffle=False
    )

    # testing set data loader
    test_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['test']['recent']),
            torch.Tensor(all_data['test']['target'])
        ),
        batch_size=batch_size_,
        shuffle=False
    )
    return all_data, true_value, train_loader, val_loader, test_loader

if __name__ == "__main__":
    # read all data from graph signal matrix file. Input: train / valid  / test : length x 3 x NUM_POINT x 12

    all_data, true_value, train_loader, val_loader, test_loader = generate_all_data(batch_size)


    stats_data = {}
    for type_ in ['week', 'day', 'recent']:
        stats = all_data['stats'][type_]
        stats_data[type_ + '_mean'] = stats['mean']
        stats_data[type_ + '_std'] = stats['std']
    np.savez_compressed(
        os.path.join(params_path, 'stats_data'),
        **stats_data
    )
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dgl.seed(seed)
    """ Loading Data Above """
    loss_function = nn.MSELoss()
    net = ActivateGraphSahe(c_in=1, c_out=64, num_nodes=num_nodes, recent=24, K=2, Kt=3)
    net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=wdecay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay)
    optimizer = Lookahead(optimizer=optimizer)
    print("\n\n")
    his_loss = []
    validation_loss_lst = []
    train_time = []
    A0_ = np.zeros((num_nodes, num_nodes))
    A_lst = []
    if  load_matrix:
        print("Constructing Global A-matrix...  By KL method. for Data4-307p. ")
        _, _, train_loader_local, _, _ = generate_all_data(16)
        for train_r, train_t in tqdm(train_loader_local, ncols=80, smoothing=0.9):
            nodes_features_all = setup_features_tuple(train_r)
            A = setup_Adj_matrix(nodes_features_all, num_nodes)
            A_lst.append(A)
        for  val_r, val_t in tqdm(val_loader, ncols=80, smoothing=0.9):
            nodes_features_all = setup_features_tuple(val_r)
            A = setup_Adj_matrix(nodes_features_all, num_nodes)
            A_lst.append(A)
        for test_r, test_t in tqdm(test_loader, ncols=80, smoothing=0.9):
            nodes_features_all = setup_features_tuple(test_r)
            A = setup_Adj_matrix(nodes_features_all, num_nodes)
            A_lst.append(A)
        A_lst = np.array(A_lst)
        np.save('./data/PEMS08/PEMS08_adj.npy',A_lst)
        print("Saved.")
    else:
        print("Loading Adjacency matrix...  ")
        A = np.load('./data/PEMS08/PEMS08_adj.npy')
        A[np.isnan(A)] = 0.
        A[np.isinf(A)] = 0.
    train_A = A[:518]
    event_flag = 0.01
    num_nodes = 170
    hawkes_batch_size = 2
    time_split = torch.FloatTensor([10, 5]).to(device)
    u_begin = get_u_begin(train_A)
    event_list = get_event_list(A, event_flag)
    train_event, ture_event = get_event_train_and_event_ture(event_list, hawkes_batch_size)
    train_event = torch.FloatTensor(train_event).to(device)
    train_event=DataLoader(train_event,batch_size=2,drop_last=True)
    D = discriminator()
    G = generator(num_nodes, len(train_event))
    optimize = optim.Adam(params=G.parameters(), lr=0.001)
    if torch.cuda.is_available():
        D = D.to(device)
        G = G.to(device)
    ture_event = torch.FloatTensor(ture_event).to(device)
    u_begin1 = torch.FloatTensor(u_begin).to(device)
    u_begin2 = torch.FloatTensor(u_begin).to(device)
    print('\nTraining Hawkes.')
    for epoch in range(50):
        hawkes_loss_sum = 0
        for history_event, test_event in zip(train_event, ture_event):
            optimize.zero_grad()
            fake_event, u_begin1, u_begin2 = G(time_split, history_event, u_begin1, u_begin2, hawkes_batch_size,
                                               num_nodes)
            u_begin1 = u_begin1.detach()
            u_begin2 = u_begin2.detach()
            hawkes_loss = D(fake_event, test_event)
            hawkes_loss_sum = hawkes_loss_sum + hawkes_loss.item()
            optimize.step()
        print('\nEpoch({}):loss:{}'.format(epoch + 1, hawkes_loss_sum))
    print('\n\nTraining finished.')
    torch.save(G.state_dict(),'./data/PEMS08/hawkes_parament')
    G.load_state_dict(torch.load('./P8/data/PEMS08/hawkes_parament'))
    fake_event_list = []
    for history_event in train_event:
        fake_event, u_begin1, u_begin2 = G(time_split, history_event, u_begin1, u_begin2, hawkes_batch_size, num_nodes)
        fake_event = fake_event.detach().cpu().numpy()
        fake_event_list.append(fake_event)
    fake_event_list = np.array(fake_event_list)
    fake_event_list[fake_event_list >= 0.001] = 1
    fake_event_list[fake_event_list <= -0.001] = -1
    fake_event_list[(fake_event_list > 0.001) & (fake_event_list < -0.001)] = 0
    event_train = torch.FloatTensor(fake_event_list).to(device)
    A_ture, A_last = get_ture_A_and_last_A(A)
    A_ture = torch.FloatTensor(A_ture).to(device)
    A_last = torch.FloatTensor(A_last).to(device)
    predict_Adj = predict_A(num_nodes=len(A_last[0]))
    predict_Adj = predict_Adj.to(device)
    predict_optimize = optim.Adam(params=predict_Adj.parameters(), lr=0.1)
    predict_loss_fuction = nn.MSELoss()
    for i in range(200):
        predict_optimize.zero_grad()
        A_fake = predict_Adj(A_last, event_train, len(A_last[0]))
        predict_loss = predict_loss_fuction(A_fake, A_ture)
        predict_loss.backward()
        predict_optimize.step()
    A_fake = A_fake.detach().cpu().numpy()
    A = A_fake
    A=torch.FloatTensor(A).to(device)
    A[A < 1e-9] = 0
    A_train = A[:258]*KMD
    A_val = A[258:344]*KMD
    A_test = A[344:]*KMD
    temp = 1
    print('\nPredict A complete.')
    with open(writedown, mode='a', encoding='utf-8') as f:
        f.write(f"seed,epoch,train_loss,valid_loss,learning_rate,_MAE,_MAPE,_RMSE,datetime\n")
    print("ActiveGNN have {} paramerters in total.".format(sum(x.numel() for x in net.parameters())))
    watch = True
    for epoch in range(1, epochs + 1):
        train_loss = []
        start_time_train = time()
        temp = 1
        temp2 = 0
        if not watch:
            break
        for  train_r, train_t in tqdm(train_loader, ncols=80, smoothing=0.9):
            if temp >=4:
                if temp%2 ==0:
                    train_r = train_r.to(device)
                    train_t = train_t.to(device)
                    net.train()
                    optimizer.zero_grad()
                    output, _, A1 = net(train_r, A_train[temp2])
                    loss = loss_function(output, train_t)
                    loss.backward()
                    optimizer.step()
                    training_loss = loss.item()
                    train_loss.append(training_loss)
                    temp2+=1
            temp+=1

        scheduler.step()
        end_time_train = time()
        train_loss = np.mean(train_loss)
        print('Epoch step: %s, t-loss: %.4f, time: %.2fs' % (epoch, train_loss, end_time_train - start_time_train))
        train_time.append(end_time_train - start_time_train)
        valid_loss = compute_val_loss(net, val_loader, loss_function, A_val, device, epoch)
        his_loss.append(valid_loss)
        _MAE, _RMSE, _MAPE = evaluate(net, test_loader, true_value, A_test, device, epoch_=epoch)
        with open(writedown, mode='a', encoding='utf-8') as f:
            f.write(
                f"{seed},{epoch},{train_loss},{valid_loss},{scheduler.get_last_lr()[0]},{_MAE},{_MAPE},{_RMSE},{datetime.now()}\n")
        params_filename = os.path.join(params_path,
                                       '%s_epoch_%s_%s.params' % (model_name, epoch, str(round(valid_loss, 2))))
        torch.save(net.state_dict(), params_filename)
        validation_loss_lst.append(float(valid_loss))
        watch_early_stop = np.array(validation_loss_lst)
        arg = np.argmin(watch_early_stop)
        print(
            f"\t >>> Lowest v-loss in {epoch} :  epoch_{arg + 1}  {validation_loss_lst[arg]}  lr = {scheduler.get_last_lr()}\n\n")
        if validation_loss_lst[arg] < 710.0 and learning_rate == 0.001:
            learning_rate = 0.0001
            optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=wdecay)
            print("Optim changed. ")
    print("\n\nTraining finished.")
    print("Training time/epoch: %.4f secs/epoch" % np.mean(train_time))
    bestId = np.argmin(his_loss)
    print("The valid loss on best model is epoch%s, value is %s" % (str(bestId + 1), str(round(his_loss[bestId], 4))))
    best_params_filename = os.path.join(params_path, '%s_epoch_%s_%s.params' % (
    model_name, str(bestId + 1), str(round(his_loss[bestId], 2))))
    net.load_state_dict(torch.load(best_params_filename))
    start_time_test = time()
    prediction= predict(net, test_loader, A_test, device)
    end_time_test = time()
    evaluate(net, test_loader, true_value, A_test, device, epoch)
    test_time = (end_time_test - start_time_test)
    print("Test time: %.2f" % test_time)
    print("Total time: %f s" % (datetime.now() - timestamp_s).seconds)
