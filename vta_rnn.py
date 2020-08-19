
# load the data
import pandas as pd
import numpy as np

from torch.autograd import Variable

# load the data
import pandas as pd
import numpy as np

from torch.autograd import Variable

import torch
class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dense_size, output_size):
        super(RNN, self).__init__()
        self.rnn1 = torch.nn.RNN(input_size, hidden_size, bias=True, nonlinearity='relu', batch_first=True)
        self.dense1 = torch.nn.Linear(hidden_size, dense_size, bias=True)
        self.relu1 = torch.nn.ReLU()
        self.dense2 = torch.nn.Linear(dense_size, output_size, bias=True)
        self.out = torch.nn.Softmax(output_size)

    def forward(self, embedding, hidden_size):
        rnn_out, _ = self.rnn1(embedding, torch.zeros(1,1,hidden_size, dtype=torch.float))
        dense = self.dense1(rnn_out)
        nonlinearity = self.relu1(dense)
        dense_out = self.dense2(nonlinearity)
        prediction = self.out(dense_out)

        return prediction

def load_dset(dset_name):
    dset = pd.read_pickle(dset_name)
    return dset    

def pad_equal_size(train_x, test_x):
    max_len = max(max([len(e) for e in train_x]), max([len(e) for e in test_x]))
    
    new_train_x = []
    new_test_x = []
    
    for e in train_x:
        element = np.zeros(max_len)
        element[:len(e)] = e
        new_train_x.append(list(element))
    
    for e in test_x:
        element = np.zeros(max_len)
        element[:len(e)] = e
        new_test_x.append(list(element))
    
    return new_train_x, new_test_x
    
    
# here the fold_id represents which data fold will be kept for testing
def get_fold_data(dset, fold_id):
    # remove the test index
    train_indices = list(range(0,10))
    train_indices.remove(fold_id)
    
    train_folds = dset.loc[dset['fold'].isin(train_indices)]
    test_fold = dset.loc[dset['fold']==fold_id]

    # select the columns we want and separate the x and y form the train and test folds
    train_x = train_folds['RR'].to_list()
    train_x = [list(e) for e in train_x]
    train_y = pd.get_dummies(train_folds['target']).values.tolist()
    
    test_x = test_fold['RR'].to_list()
    test_x = [list(e) for e in test_x]
    test_y = pd.get_dummies(test_fold['target']).values.tolist()
    
    # pad the sequences with zeros so they have equal lengths
    train_x, test_x = pad_equal_size(train_x, test_x)
    
    return train_x, train_y, test_x, test_y


from torch.utils.data import Dataset
class VTADataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = np.asarray(self.x[idx])
        x = x.reshape(x.shape + (1,))

        return x, self.y[idx]


def train_model(dset, fold_id):
    n_epochs = 3
    input_size = 1
    hidden_size = 50
    dense_size = 70
    output_size = 2

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda:0" if use_cuda else "cpu")
    device = torch.device("cpu")
    torch.backends.cudnn.benchmark = True

    # construct the dataset object
    train_x, train_y, test_x, test_y = get_fold_data(dset, fold_id)
    
    train_x = torch.FloatTensor(train_x).double()
    train_y = torch.FloatTensor(train_y).double()
    test_x = torch.FloatTensor(test_x).double()
    test_y = torch.FloatTensor(test_y).double()
    
    dset_train = VTADataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(dset_train, batch_size=1)

    dset_test = VTADataset(test_x, test_y)
    test_loader = torch.utils.data.DataLoader(dset_test, batch_size=1)


    # build the model and set it to the GPU
    model = RNN(input_size, hidden_size, dense_size, output_size)
    model.to(device)

    # describe the model we have built
#     print(list(model.parameters()))

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    
    loss_during = []
    acc_during = {'training':[],
                 'testing':[]}

    for epoch in range(n_epochs):
        # flag the model as training
        model.train()
        for batch, (input_batch, target_batch) in enumerate(train_loader):            
            # transfer the batch to device
            input_batch = Variable(input_batch).float().to(device)
            target_batch = Variable(target_batch).type(torch.LongTensor).to(device)

            optimizer.zero_grad()
            output = model(input_batch, hidden_size)
            loss = loss_fn(output, target_batch)

            # backpropagate error
            loss.backward()
            optimizer.step()
            
            if batch % 300 == 0:
                print(f"At batch {batch} - loss is {loss.item()}")
#                 print(f"At batch {batch} - loss is {loss.data[0]}")
            
        print(f"At epoch {epoch} - loss is {loss.item()}")
        loss_during.append(loss.item())


        # See what the scores are after one epoch
        model.eval()
        with torch.no_grad():
            correct = 0
            for input_batch, target_batch in train_loader:
                # transfer the batch to device
                input_batch = Variable(input_batch).float().to(device)
                target_batch = Variable(target_batch).type(torch.LongTensor).to(device)

                output = model(input_batch, hidden_size)
                
                prediction = np.zeros(output_size)
                out = output.tolist()[-1][-1]
                prediction[out.index(max(out))] = 1
                correct += 1 if all(prediction==target_batch.tolist()[0]) else 0
                
            print(f"The accuracy on train data at epoch {epoch} is {float(correct)/len(train_y)}")
            acc_during['training'].append(float(correct)/len(train_y))
            
            correct = 0
            for input_batch, target_batch in test_loader:
                # transfer the batch to device
                input_batch = Variable(input_batch).float().to(device)
                target_batch = Variable(target_batch).type(torch.LongTensor).to(device)

                output = model(input_batch, hidden_size)
                
                prediction = np.zeros(output_size)
                out = output.tolist()[-1][-1]
                prediction[out.index(max(out))] = 1
                correct += 1 if all(prediction==target_batch.tolist()[0]) else 0

            print(f"The accuracy on test data at epoch {epoch} is {float(correct)/len(test_y)}")
            acc_during['testing'].append(float(correct)/len(test_y))

    return loss_during, acc_during

# combine the original case data with the control data so it ready to be fed to a classifier
# combine_control_case('case_scdHeFT_15min.csv')
# combine_control_case('case_scdHeFT_2min.csv')
# combine_control_case('case_scdHeFT_4min.csv')

# describe the dataset
# describe_dataset('case_control_15.pkl')

# plot the RR for a case and control example
# plot_case_control_RR('case_control_15.pkl')

# generate new fold indices
# generate_kfold_indices()

fold_id = 1
dset = load_dset('case_control_15.pkl')

# train an RNN model on one of the datasets
loss_during, acc_during = train_model(dset, fold_id)



import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(range(len(loss_during)), loss_during)
plt.xlabel('Training epochs')
plt.ylabel('Loss')
fig.savefig("loss_training_15.pdf", bbox_inches='tight')


fig, ax = plt.subplots()
ax.plot(range(len(acc_during['training'])), acc_during['training'], label='training')
ax.plot(range(len(acc_during['testing'])), acc_during['testing'], label='testing')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
fig.savefig("acc_tr_test_15.pdf", bbox_inches='tight')