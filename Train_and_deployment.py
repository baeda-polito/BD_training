import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
from sklearn.metrics import mean_squared_error, r2_score
from BD_model import BuildingDynamics


j=0
lookback =12
path = 'energy_plus_simulation_results/Restaurant/'


#=============================================================================#
#LOAD PREPROCESSED TENSORS
from preprocess import preprocess_function
totimp,totlab,maxT,minT, num_dataset= preprocess_function(path,j,lookback)

#===================================================================================#
#VERIFY GPU AVAILABILITY
#torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()
# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


#=================================================================================#
# Split data into train/validation/test portions
train_x = []
train_y = []

# Test on the last dataset
test_portion = round(1/num_dataset * len(totimp))
validation_portion = round(1/num_dataset * len(totimp))
train_x = totimp[:-test_portion-validation_portion]
train_y = totlab[:-test_portion-validation_portion]
val_x = totimp[-test_portion-validation_portion:-test_portion]
val_y = totlab[-test_portion-validation_portion:-test_portion]
test_x = (totimp[-test_portion:])
test_y = (totlab[-test_portion:])
# val_x = test_x
# val_y = test_y


#Dimensions of train and test tensors
print(train_x.shape)
print(train_y.shape)
print(test_y.shape)
#
# # Test plot
# plt.plot(test_y,'g')
# plt.show()
# #Validation plot
# plt.plot(val_y,'b')
# plt.show()
#
# #Plot of target variable for train, test and validation with different colours
# x1=np.arange(0,len(train_y))
# x2=np.arange(len(train_y),len(train_y)+len(val_y))
# x3=np.arange(len(train_y)+len(val_y),len(train_y)+len(val_y)+len(test_y))
# plt.plot(x1,train_y)
# plt.plot(x2,val_y)
# plt.plot(x3,test_y)
# plt.show()


#===========================================================#
#HYPER PARAMETERS
lookback = lookback
train_episodes = 300
lr = 0.005 #0.005
num_layers = 2
num_hidden = 8
batch_size = 150



# DATA LOADER TO LOAD INPUT AND TARGET FROM TRAINING SET AND VALIDATION SET
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
train_loader = DataLoader(train_data, shuffle = True, batch_size=batch_size, drop_last=True)
validation_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
validation_loader = DataLoader(validation_data, shuffle=False, batch_size=batch_size, drop_last=True)





# create NN
#generalize the number of features and the number of timesteps by linking them to the preprocessing
n_features = train_x.shape[2]
n_timesteps=lookback


#save and load models

FILE = "models/Restaurant.pth"  #"Small_Office.pth"   "Restaurant.pth"    "Retail.pth"
# torch.save(mv_net.state_dict(),FILE)
mv_net = BuildingDynamics(n_features, n_timesteps,num_hidden,num_layers)
mv_net.load_state_dict(torch.load(FILE))
mv_net.eval()




#initialize the network,criterion and optimizer
#mv_net = MV_LSTM(n_features,n_timesteps)
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(mv_net.parameters(), lr=lr)

#initialize the training loss and the validation loss
LOSS = []
VAL_LOSS = []


#=========================================================================================#
#START THE TRAINING PROCESS
mv_net.train()

for t in range(train_episodes):

    h = mv_net.init_hidden(batch_size)  #hidden state is initialized at each epoch

    for x, label in train_loader:
        h = mv_net.init_hidden(batch_size) #since the batch is big enough, a stateless mode is used (also considering the possibility to shuffle the training examples, which increase the generalization ability of the network)
        h = tuple([each.data for each in h])
        output, h = mv_net(x.float(), h)
        label = label.unsqueeze(1) #use .unsqueeze to avoid problems with dimensions
        loss = criterion(output, label.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    LOSS.append(loss.item())


    # VALIDATION LOOP
    h = mv_net.init_hidden(batch_size)
    for inputs, labels in validation_loader:
        h = tuple([each.data for each in h])
        val_output, h = mv_net(inputs.float(), h)
        val_labels = labels.unsqueeze(1)
        val_loss = criterion(val_output, val_labels.float())
    VAL_LOSS.append(val_loss.item())
    print('step : ', t, 'Training Loss : ', loss.item(), 'Validation Loss :', val_loss.item())



#plot to verify validation and train loss, in order to avoid underfitting and overfitting
plt.plot(LOSS,'--',color='r', linewidth = 1, label = 'Train Loss')
plt.plot(VAL_LOSS,color='b', linewidth = 1, label = 'Validation Loss')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.legend()
plt.show()

#=========================================================================================#
#1h PREDICTION TEST
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True)
test_losses = []
h = mv_net.init_hidden(batch_size)

mv_net.eval()
ypred=[]
ylab=[]
for inputs, labels in test_loader:
    h = tuple([each.data for each in h])
    test_output, h = mv_net(inputs.float(), h)
    labels = labels.unsqueeze(1)
    test_output = test_output.detach().numpy()
    #RESCALE OUTPUT
    test_output = np.reshape(test_output, (-1, 1))
    test_output = test_output*(maxT-minT)+minT

    # labels = labels.item()
    labels = labels.detach().numpy()
    labels = np.reshape(labels, (-1, 1))
    #RESCALE LABELS
    labels = labels * (maxT - minT) + minT
    ypred.append(test_output)
    ylab.append(labels)

flatten = lambda l: [item for sublist in l for item in sublist]
ypred = flatten(ypred)
ylab = flatten(ylab)
ypred = np.array(ypred, dtype=float)
ylab = np.array(ylab, dtype = float)


plt.plot(ypred, color='orange', label="Predicted")
plt.plot(ylab,  linestyle="dashed", linewidth=1, label="Actual")
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
#plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlim(left=0,right=306)
plt.ylabel('Mean Air Temperature [°C]')
plt.xlabel('Time [h]')
plt.title('Temperature prediction 1 hour ahead using LSTM')
plt.legend()
plt.show()


#METRICS
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

MAPE = mean_absolute_percentage_error(ylab, ypred)
RMSE=mean_squared_error(ylab,ypred)**0.5
R2 = r2_score(ylab,ypred)

print('MAPE:%0.5f%%'%MAPE)
print('RMSE:', RMSE.item())
print('R2:', R2.item())


plt.scatter(ylab,ypred,   edgecolor= 'white', linewidth=1,alpha=0.1)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
#plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('Actual Temperature [°C]')
plt.ylabel('Predicted Temperature [°C]')
plt.title('Comparison among predicted and actual temperature')
plt.show()




#========================================================================================#
#CLOSED LOOP TEST

k = 0
CL_batch_size = 1
mv_net.eval()
h = mv_net.init_hidden(CL_batch_size)
Tout = np.zeros(len(test_y))

for b in range(0,len(test_y)):
    Current_variables = test_x[b,:, :-1]  #all variables are used t timestep t except for target variable (internal temperature)
    T_input = test_x[b,:, -1]  # internal temperature selected at timestep  t-1 t-lookback
    if k < lookback:
        if k == 0:
            T_lag = T_input
        else:
            T_lag = np.concatenate([T_input[0:lookback-k], Tout[0:k]])
    else:
        T_lag = Tout[k-lookback:k]

    inputs = np.column_stack([Current_variables, T_lag])  #stack input at timestep t with internal temperature at timestep t-1
    inputs = torch.tensor(inputs, dtype=torch.float32)
    inputs = inputs[np.newaxis, :,:]
    h = tuple([each.data for each in h])
    test_output, h = mv_net(inputs.float(), h)
    Tout[k] = test_output
    k = k+1
Tpred = Tout* (maxT - minT) + minT
Treal = test_y* (maxT - minT) + minT

#
plt.plot(Tpred, color='orange', label="Predicted")
plt.plot(Treal, linestyle="dashed", linewidth=1, label="Actual")
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
#plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlim(left=0,right=306)
plt.ylabel('Mean Air Temperature [°C]')
plt.xlabel('Time [h]')
plt.title('Simulation testbed using LSTM')
plt.legend()
plt.show()

# METRICS
MAPE_sim= mean_absolute_percentage_error(Treal, Tpred)
RMSE_sim=mean_squared_error(Treal,Tpred)**0.5
R2_sim = r2_score(Treal,Tpred)
print('MAPE_sim:%0.5f%%'%MAPE_sim)
print('RMSE_sim:%0.5f%%'%RMSE_sim)
print('R2_sim:%0.5f%%'%R2_sim)
#
#
plt.scatter(Treal,Tpred, edgecolor= 'white', linewidth=1,alpha=0.05)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
#plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('Actual Temperature [°C]')
plt.ylabel('Predicted Temperature [°C]')
plt.title('Comparison among predicted and actual temperature \n in simulation environment using LSTM')
plt.show()



#Temperature_results = pd.DataFrame()
#Temperature_results = Temperature_results.assign(real_T=ylab)
#Temperature_results = Temperature_results.assign(T_pred=y_pred)
#Temperature_results = Temperature_results.assign(T_sim=Tpred)

