# import package
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

# import data
init_data = pd.read_excel('123.xlsx')
init_data = init_data.to_numpy()
init_data = init_data[:,[0,1,2,3]]

# split train and predict
num = 140

# Normalization
min_max_scaler = MinMaxScaler(feature_range=(0,1))
data = min_max_scaler.fit_transform(init_data)

# anti-normalization
data_pre = data

# split train samples and test samples
# distinguish features and labels
train_labels = data[:num,3]
train_labels = train_labels.reshape(num,1)
train_features = data[:num,[0,1,2]]
test_labels = data[num:,3]
test_labels = test_labels.reshape(201-num,1)
test_features = data[num:,[0,1,2]]

# parameters
max_epochs = 1300  # epochs
learn_rate = 0.0025  # learning_rate
#mse_final = 6.5e-4   # threehold of terminate
sample_num = train_features.shape[0]  # numbers of samples
input_num = 3  # numbers of features
out_num = 1  # numbers of labels
hidden_unit_num = 3  # nodes of hidden layers


# weight
w1 = 0.5*np.random.rand(input_num,hidden_unit_num)-0.1
# bias
b1 = 0.5*np.random.rand(1,hidden_unit_num)-0.1

# weight
w2 = 0.5*np.random.rand(hidden_unit_num,1)-0.1
# bias
b2 = 0.5*np.random.rand(1,1)-0.1


# active function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# loss function
mse_history = []
for i in range(max_epochs):
    # forward
    hidden_out = sigmoid(np.dot(train_features,w1)+b1)
    network_out = np.dot(hidden_out,w2)+b2
    # loss
    error = train_labels - network_out
    mse = np.average(np.square(error))
    mse_history.append(mse)
    #if mse<mse_final:
    #    break

    #BP
    delta2 = - error
    delta1 = np.dot(delta2,w2.T)*hidden_out*(1-hidden_out)

    # Gradient descent optimization function
    delta_w2 = np.dot(hidden_out.T,delta2)
    delta_b2 = np.dot(delta2.T,np.ones((sample_num,1)))

    delta_w1 = np.dot(train_features.T,delta1)
    delta_b1 = np.dot(np.ones((1,sample_num)),delta1)

    # update
    w2 = w2 - learn_rate * delta_w2
    b2 = b2 - learn_rate * delta_b2
    w1 = w1 - learn_rate * delta_w1
    b1 = b1 - learn_rate * delta_b1

#----------------train end-------------------------

# draw loss
plt.plot(mse_history)
plt.show()

# predict function
def predict(test_features):
    hidden_out = sigmoid(np.dot(test_features, w1) + b1)
    network_out = np.dot(hidden_out, w2) + b2
    return  network_out



# predict
prediction = predict(test_features)
prediction = prediction.reshape(1,-1)


# anti normalization
data_pre[num:,3] = prediction
prediction_inver = min_max_scaler.inverse_transform(data_pre)

# save results
np.savetxt('BP.txt',prediction_inver[:,3])

# calculate rmse
error = prediction_inver[num:,3]-init_data[num:,3]
error = np.square(error)
error = np.sum(error)
error = error/61
RMSE = np.sqrt(error)
print("RMSE = ", RMSE)


# draw results
plt.plot(prediction_inver[num:,3], label="预测", color="#F08080")
plt.plot(init_data[num:,3], label="实际", color="#DB7093", linestyle="--")
plt.show()

