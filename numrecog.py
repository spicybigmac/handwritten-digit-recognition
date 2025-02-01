import numpy as np
import csv
import pickle

# ---------- inputting data (idk how to speed up) ----------
with open("number_recognition/mnist/mnist_train.csv") as data:
    rawtrain = list(csv.reader(data))
    train = []; trainlabels = []
    for i in range(1,len(rawtrain)):
        if i%100 == 0:
            print(f"\n\n\n\n\n\n\n\n{round(i/len(rawtrain)*100, 1)}%")

        line = rawtrain[i]
        trainlabels.append([int(int(x) == int(line[0])) for x in range(10)])
        train.append([int(line[j])/255 for j in range(1,len(line))])
with open("number_recognition/mnist/mnist_test.csv") as data:
    rawtest = list(csv.reader(data))
    test = []; testlabels = []
    for i in range(1,len(rawtest)):
        line = rawtest[i]
        testlabels.append([int(int(x) == int(line[0])) for x in range(10)])
        test.append([int(line[j])/255 for j in range(1,len(line))])

train = np.array(train).T
trainlabels = np.array(trainlabels).T
test = np.array(test).T
testlabels = np.array(testlabels).T

# ---------- vars ----------
alpha = 0.09
epochs = 10000000

layers = 4
nodecount = [train.shape[0], 12, 15, 10]

regumulti = 0.0005

# ---------- initialize parameters ----------
with open("number_recognition/numparams.txt", "rb") as f:
    try:
        params, maxacc = pickle.load(f)
    except EOFError:
        params = {}
        maxacc = 0

if params == {}:
    for i in range(1,layers):
        params["W"+str(i)] = np.random.randn(nodecount[i], nodecount[i-1])
        params["b"+str(i)] = np.zeros([nodecount[i], 1])

# ---------- important functions ----------
def relu(arr):
    return np.maximum(0,arr)
def softmax(arr):
    exp = np.exp(arr)
    expsum = np.sum(exp,axis=0,keepdims=1)
    return exp/expsum

def computecost(A3, Y, m):
    inds = np.argmax(Y, axis=0, keepdims=1).astype(int)
    prob = A3[inds.T, np.arange(m).reshape(m,1)]

    log = np.log(prob)
    loss = -(np.sum(log)/len(log))
    return loss
    
def accuracy():
    # test acc
    Z1 = np.dot(params["W1"], test) + params["b1"]
    A1 = relu(Z1)
    
    Z2 = np.dot(params["W2"], A1) + params["b2"]
    A2 = relu(Z2)

    Z3 = np.dot(params["W3"], A2) + params["b3"]
    A3 = softmax(Z3)

    predictions = np.argmax(A3, axis=0, keepdims=1).astype(int)
    correct = np.argmax(testlabels, axis=0, keepdims=1).astype(int)
    compare = np.stack((predictions, correct), axis=1).reshape(2, testlabels.shape[1])
    acc = np.sum(compare[0, np.arange(testlabels.shape[1])] == compare[1, np.arange(testlabels.shape[1])])/testlabels.shape[1] * 100

    # train acc
    Z1 = np.dot(params["W1"], train) + params["b1"]
    A1 = relu(Z1)
    
    Z2 = np.dot(params["W2"], A1) + params["b2"]
    A2 = relu(Z2)

    Z3 = np.dot(params["W3"], A2) + params["b3"]
    A3 = softmax(Z3)

    predictions = np.argmax(A3, axis=0, keepdims=1).astype(int)
    correct = np.argmax(trainlabels, axis=0, keepdims=1).astype(int)
    compare = np.stack((predictions, correct), axis=1).reshape(2, trainlabels.shape[1])
    
    # print(compare[0, np.arange(testlabels.shape[1])] == compare[1, np.arange(testlabels.shape[1])])
    return np.sum(compare[0, np.arange(trainlabels.shape[1])] == compare[1, np.arange(trainlabels.shape[1])])/trainlabels.shape[1] * 100, acc


# ---------- create model ----------
def model(alpha, maxacc):
    for i in range(epochs):
        # forwardprop
        Z1 = np.dot(params["W1"], train) + params["b1"]
        A1 = relu(Z1)
        
        Z2 = np.dot(params["W2"], A1) + params["b2"]
        A2 = relu(Z2)

        Z3 = np.dot(params["W3"], A2) + params["b3"]
        A3 = softmax(Z3)

        if i%50 == 0:
            # print(f"Cost after epoch {i}: {computecost(A3, trainlabels, A3.shape[1])}")
            acctrain, acctest = accuracy()
            if acctest > maxacc:
                with open("number_recognition/numparams.txt", "wb") as f:
                    pickle.dump((params, maxacc), f)

            print(f"Epoch {i} train: {round(acctrain,1)}%, test: {round(acctest,1)}%")
            maxacc = max(acctest, maxacc)

        
        m = A3.shape[1]

        # backprop
        dZ3 = (A3-trainlabels)/m
        dW3 = np.dot(dZ3, A2.T) + regumulti*(params["W3"]**2) # regularization
        db3 = np.sum(dZ3, axis=1, keepdims=1)

        dZ2 = np.dot(params["W3"].T, dZ3)*(Z2>0)
        dW2 = np.dot(dZ2, A1.T) + regumulti*(params["W2"]**2)
        db2 = np.sum(dZ2, axis=1, keepdims=1)

        dZ1 = np.dot(params["W2"].T, dZ2)*(Z1>0)
        dW1 = np.dot(dZ1, train.T) + regumulti*(params["W1"]**2)
        db1 = np.sum(dZ1, axis=1, keepdims=1)

        # change params
        params["W3"] -= alpha*dW3
        params["b3"] -= alpha*db3
        params["W2"] -= alpha*dW2
        params["b2"] -= alpha*db2
        params["W1"] -= alpha*dW1
        params["b1"] -= alpha*db1


model(alpha, maxacc)