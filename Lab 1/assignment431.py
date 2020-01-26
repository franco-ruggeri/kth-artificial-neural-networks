import tensorflow as tf
print(tf.__version__)

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt

def mg_series(t_stop):

    beta = 0.2
    gamma = 0.1
    tau = 25

    # x(0)=1.5
    x = [1.5]
    for t in range(1,t_stop):
        delay = t-tau
        if delay < 0:
            delay = 0
        elif delay == 0:
            delay = 1.5
        else:
            delay = x[delay]

        x.append(x[-1] + ((beta*delay)/(1+delay**10)) - gamma*x[-1])

    return np.asarray(x)

def generate_data(t_start, t_stop, partition, std):
    t = np.arange(t_start, t_stop)
    x = mg_series(t_stop+5) # add 5 for labels

    if std > 0:
        x += np.random.normal(0, std, x.shape)

    inputs = [x[t-20], x[t-15], x[t-10], x[t-5], x[t]]
    inputs = np.asarray(inputs)
    labels = x[t+5]

    # size of test according to task description
    test_size = 200
    validation_size = int(np.floor(inputs.shape[1] * partition))
    training_size = inputs.shape[1] - (test_size + validation_size)

    test_inputs  = inputs[:,training_size+validation_size:training_size+validation_size+test_size]
    test_labels  = labels[training_size+validation_size:training_size+validation_size+test_size]

    train_inputs = inputs[:,:training_size]
    train_labels = labels[:training_size]
    valid_inputs = inputs[:,training_size:training_size+validation_size]
    valid_labels = labels[training_size:training_size+validation_size]

    training = {'inputs': train_inputs.T, 'labels': train_labels.T}
    validation = {'inputs': valid_inputs.T, 'labels': valid_labels.T}
    test = {'inputs': test_inputs.T, 'labels': test_labels.T}

    return training, validation, test, x

training, validation, test, mg_seris = generate_data(300,1500,0.3,0)
train_X, train_Y = training['inputs'], training['labels']
validation_X, validation_Y = validation['inputs'], validation['labels']
test_X, test_Y = test['inputs'], test['labels']


n_hidden1 = 8
n_hidden2 = 5
n_input = 5
n_output = 1
# learning parameters
learning_rate = 0.001
epochs = 20


#X = tf.placeholder("float")
#Y = tf.placeholder("float")



layers = [8,5]
model = tf.keras.Sequential()

for idx, nodes in enumerate(layers):

    if idx == 0:
        model.add(tf.keras.layers.Dense(nodes,
                                        input_dim=train_X.shape[1]))
    else:
        model.add(tf.keras.layers.Dense(nodes))


model.add(tf.keras.layers.Dense(n_output))
print(train_X.shape[1])

model.compile(loss='mean_squared_error', optimizer='adam')

train_model = model.fit(train_X, train_Y,
                        validation_data=(validation_X, validation_Y),
                        batch_size=train_X.shape[1],
                        epochs = epochs)

loss, accuracy = model.evaluate(x=test_X,
                          y=test_Y,
                          batch_size=test_X.shape[1])
print(loss, accuracy)