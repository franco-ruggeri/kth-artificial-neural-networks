import tensorflow as tf
print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt


def plot_predictions(predictions, test):
    t = np.arange(0, test.shape[0])
    p = np.array(predictions)
    plt.plot(t, p, label='prediction')
    plt.plot(t, test, label='time series')
    plt.legend()
    plt.show()

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


def neural_network(training, validation, test, network_settings):
    train_X, train_Y = training['inputs'], training['labels']
    validation_X, validation_Y = validation['inputs'], validation['labels']
    test_X, test_Y = test['inputs'], test['labels']

    n_output = 1

    model = tf.keras.Sequential()
    for idx, nodes in enumerate(network_settings['nLayers']):
        if idx == 0:
            model.add(tf.keras.layers.Dense(nodes,
                                            input_dim=train_X.shape[1],

                                            kernel_regularizer=tf.keras.regularizers.l2(l=network_settings['reg_param'] if network_settings['reg_param'] > 0 else None)
                                            ))
        else:
            model.add(tf.keras.layers.Dense(nodes,

                                            kernel_regularizer=tf.keras.regularizers.l2(l=network_settings['reg_param'] if network_settings['reg_param'] > 0 else None)
                                            ))

    model.add(tf.keras.layers.Dense(n_output))

    model.compile(loss='mean_squared_error', optimizer='adam')

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                min_delta=network_settings['min_delta'],
                                                patience=network_settings['patience'])

    history = model.fit(train_X, train_Y,
                            callbacks=[callback],
                            validation_data=(validation_X, validation_Y),
                            batch_size=train_X.shape[1],
                            epochs=network_settings['epochs'])

    #print('\nhistory dict:', history.history)
    training_loss_values = history.history['loss']
    validation_loss = history.history['val_loss']

    #print(training_loss_values)
    #print(validation_loss)

    mse_loss = model.evaluate(x=test_X,
                              y=test_Y,
                              batch_size=test_X.shape[1],
                              verbose=0)
    #print(mse_loss)

    test_predictions = model.predict(test_X)
    weights = model.layers[0].get_weights()[0]
    biases = model.layers[0].get_weights()[1]





    return test_predictions, mse_loss, weights



def two_layer_network():

    training, validation, test, mg_time_series = generate_data(300,1500,0.3, std=0)

    iterations = 1
    mse_loss_matrix = np.zeros((10,8))

    for iter in range(iterations):

        n_nodes = [8]

        for idx, node in enumerate(n_nodes):
            network_settings = {'nLayers': [node],
                                'epochs': 20,
                                'min_delta': 0.0006,
                                'patience': 3,
                                'reg_param': 0.000001}

            predictions, mse_loss, weights = neural_network(training, validation, test, network_settings)

            time_series = mg_time_series[300 + 5:1500 + 5][-predictions.size:]

            plot_predictions(predictions, time_series)



            #mse_loss_matrix[iter][idx] = mse_loss

    #print(np.mean(mse_loss_matrix,axis=0))


def three_layer_network():

    std_devs = [0, 0.03, 0.09, 0.18]

    training, validation, test, mg_time_series = generate_data(300, 1500, 0.3, std=std_devs[1])
    network_settings = {'nLayers': [8],
                        'epochs': 20,
                        'min_delta': 0.0000000006,
                        'patience': 3,
                        'reg_param': 0.001}

    predictions, mse_loss, weights = neural_network(training, validation, test, network_settings)
    time_series = mg_time_series[300 + 5:1500 + 5][-predictions.size:]

    plot_predictions(predictions, time_series)

two_layer_network()





