# train a neural network to recognize drawn numbers

from mlpt import network
import numpy as np
import pickle
import gzip
import random

def load_data():
    """Load data from pickle"""
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere"""
    e = np.zeros(10)
    e[j] = 1.0
    return e

def load_data_wrapper():
    """Format data into nice arrays"""
    # load
    tr_d, va_d, te_d = load_data()
    # format
    training_inputs = [np.reshape(x, 784) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = (training_inputs, training_results)

    validation_inputs = [np.reshape(x, 784) for x in va_d[0]]
    validation_results = [vectorized_result(y) for y in va_d[1]]
    validation_data = (validation_inputs, validation_results)

    test_inputs = [np.reshape(x, 784) for x in te_d[0]]
    test_results = [vectorized_result(y) for y in te_d[1]]
    test_data = (test_inputs, test_results)
    # return
    return (training_data, validation_data, test_data)

def linked_shuffle(lis1, lis2):
    """shuffle two lists identically"""
    c = list(zip(lis1, lis2))
    random.shuffle(c)
    return list(zip(*c))

def max_ind(lis):
    """return the index of the maximum value in lis"""
    max_val = lis[0]
    ind = 0
    for f in range(len(lis[1:])):
        if lis[f+1] > max_val:
            max_val = lis[f+1]
            ind = f+1
    return ind

def num_correct(net, data):
    """returns how many the network gets correct"""
    num = 0
    for f in range(len(data[0])):
        if max_ind(net.evaluate(data[0][f])) == max_ind(data[1][f]):
            num += 1
    return num


def train():
    """creat and train a network"""
    name = "num_rdr_16"
    print("\nTraining", name)
    num_rdr = network(784, 40, 10)
    training_data, validation_data, test_data = load_data_wrapper()
##    print(num_rdr.avg_cost(training_data[0], training_data[1]))
    costs = num_rdr.all_cost(training_data[0], training_data[1])
    print(sum(costs)/len(costs))
    temp_data = list(zip(*sorted(zip(training_data[0], training_data[1], costs), key=lambda x: -x[2])))
    training_data = temp_data[:2]
    costs = temp_data[2]
    slsz = 10
    target = 0.01
    last_ch = []
    step = 1
    while costs[0] > target:
        n = 0
        while 1:
            if not n%100:
                print('.', end='')
            sl = slice(slsz*n, slsz*(n+1))
            if not training_data[0][sl] or costs[slsz*n] < target:
                break
            last_ch = num_rdr.back_propagate(training_data[0][sl], training_data[1][sl], l_rate=step, momentum=last_ch)
            last_ch *= 0.35 # momentum factor
            n += 1
        print()
        prev_cost = sum(costs)/len(costs)
        costs = num_rdr.all_cost(training_data[0], training_data[1])
        print(sum(costs)/len(costs))
        temp_data = list(zip(*sorted(zip(training_data[0], training_data[1], costs), key=lambda x: -x[2])))
        training_data = temp_data[:2]
        costs = temp_data[2]
        if sum(costs)/len(costs) > prev_cost:
            step *= 0.9  # learning rate decay
        print('step', step)

##        step = num_rdr.avg_cost(training_data[0], training_data[1])
##        print(step)
##        training_data = linked_shuffle(*training_data)

        file = open("networks/"+name+".pkl", "wb")
        pickle.dump(num_rdr, file)
        file.close()

def test():
    """show the fruits of our labour"""
    name = "num_rdr_15"
    file = open("networks/"+name+".pkl", "rb")
    num_rdr = pickle.load(file)
    file.close()
    training_data, validation_data, test_data = load_data_wrapper()
    print()
    print("Network", name)

    # calculate various metrics
    td_ln = len(training_data[0])
    vd_ln = len(validation_data[0])
    sd_ln = len(test_data[0])
    ac_td = num_rdr.avg_cost(training_data[0], training_data[1])
    nc_td = num_correct(num_rdr, training_data)
    ac_vd = num_rdr.avg_cost(validation_data[0], validation_data[1])
    nc_vd = num_correct(num_rdr, validation_data)
    ac_sd = num_rdr.avg_cost(test_data[0], test_data[1])
    nc_sd = num_correct(num_rdr, test_data)

    # output data nice
    print("Average cost over training data: {0:.4}".format(ac_td))
    print("The network corectly identified {0}/{1} or {2:.3g}%".format(nc_td, td_ln, 100*nc_td/td_ln))
    print("Average cost over validation data: {0:.4}".format(ac_vd))
    print("The network corectly identified {0}/{1} or {2:.3g}%".format(nc_vd, vd_ln, 100*nc_vd/vd_ln))
    print("Average cost over test data: {0:.4}".format(ac_sd))
    print("The network corectly identified {0}/{1} or {2:.3g}%".format(nc_sd, sd_ln, 100*nc_sd/sd_ln))


if __name__ == '__main__':
    test()
    print('\nFinished')
