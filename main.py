from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from subprocess import check_output
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import csv
import os


batch_size = 30
time_series = 40
model_size = [128, 128, 128, 128]


class JimoLoadingBar():
    prefix = '|'
    postfix = '|{processed:5} /{pending:5}, {percentage:7.2%}'
    phases = ' ▏▎▍▌▋▊▉█'

    def __init__(self, pending=None):
        self.rows, self.columns = map(int,
                                      check_output(['stty', 'size']).split())
        self.data = {'percentage': 0., 'processed': 0, 'pending': pending}
        self.pending = pending

    def update(self, processed, postfix_data={None: None},
               prefix_data={None: None}):
        if self.pending is not None:
            self.data.update({'processed': processed})
        self.data['percentage'] = min(processed / (self.pending or 1.), 1.)
        bar_length = (self.columns -
                      len(self.postfix.format(**self.data, **postfix_data)) -
                      len(self.prefix.format(**self.data, **prefix_data)) - 1)
        occupied_space = self.data['percentage'] * bar_length
        full_element = self.phases[-1] * int(occupied_space)
        partial_element = (self.phases[int(occupied_space %
                                       1 * len(self.phases))]
                           if len(full_element) < bar_length else '')
        empty_element = self.phases[0] * (bar_length - len(full_element) -
                                          len(partial_element))
        print('\r' + self.prefix.format(**self.data, **prefix_data) +
              full_element + partial_element + empty_element +
              self.postfix.format(**self.data, **postfix_data),
              end='', flush=True)


def read_stocks(symbol, return_date=False, reverse=True):
    date = []
    value = []
    with open('data/{}'.format(symbol)) as f:
        reader = csv.reader(f)
        next(reader)
        try:
            for row in reader:
                if row == []:
                    break
                if return_date:
                    date.append(datetime.strptime(row[0], '%d-%b-%y'))
                value.append(float(row[4]))
        except ValueError:
            for row in reader:
                if row == []:
                    break
                if return_date:
                    date.append(datetime.strptime(row[0], '%Y-%m-%d'))
                value.append(float(row[4]))
    if return_date:
        if (date[0] < date[1]):
            return np.array(date), np.array(value)
        return np.flip(np.array(date), 0), np.flip(np.array(value), 0)
    elif reverse:
        return np.flip(np.array(value), 0)
    else:
        return np.array(value)


def normalize(data):
    new_data = np.array([(np.append([0], i) -
                          np.append(i, [0]))[:-1] for i in data])
    return np.negative(new_data[:, 0]), np.array(new_data[:, 1:])


def recover(start, data):
    new_data = []
    for start_num, i in zip(start, data):
        new_data.append([start_num])
        for j in i:
            new_data[-1].append(new_data[-1][-1] + j)
    return np.array(new_data)


def get_dates(start_date, num_days):
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    dates = pd.date_range(start_date, periods=num_days, freq=us_bd)
    return np.array(dates.to_pydatetime(), dtype=np.datetime64)


def get_model():
    # batch_size, time_series
    placeholder = tf.placeholder(tf.float32, shape=(None, None))
    stock_values = tf.expand_dims(placeholder, 2)

    x_data = stock_values[:, :-1]
    y_data = stock_values[:, 1:]

    cell = [tf.nn.rnn_cell.GRUCell(i) for i in model_size]
    cell = tf.nn.rnn_cell.MultiRNNCell(cell)
    cell = tf.contrib.rnn.OutputProjectionWrapper(cell, 1)
    val, _ = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
    y_data, val = tf.squeeze(y_data, [2]), tf.squeeze(val, [2])

    cost = tf.losses.mean_squared_error(y_data, val)
    train_fn = tf.train.AdamOptimizer().minimize(cost)

    eval_max = tf.reduce_max(tf.abs(val - y_data))
    eval_min = tf.reduce_min(tf.abs(val - y_data))
    eval_mean, eval_var = tf.nn.moments(tf.reshape(val - y_data, [-1]), [0])

    pred_fn = tf.concat((y_data[:, 0:1], val), 1)

    return (placeholder, train_fn, (eval_mean, eval_max, eval_min, eval_var),
            pred_fn)


def get_data():
    data = []
    for i in range(batch_size):
        value = read_stocks(random.choice(os.listdir('./data/')))
        index = np.random.randint(len(value) - time_series)
        data.append(value[index:index + time_series])
    return normalize(np.array(data))


def get_pred_data():
    date, data = read_stocks(random.choice(os.listdir('./data/')),
                             return_date=True)
    index = np.random.randint(len(date) - time_series)
    start, data = normalize(np.array([data[index:index + time_series]]))
    date = np.array([date[index:index + time_series]])
    return start, date, data


def train_model(session, placeholder, train_fn, data_fn, steps):
    bar = JimoLoadingBar(steps)
    bar.update(0)
    for i in range(steps):
        start, data = data_fn()
        session.run(train_fn, {placeholder: data})
        bar.update(i)
    bar.update(steps)
    print()


def eval_model(session, placeholder, eval_fn, data_fn):
    start, data = data_fn()
    mean, _max, _min, var = sess.run(eval_fn, {placeholder: data})
    print('mean: {:.3f}, max: {:.3f}, '
          'min: {:.3f}, var: {:.3f}'.format(abs(mean), _max, _min, var))
    return mean, _max, _min, var


def predict_model(session, placeholder, pred_fn, data_fn, length):
    start, date, data = data_fn()
    result = session.run(pred_fn, {placeholder: data})
    prediction = np.copy(result[:, :-1])
    for i in range(length):
        prediction = np.append(prediction, result[:, -1:], axis=1)
        result = sess.run(pred_fn, {placeholder: prediction[:, -time_series:]})
    prediction = np.append(prediction, result[:, -1:], axis=1)
    prediction = recover(start, prediction)
    dates = np.array([get_dates(date[i][0], len(prediction[i]))
                     for i in range(len(date))])
    data = recover(start, data)
    return date, data, dates, prediction


if __name__ == '__main__':
    print('initializing model...')
    placeholder, train_fn, eval_fn, pred_fn = get_model()
    sess = tf.Session()
    saver = tf.train.Saver()
    try:
        saver.restore(sess, tf.train.latest_checkpoint('./.model/'))
    except ValueError:
        sess.run(tf.global_variables_initializer())

    # """
    date, data, dates, prediction = predict_model(sess, placeholder, pred_fn,
                                                  get_pred_data, 100)
    plt.plot(date[0], data[0], dates[0], prediction[0])
    plt.show()
    # """

    """
    try:
        for i in range(100):
            train_model(sess, placeholder, train_fn, get_data, 100)
            mean, _max, _min, var = eval_model(sess, placeholder,
                                            eval_fn, get_data)
    except KeyboardInterrupt:
        saver.save(sess, './.model/model')
    # """
