from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from subprocess import check_output
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import pathlib
import csv
import os


max_time = 30
batch_size = 1
pred_input_series = 90


symbols = ['AIRT', 'ATSG', 'ALK', 'ALGT', 'AAL', 'ARCB', 'ASC', 'AAWW', 'AVH',
           'AZUL', 'BSTI', 'BCO', 'BRS', 'CHRW', 'CNI', 'CP', 'CPLP', 'CEA',
           'ZNH', 'VLRS', 'CPA', 'CAAP', 'CMRE', 'CVTI', 'CYRX', 'CYRXW',
           'CSX', 'DAC', 'DAL', 'DHT', 'DCIX', 'DSX', 'DSXN', 'LPG', 'DRYS',
           'EGLE', 'ECHO', 'ERA', 'EURN', 'ESEA', 'EXPD', 'FDX', 'FWRD',
           'FRO', 'GNK', 'GWR', 'GSL', 'GLBS', 'GOL', 'GRIN', 'OMAB', 'PAC',
           'ASR', 'GSH', 'HA', 'HTLD', 'HUBG', 'HUNT', 'HUNTU', 'HUNTW',
           'JBHT', 'JBLU', 'KSU', 'KSU', 'KNX', 'LSTR', 'LTM', 'MRTN',
           'NVGS', 'NNA', 'NM', 'NMM', 'NAO', 'NSC', 'ODFL', 'OSG', 'PTSI',
           'PANL', 'PATI', 'PHII', 'PHIIK', 'PXS', 'RLGT', 'RRTS', 'RYAAY',
           'SB', 'SAIA', 'SNDR', 'SALT', 'SLTB', 'SBBC', 'SBNA', 'STNG',
           'CKH', 'SMHI', 'SHIP', 'SHIPW', 'SSW', 'SSWA', 'SSWN', 'SFL',
           'SINO', 'SKYW', 'LUV', 'SAVE', 'SBLK', 'SBLKZ', 'GASS', 'TK',
           'TOPS', 'TRMD', 'TNP', 'USX', 'UNP', 'UAL', 'UPS', 'ULH', 'USAK',
           'USDP', 'WERN', 'YRCW', 'ZTO']


symbols = ['AIRT']


use_symbols = symbols[:10]


stock_keys = ['date', 'open', 'high', 'low', 'close', 'volume']
stock_types = {i: [lambda i: datetime.datetime.strptime(i, '%Y-%m-%d'),
                   float, float, float, float, int][idx]
               for idx, i in enumerate(stock_keys)}


def get_stock(symbol):
    with open('./Data/{}.csv'.format(symbol)) as f:
        data = {i: [] for i in stock_keys}
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            for i, j in zip(stock_keys, row):
                data[i].append(stock_types[i](j))
        date = np.array(data.pop('date'))
        return (np.array(date, dtype=np.datetime64),
                {i: np.array(j) for i, j in data.items()})


def input_fn(fn_type):
    def generator(features, labels, *args):
        if fn_type in ['train', 'evaluate']:
            for i in range(args[0]):
                symbol = symbols[np.random.randint(len(symbols))]
                stock_date, stock_data = get_stock(symbol)
                rand_idx = np.random.randint(len(stock_date) - max_time)
                yield {}, {i: np.flip(j[rand_idx:rand_idx + max_time], 0)
                           for i, j in stock_data.items() if i == 'close'}
        elif fn_type == 'predict':
            symbol = symbols[np.random.randint(len(symbols))]
            stock_date, orig_stock_data = get_stock(symbol)
            rand_idx = np.random.randint(len(stock_date) -
                                         pred_input_series)
            stock_data = {i: np.flip(j[rand_idx:rand_idx +
                                       pred_input_series], 0)
                          for i, j in orig_stock_data.items() if i == 'close'}
            yield ({}, stock_data, {'symbol': symbol,
                                    'date': np.flip(stock_date, 0),
                                    'data': {i: np.flip(j, 0) for i, j in
                                             orig_stock_data.items()},
                                    'idx': rand_idx})
    return generator


def get_dates(start_date, num_days):
    start_date += np.timedelta64(1, 'D')
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    dates = pd.date_range(start_date, periods=num_days, freq=us_bd)
    return np.array(dates.to_pydatetime(), dtype=np.datetime64)


def plot_stock(date, data, symbol):
    plt.plot(date, data['close'], label=symbol)
    plt.legend()


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


class custom_hook():
    def __init__(self, estimator):
        self.estimator = estimator

    def begin_train(self, *args):
        pass

    def begin_train_step(self, *args):
        pass

    def after_train_step(self, *args):
        pass

    def after_train(self, *args):
        pass

    def begin_evaluate(self, *args):
        pass

    def begin_evaluate_step(self, *args):
        pass

    def after_evaluate_step(self, *args):
        pass

    def after_evaluate(self, *args):
        pass


class LoggingHook(custom_hook):
    def begin_train(self, *args):
        self.step = 0
        self.num_steps = args[0]
        self.cost = 0.
        self.loading_bar = JimoLoadingBar(pending=self.num_steps)
        self.loading_bar.prefix = 'training |'
        self.loading_bar.postfix = ('|{_processed:5} /{_pending:5}, '
                                    '{percentage:7.2%}, cost: {cost:.2E}')
        self.loading_bar.phases = ' ▏▎▍▌▋▊▉█'
        self.pending = self.estimator.step + self.num_steps

    def begin_train_step(self, *args):
        self.step += 1
        self.loading_bar.update(self.step, {'_processed': self.estimator.step,
                                            '_pending': self.pending,
                                            'cost': self.cost})

    def after_train_step(self, *args):
        self.loading_bar.update(self.step, {'_processed': self.estimator.step,
                                            '_pending': self.pending,
                                            'cost': args[0]})
        self.cost = args[0]

    def after_train(self, *args):
        self.step += 1
        self.loading_bar.update(self.step, {'_processed': self.estimator.step,
                                            '_pending': self.pending,
                                            'cost': self.cost})
        self.estimator.save()
        print()

    def begin_evaluate(self, *args):
        print('\033[1mbegin evaluation...\033[0m\t\t\t\t', end='', flush=True)

    def after_evaluate(self, *args):
        # accuracy, max, min, stddev
        print('\r\033[1maccuracy: \033[93m{0:.2}\033[0m'
              '        \033[1mstandard deviation: \033[93m{3:.2}\033[0m'
              '        \033[1mmax: \033[93m{1:.2}\033[0m'
              '        \033[1mmin: \033[93m{2:.2}\033[0m'.format(*args))


def change_checkpoint(instance_name, checkpoint_path):
    with open('./.model/'+instance_name+'/checkpoint') as f:
        lines = f.readlines()
    lines[0] = ('model_checkpoint_path: '
                '"{}"\n'.format(os.path.basename(checkpoint_path)))
    with open('./.model/'+instance_name+'/checkpoint', 'w') as f:
        f.writelines(lines)


def get_checkpoint(instance_name):
    with open('./.model/'+instance_name+'/checkpoint') as f:
        return './.model/'+instance_name+'/'+f.readlines()[0].split('"')[1]


class neural_network():
    def __init__(self, features, labels, hidden_units=[128],
                 learning_rate=.001, beta1=.9, beta2=.999, epsilon=1e-8,
                 dropout=0., cost_fn=None,
                 cell_type=tf.contrib.rnn.LayerNormBasicLSTMCell, debug=False,
                 **kwargs):
        tf.reset_default_graph()
        self.name = 'stockRegressor'
        self.features = features
        self.labels = labels
        self._hook = LoggingHook(self)
        self._debug = debug
        self._debug_ops = []
        self._initialize_layers(hidden_units, learning_rate, beta1, beta2,
                                epsilon, dropout, cost_fn, cell_type)
        self._saver = tf.train.Saver(max_to_keep=10)
        self._sess = tf.Session()
        self._merge = tf.summary.merge_all()
        if pathlib.Path('./.model/'+self.name+'/checkpoint').exists():
            self._saver.restore(self._sess, get_checkpoint(self.name))
        else:
            self._sess.run((tf.global_variables_initializer(),
                            tf.tables_initializer()))
            pathlib.Path('./.model/'+self.name).mkdir(parents=True,
                                                      exist_ok=True)
        self._writer = tf.summary.FileWriter('./.model/'+self.name,
                                             self._sess.graph)

    def train(self, batch_fn, steps):
        if not self._debug:
            self._hook.begin_train(steps)
        for feature_batch, label_batch in batch_fn(self.features,
                                                   self.labels, steps):
            feed_dict = {self._feature_placeholders[i]:
                         feature_batch[i] for i in self.features.keys()}
            labels = {self._label_placeholders[i]: label_batch[i] for i in
                      self.labels.keys()}
            feed_dict.update(labels)
            if not self._debug:
                self._hook.begin_train_step()
            summary, _, cost, debug = self._sess.run((self._merge,
                                                      self._train_fn,
                                                      self._cost_fn,
                                                      self._debug_ops),
                                                     feed_dict)
            if self._debug:
                print(debug)
            self._writer.add_summary(summary, self.step)
            if not self._debug:
                self._hook.after_train_step(cost)
        if not self._debug:
            self._hook.after_train()

    def evaluate(self, batch_fn, num_steps):
        accuracy = []
        if not self._debug:
            self._hook.begin_evaluate()
        for feature_batch, label_batch in batch_fn(self.features, self.labels,
                                                   num_steps):
            feed_dict = {self._feature_placeholders[i]:
                         feature_batch[i] for i in self.features.keys()}
            labels = {self._label_placeholders[i]: label_batch[i] for i in
                      self.labels.keys()}
            feed_dict.update(labels)
            if not self._debug:
                self._hook.begin_evaluate_step()
            summary, result, debug = self._sess.run((self._merge,
                                                     self._evaluate_fn,
                                                     self._debug_ops),
                                                    feed_dict)
            if self._debug:
                print(debug)
            self._writer.add_summary(summary, self.step)
            if not self._debug:
                self._hook.after_evaluate_step()
            accuracy.append(result)
        max = np.max(accuracy)
        min = np.min(accuracy)
        stddev = np.std(accuracy)
        accuracy = np.mean(accuracy)
        if not self._debug:
            self._hook.after_evaluate(accuracy, max, min, stddev)
        return accuracy

    def predict(self, batch_fn, predict_length):
        for feature_batch, label_batch, batch_info in batch_fn(self.features,
                                                               self.labels):
            feed_dict = {self._feature_placeholders[i]:
                         feature_batch[i] for i in self.features.keys()}
            labels = {self._label_placeholders[i]: label_batch[i] for i in
                      self.labels.keys()}
            feed_dict.update(labels)
            pred_values = {i: [] for i in self.labels.keys()}
            for idx in range(predict_length):
                summary, result, debug = self._sess.run((self._merge,
                                                         self._predict_fn,
                                                         self._debug_ops),
                                                        feed_dict)
                if self._debug:
                    print(debug)
                for idx, i in enumerate(self._label_order):
                    key = self._label_placeholders[i]
                    feed_dict[key] = np.concatenate((feed_dict[key][1:],
                                                     [result[-1, idx]]))
                    pred_values[i].append(result[-1, idx])
                self._writer.add_summary(summary, self.step)
            yield pred_values, batch_info

    def save(self, checkpoint=True):
        pathlib.Path('./.model/'+self.name).mkdir(parents=True, exist_ok=True)
        if checkpoint:
            self._saver.save(self._sess, './.model/{0}/{0}'.format(self.name),
                             global_step=self.step)
        else:
            self._saver.save(self._sess, './.model/{0}/{0}'.format(self.name))

    def close(self):
        self._sess.close()

    @property
    def step(self):
        return self._sess.run(self._step)

    def _initialize_layers(self, hidden_units, learning_rate, beta1, beta2,
                           epsilon, dropout, cost_fn, cell_type):
        self._feature_placeholders = {}
        self._label_placeholders = {}
        self._label_order = []
        feature_columns = []
        label_columns = []
        for i, j in self.features.items():
            placeholder, column = self._generate_placeholder(i, j)
            self._feature_placeholders[i] = placeholder
            feature_columns.append(column)
        for i, j in self.labels.items():
            placeholder, column = self._generate_placeholder(i, j)
            self._label_placeholders[i] = placeholder
            self._label_order.append(i)
            label_columns.append(column)
        feature_columns = (tf.feature_column.input_layer(
                           self._feature_placeholders, feature_columns)
                           if self.features else None)
        label_columns = (tf.feature_column.input_layer(
                         self._label_placeholders, label_columns))
        # self._normalize_ratio = tf.reduce_max(label_columns)
        # label_columns /= self._normalize_ratio
        y, state = self._generate_hidden_layers(feature_columns,
                                                label_columns[:-1, :],
                                                len(self.labels.keys()),
                                                hidden_units, dropout,
                                                cell_type)
        # label_columns *= self._normalize_ratio
        self._debug_ops.append(y)
        # y *= self._normalize_ratio
        self._step = tf.Variable(0, name='global_step', trainable=False)
        self._evaluate_fn = self._get_evaluation(y, label_columns[1:, :])
        self._predict_fn = self._get_result(y)
        self._cost_fn = self._get_cost(y, label_columns[1:, :], cost_fn)
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1,
                                           beta2, epsilon)
        self._train_fn = optimizer.minimize(self._cost_fn, self._step,
                                            name='Train')
        tf.summary.histogram('cost', self._evaluate_fn)

    def _generate_placeholder(self, name, info):
        fc = tf.feature_column
        ph, ic, ccvl, cchb = (tf.placeholder, fc.indicator_column,
                              fc.categorical_column_with_vocabulary_list,
                              fc.categorical_column_with_hash_bucket)
        if type(info) is dict:
            if info['type'] == 'vocab':
                placeholder = ph(tf.string, shape=[None], name=name)
                return placeholder, ic(ccvl(name, info['vocab']))
            elif info['type'] == 'hash':
                placeholder = ph(tf.string, shape=[None], name=name)
                return placeholder, ic(cchb(name, info['num_buckets']))
        else:
            if info == 'numeric':
                placeholder = ph(tf.float32, shape=[None], name=name)
                return placeholder, tf.feature_column.numeric_column(name)
        raise NotImplementedError('type {} not supported'.format(
                                info['type'] if type(info) is dict else info))

    def _generate_hidden_layers(self, feature_columns, label_columns,
                                num_labels, hidden_units, dropout, cell_type):
        label_columns = tf.expand_dims(label_columns, 0)
        cells = [cell_type(num_units=i) for i in hidden_units]
        cells = tf.nn.rnn_cell.MultiRNNCell(cells)
        cells = tf.contrib.rnn.OutputProjectionWrapper(cells, num_labels)
        result, state = tf.nn.dynamic_rnn(cells, label_columns,
                                          dtype=label_columns.dtype)
        return tf.squeeze(result, [0]), state

    def _get_evaluation(self, y, y_true):
        return tf.reduce_mean(tf.abs(y-y_true))

    def _get_result(self, y):
        return y

    def _get_cost(self, y, y_true, cost_fn):
        return (cost_fn(y_true, y) if cost_fn is not None else
                tf.losses.mean_squared_error(y_true, y))


rnn = neural_network({}, {'close': 'numeric'},
                     hidden_units=[128, 128, 128, 128], debug=False)
# """
while True:
    rnn.train(input_fn('train'), 100)
    rnn.evaluate(input_fn('evaluate'), 100)
# """
for result, info in rnn.predict(input_fn('predict'), 1000):
    plot_stock(info['date'], info['data'], info['symbol'])
    idx = info['idx'] + pred_input_series
    dates = get_dates(info['date'][idx], len(next(iter(result.values()))))
    plot_stock(dates, result, 'Prediction')

plt.show()
