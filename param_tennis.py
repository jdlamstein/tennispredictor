import os
import glob
import datetime

__author__='Josh Lamstein'
class Param:
    def __init__(self, datadir, props=None):
        self.train_len = 120913
        self.val_len = 25910
        self.test_len = 25910

        self.input_size = (36,)
        self.output_size = 2

        self.parent = datadir
        self.timestring = update_timestring()
        self.tfrecord_dir = os.path.join(self.parent, 'tfrecords')
        self.model_dir = os.path.join(self.parent, 'models')
        self.data_dir = os.path.join(self.parent, 'tennis_data')
        self.fig_dir = os.path.join(self.parent, 'figures')
        self.atp_dir = os.path.join(self.parent, 'tennis_atp')
        if props is None:
            self.batch_size = 128
            self.epochs = 100
            self.learning_rate = 1e-6
            self.optimizer = 'adam'
            self.momentum = 0.9  # sgd
            self.nesterov = True  # sgd
        else:
            self.batch_size = props['batch_size']
            self.epochs = props['epochs']
            self.learning_rate = props['learning_rate']
            self.optimizer = props['optimizer']
            self.momentum = props['momentum']
            self.nesterov = props['nesterov']

        self.dirs = [self.tfrecord_dir, self. model_dir, self.data_dir, self.fig_dir, self.atp_dir]
        for d in self.dirs:
            if not os.path.exists(d):
                os.mkdir(d)

def update_timestring():
    now = datetime.datetime.now()
    timestring = '%.4d_%.2d_%.2d_%.2d_%.2d_%.2d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
    return timestring
