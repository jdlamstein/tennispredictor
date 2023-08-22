import pyfiglet
from param_tennis import Param
from main.train import Train
import argparse

def gridsearch(csv):
    props = {'learning_rate': 1e-6, 'epochs': 100, 'batch_size': 32, 'optimizer': 'adam',
             'momentum': 0.9, 'nesterov': True}
    batch_sizes = [32, 128, 256, 512]
    lrs = [1e-5, 1e-6]
    optimizers = ['sgd', 'adam', 'adamw']
    for optimizer in optimizers:
        for lr in lrs:
            for batch_size in batch_sizes:
                props['learning_rate'] = lr
                props['batch_size'] = batch_size
                props['optimizer'] = optimizer
                p = Param(props)
                Tr = Train(csv, p)
                Tr.train()


if __name__ == '__main__':
    result = pyfiglet.figlet_format("Tennis Gridsearch Training", font="slant")
    csv = r'D:\Data\Sports\tennis\tennis_data\atp_database.csv'
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=csv, help='CSV Data for gridsearch. ')
    args = parser.parse_args()
    print(args)
    gridsearch(args.csv)
