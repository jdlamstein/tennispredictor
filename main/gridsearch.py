import pyfiglet
from param_tennis import Param
from main.train import Train

def gridsearch():
    props = {'learning_rate':1e-6, 'epochs':100, 'batch_size':32}
    csv = r'D:\Data\Sports\tennis\tennis_data\atp_database.csv'
    models = ['vgg19']
    batch_sizes = [16, 32, 64]
    lrs = [1e-5, 1e-6]
    optimizers = ['sgd', 'adam', 'adamw']
    l2s = [0]
    wds = [1e-5]
    momentums = [.9]
    regs = [None]

    for lr in lrs:
        for batch_size in batch_sizes:
            props['learning_rate'] = lr
            props['batch_size'] = batch_size
            p = Param(props)
            Tr = Train(csv, p)
            Tr.train()

if __name__=='__main__':
    result = pyfiglet.figlet_format("Tennis Gridsearch Training", font="slant")
    gridsearch()