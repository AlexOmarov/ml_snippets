from tensorboard import program

from business.mnist import start as start

if __name__ == '__main__':
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', 'logs/fit'])
    #  Launch trainings
    start.train('accuracy')

    url = tb.main()
