from datasets.mnist import MNIST
from datasets.cifar10_gth import CIFAR10_GTH

dataset = CIFAR10_GTH(path='data/MNIST')
cl = 0
train_set = dataset.train(cl)
test_set  = dataset.test(cl, novel_ratio = 1)


