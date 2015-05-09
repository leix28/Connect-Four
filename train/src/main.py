import theano
import theano.tensor as T
import numpy

from DataLoader import DataLoader
from MLP import MLP

def train_model(filename):
    learning_rate = 0.05
    patience = 10000
    size = 1000
    batch = 100

    loader = DataLoader(filename, batch)
    rng = numpy.random.RandomState()


    print '... building the model'

    x = T.matrix('x')
    y = T.ivector('y')

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=12*12*5,
        n_hidden=size,
        n_out=12
    )

    cost = (
        classifier.negative_log_likelihood(y)
    )

    gparams = [T.grad(cost, param) for param in classifier.params]

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    print '... training'

    for i in xrange(patience):
        ip, op = loader.get_data();

        test_model = theano.function(
            inputs=[],
            outputs=classifier.errors(y),
            givens={
                x: ip,
                y: op
            }
        )

        train_model = theano.function(
            inputs=[],
            outputs=cost,
            updates=updates,
            givens={
                x: ip,
                y: op
            }
        )
        before = test_model()
        train_model()
        after = test_model()

        print 100.0 * i / patience, '%', before, after


    W1 = classifier.params[0].get_value()
    b1 = classifier.params[1].get_value()

    W2 = classifier.params[2].get_value()
    b2 = classifier.params[3].get_value()

    W3 = classifier.params[4].get_value()
    b3 = classifier.params[5].get_value()


    out = open('W1.txt', 'w')
    print >> out, '\n'.join(['\t'.join(['%.6f'%item for item in row]) for row in W1])
    out.close()

    out = open('b1.txt', 'w')
    print >> out, '\n'.join(['%.6f'%item for item in b1])
    out.close()

    out = open('W2.txt', 'w')
    print >> out, '\n'.join(['\t'.join(['%.6f'%item for item in row]) for row in W2])
    out.close()

    out = open('b2.txt', 'w')
    print >> out, '\n'.join(['%.6f'%item for item in b2])
    out.close()

    out = open('W3.txt', 'w')
    print >> out, '\n'.join(['\t'.join(['%.6f'%item for item in row]) for row in W3])
    out.close()

    out = open('b3.txt', 'w')
    print >> out, '\n'.join(['%.6f'%item for item in b3])
    out.close()

if __name__ == '__main__':
    train_model('train_data_shuffle.txt')
