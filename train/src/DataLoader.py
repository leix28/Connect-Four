import numpy
import struct
import random
import theano
import theano.tensor as T
import cStringIO

class DataLoader(object):
    def __init__(self, filename, batch = 20):
        self.filename = filename
        self.data = open(filename, 'r')
        self.batch = batch

    def get_data(self):

        ip = numpy.zeros((self.batch, 12 * 12, 5), dtype = numpy.float32)
        op = numpy.zeros((self.batch), dtype = numpy.int32)

        i = 0
        while i < self.batch:
            line = self.data.readline()
            if len(line) == 0:
                self.data.close()
                self.data = open(self.filename, 'r')

                continue

            line = line.split();

            op[i] = numpy.int32(line[0]);

            for j in xrange(12 * 12):
                if line[1][j] == '.':
                    ip[i][j][0] = 1;
                elif line[1][j] == 'O':
                    ip[i][j][1] = 1;
                elif line[1][j] == 'X':
                    ip[i][j][2] = 1;
                elif line[1][j] == 'A':
                    ip[i][j][3] = 1;
                elif line[1][j] == 'B':
                    ip[i][j][4] = 1;


            i = i + 1

        ip = ip.reshape((self.batch, 12 * 12 * 5))
        ip = theano.shared(numpy.asarray(ip, dtype=theano.config.floatX), borrow = True)
        op = theano.shared(numpy.asarray(op, dtype=theano.config.floatX), borrow = True)
        return (ip, T.cast(op, 'int32'))
