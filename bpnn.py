# coding:utf-8
import math
import random


class BpnnNode:
    '''
    ��Ԫ�ڵ�
    '''
    def __init__(self, input_num):
        '''
        :param input_num:��������
        :return:None
        '''
        self.input = [1.0] * (input_num + 1)
        self.weight = [0.0] * (input_num + 1)
        self.old_weight = [0.0] * (input_num + 1)
        for i in range(0, len(self.weight)):
            self.weight[i] = self.rand(-0.5, 0.5)
        self.delta = 0.0
        self.output = 0.0

    def rand(self,a, b):
        '''
        :param a:�½�
        :param b:�Ͻ�
        :return:���ֵ
        '''
        return (b-a)*random.random() + a

    def sigmoid(self, x):
        '''
        ��ֵ����
        '''
        return 1.0 / (1.0 + math.e ** -x)
        # return math.tanh(x)

    def dsigmoid(self, y):
        '''
        ��ֵ������
        '''
        return y * (1.0 - y)
        # return 1.0 - y**2

    def compute(self, data):
        '''
        �������ֵ
        :param data:�������
        :return:None
        '''
        if len(data) != len(self.input) - 1:
            raise ValueError('wrong length of input for node!')
        for i in range(len(self.input) - 1):
            self.input[i] = data[i]
        net = 0.0
        for i in range(len(self.input)):
            net += self.input[i] * self.weight[i]
        self.output = self.sigmoid(net)

    def update(self, weight_list, delta_list, eta, momentum):
        '''

        :param weight_list:
        :param delta_list:
        :param eta:
        :param momentum:
        :return:
        '''
        self.delta = 0.0
        for i in range(len(weight_list)):
            self.delta += weight_list[i] * delta_list[i]
        self.delta *= self.dsigmoid(self.output)
        for i in range(len(self.weight)):
            change = eta * self.delta * self.input[i]+momentum*(self.weight[i]-self.old_weight[i])
            # change= eta * self.delta * self.input[i]
            self.old_weight[i]=self.weight[i]
            self.weight[i] += change


class Bpnn:
    def __init__(self, input_num, node_num_list):
        '''

        :param input_num:
        :param node_num_list:
        :return:
        '''
        self.bpnn = []
        self.input_num = input_num
        tmp_input_num = input_num
        for node_num in node_num_list:
            self.bpnn.append([BpnnNode(tmp_input_num) for i in range(node_num)])
            tmp_input_num = node_num

    def compute(self, input):
        '''

        :param input:
        :return:
        '''
        if len(input) != self.input_num:
            raise ValueError('wrong length of input for bpnn!')
        tmp_input = input
        for node_list in self.bpnn:
            for node in node_list:
                node.compute(tmp_input)
            tmp_input = [node.output for node in node_list]

    def output(self):
        '''

        :return:
        '''
        return [node.output for node in self.bpnn[-1]]

    def error(self, example_list):
        '''

        :param example_list:
        :return:
        '''
        error = []
        for example in example_list:
            self.compute(example[0])
            output = self.output()
            target = example[1]
            e = 0.
            for i in range(len(output)):
                e += (target[i] - output[i]) ** 2
            error.append(e / 2.)
        return error

    def update(self, target, eta, momentum):
        '''

        :param target:
        :param eta:
        :param momentum:
        :return:
        '''
        if len(target) != len(self.bpnn[-1]):
            raise ValueError('wrong length of target for bpnn!')
        for i in range(len(self.bpnn[-1])):
            self.bpnn[-1][i].update([1.0], [target[i] - self.bpnn[-1][i].output], eta, momentum)
        tmp_list = range(len(self.bpnn) - 1)
        tmp_list.reverse()
        for i in tmp_list:
            delta_list = [node.delta for node in self.bpnn[i + 1]]
            for j in range(len(self.bpnn[i])):
                weight_list = [node.old_weight[j] for node in self.bpnn[i + 1]]
                self.bpnn[i][j].update(weight_list, delta_list, eta, momentum)

    def train(self, example_list, error, eta=0.3, momentum=0.2):
        '''

        :param example_list:
        :param error:
        :param eta:
        :param momentum:
        :return:
        '''
        error_now=sum(self.error(example_list))
        self.n = 0
        while  error_now> error:
            self.n += 1
            if self.n % 50 ==0:
                print(' error_now = %f  train_times_now = %d ' % (error_now, self.n))
            # print '\n----------\n'
            for example in example_list:
                self.compute(example[0])
                # print self.output()
                self.update(example[1], eta, momentum)
            error_now=sum(self.error(example_list))
            # print error_now
        print('error_n: %d' % self.n)

    def debug_input(self):
        '''

        :return:
        '''
        print 'debug input'
        for i in range(len(self.bpnn)):
            print '\t', 'layer ', i # \t Tab
            for node in self.bpnn[i]:
                print '\t\t', node.input

    def debug_output(self):
        '''

        :return:
        '''
        print 'debug output'
        for i in range(len(self.bpnn)):
            # print '\t', 'layer ', i
            print '\t\t', [node.output for node in self.bpnn[i]]

    def debug_delta(self):
        '''

        :return:
        '''
        print 'debug delta'
        for i in range(len(self.bpnn)):
            print '\t', 'layer ', i
            for node in self.bpnn[i]:
                print '\t\t', node.delta

    def debug_weight(self):
        '''

        :return:
        '''
        print 'debug weight'
        for i in range(len(self.bpnn)):
            print '\t', 'layer ', i
            for node in self.bpnn[i]:
                print '\t\t', node.weight

    def debug_train(self, example_list, times, eta=0.4, momentum=0.3):
        '''

        :param example_list:
        :param times:
        :param eta:
        :param momentum:
        :return:
        '''
        for i in range(times): #训练次数
            error = 0.0 #本次误差
            for example in example_list: #数据中每行
                self.compute(example[0])
                self.update(example[1], eta, momentum)
            if i % 80 == 0:
                print('    [%3.2f %%] ' % ((i*1.0/times)*100))


# example_list = [
#     [[0, 0, 0],[0]],
#     [[0, 1, 0],[0]],
#     [[1, 0, 0],[0]],
#     [[0, 1, 1],[1]],
#     [[1, 0, 1],[1]],
#     [[1, 1, 0],[0]]
# ]
# bpnn = Bpnn(3, [4 ,4,1])
# bpnn.train(example_list, 0.001)
# bpnn.debug_train(example_list,10000)
# bpnn.compute([0.9,0.9])
# print bpnn.output()