#__author__ = 'cuihe'
# coding:utf-8

import math
import random
import BPNN


random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix I*J filled by fill, default=0.0
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def S_fy(x):
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        # create weights
        self.wi = makeMatrix(self.ni, self.nh) #神经网络第一层 第二层的连接权值
        self.wo = makeMatrix(self.nh, self.no) #神经网络第二层 第三层的连接权值
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-2.0, 2.0)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-1.0, 1.0)

        # last change in weights for momentum
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        #按照已有的权值运算一遍，并非更新
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            #self.ai[i] = S_fy(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh): #对隐含层的每一个神经元
            sum = 0.0 #这个神经元初始化为0
            for i in range(self.ni): #接受前一层所有的神经元信息
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = S_fy(sum) #S化后存入

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = S_fy(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no): #每一个输出
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, patterns):
        FOutput = open('x1x2_Output.txt', 'w')
        for p in patterns:
            temp = self.update(p[0])
            print(p[0], "->", temp) #update的参数是inputs
            for item in p[0]:
                FOutput.write(str(item)+' '),
            for item in temp:
                FOutput.write(str(item)+' '),
            FOutput.write('\n')
        FOutput.close()

    def weights(self):
        print('Input weights: '),
        for i in range(self.ni):
            print(self.wi[i] ),
        print
        print('Output weights: '),
        for j in range(self.nh):
            print(self.wo[j] ),
        print

    def train(self, patterns, iterations=100000, N=0.001, M=0.001):
        # N: learning rate
        # M: momentum factor
        # change = hidden_deltas[j]*self.ai[i]
        # self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
        # self.ci[i][j] = change

        XLErrorList = []
        olderror = 0

        for i in range(iterations): #训练次数
            error = 0.0 #本次误差
            for p in patterns: #数据中每行
                inputs = p[0] #每行的第一个数据是一个输入数组
                targets = p[1] #每行的后一个数据是期望输出
                self.update(inputs) #return self.ao[:]
                error = error + self.backPropagate(targets, N, M) #这次训练的累加误差
            if i % 80 == 0: #每???0次训练打印一次误差
                print('error=%-.9f' % error),
                XLErrorList.append(error)
                if i % 640 == 0:
                    print('    [%3.2f %%]    delta=%3.9f\n' % ((i*1.0/iterations)*100, abs(error-olderror)))
            olderror = error
        print('\n')

        XLErrorOutput = open('x1x2-XLError.txt', 'w')
        for item in XLErrorList:
            XLErrorOutput.write(str(item)+' '),
        XLErrorOutput.close()


def demo():

    TestList2 = []
    TestFileList = ['sinx_InputData.txt', '1sinx1_InputData.txt', 'x1x2_InputData.txt']
    TestFileXL = [9, 9, 121]
    FileNum = 2
    TestFile = TestFileList[FileNum]  #确保数据在这个文件
    f = open(TestFile,'r')
    for line in f: #对于每一行
        TestList = [float(x) for x in line.split()] #读取这行的每一个实数，形成一行数据
        TestList2.append([TestList]) #形成2维数组

    traindata = []
    for i in range(len(TestList2)):
        tempLa = TestList2[i]
        tempLb = tempLa[0]
        tempLc = tempLb[len(tempLb)-1]
        tempLb = [tempLb[:len(tempLb)-1]]
        tempLb.append([tempLc])
        traindata.append(tempLb)
    #
    #
    datalen = len(traindata[0][0]) #输入层的数量
    #
    # n = NN(datalen, datalen+24, 1)
    #
    # n.train(traindata[0:TestFileXL[FileNum]]) #def train(self, patterns, iterations=500, N=0.02, M=0.01):
    # #n.train(traindata[:])
    # n.test(traindata[TestFileXL[FileNum]:])

    example_list=traindata[:TestFileXL[FileNum]]
    bpnn = BPNN.Bpnn(datalen, [datalen+16, datalen+16, 1])
    bpnn.train(example_list, 0.165, 0.1, 0.1)
    #bpnn.debug_train(example_list,2000)

    FOutput = open('x1x2_Output.txt', 'w')
    for line in traindata[TestFileXL[FileNum]:]:
        bpnn.compute(line[0])
        print(line[0][0], line[0][1],' -> ', bpnn.output()[0])
        for item in line[0]:
            FOutput.write(str(item)+' '),
        for item in bpnn.output():
            FOutput.write(str(item)+' \n')
    FOutput.close()



if __name__ == '__main__':
    demo()


# clear;
# file_t = fopen('D:\!zju\!IntelligentSystem\HW#4\x1x2_Output.txt','r');
# [x fx] = fscanf(file_t,'%f %f');
# for i=1:3:fx
# x1((i+2)/3)=x(i,1);
# x2((i+2)/3)=x(i+1,1);
# yy((i+2)/3)=x(i+2,1);
# end
# fclose(file_t);
# x1=reshape(x1,21,21);
# x2=reshape(x2,21,21);
# yy=reshape(yy,21,21);
#
# for i=1:21
# for j=1:21
# if x1(i,j)==0 temp1=1;
# else temp1=sin(x1(i,j))/x1(i,j);end
# if x2(i,j)==0 temp2=1;
# else temp2=sin(x2(i,j))/x2(i,j);end
# YY(i,j)=temp1*temp2;
# end
# end
# %mesh(x1,x2,YY); %理论图
# mesh(x1,x2,yy); %训练图