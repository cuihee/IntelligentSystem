__author__ = 'cuihe'
# coding:utf-8

import math

def WriteFile(File, DisX=11, startX=-10.0, endX=10.0, OpenStlye='w'):
    PerX = (endX-startX)/(DisX-1)
    f = open(File,OpenStlye)
    for i in range(DisX):
        for j in range(DisX):
            X1 = startX+i*PerX
            if X1==0:
                ZX1=1
            else:
                ZX1 = math.sin(X1)/X1
            X2 = startX+j*PerX
            if X2==0:
                YX2=1
            else:
                YX2 = math.sin(X2)/X2
            resultZ_Y = ZX1 * YX2
            f.write("%.6f %.6f %.6f \n" %(X1, X2, resultZ_Y) )#
    f.close()


def MakeFx_x1x2():
    FileName = 'x1x2_InputData.txt'
    WriteFile(FileName, 11, -10.0, 10.0, 'w')
    WriteFile(FileName, 21, -10.0, 10.0, 'a')


if __name__ == '__main__':
    MakeFx_x1x2()
