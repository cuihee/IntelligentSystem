__author__ = 'cuihe'
# coding:utf-8

import math

def WriteFile(File, DisX=9, startX=0.0, endX=2*math.pi, OpenStlye='w'):
    PerX = (endX-startX)/(DisX-1)
    print('PerX=%f' %PerX )
    f = open(File,OpenStlye)
    for i in range(DisX):
        temp = i*PerX
        f.write("%.6f %.6f\n" %(temp, math.sin(temp)))
        print('sin(temp)=%f' %math.sin(temp))
    f.close()

def MakeFx_sinx():
    FileName = 'sinx_InputData.txt'
    WriteFile(FileName, 9, 0.0, 2*math.pi, 'w')
    WriteFile(FileName, 361, 0.0, 2*math.pi, 'a')


if __name__ == '__main__':
    MakeFx_sinx()
