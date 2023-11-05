from pylab import *
import matplotlib.pyplot as plt

names = ['0', '0.1', '0.2', '0.3', '0.4','0.5','0.6','0.7','0.8','0.9','1','1.1','1.2','1.3','1.4','1.5']
x = range(len(names))
y1 = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]
y = [0.9595,0.9303,0.9042,0.8794,0.8521,0.8255,0.8013,0.7733,0.7467,0.7194,0.6949,0.6729,0.6511,0.6305,0.6111,0.5941]
y2 = [0.789,0.789,0.789,0.789,0.789,0.789,0.789,0.789,0.789,0.789,0.789,0.789,0.789,0.789,0.789,0.789]


def f_mea(a,b):
    fmeasure = (2*a*b)/(a+b)
    return fmeasure
z = []
for i in range(len(y)):
    f = f_mea(y1[i],y[i])
    z.append(f)
print(z)

# y = [0.2786,0.09526,0.03631,0.01232,0.00291]

# y1=[0.452,0.1272,0.03737902,0.01120912,0.00281906]
plt.plot(x, y, marker='o', mec='r', mfc='w',label=u'th-srcc')
plt.plot(x, z, marker='*', ms=10,label=u'fmeasure')
plt.plot(x, y2, marker='*', ms=10,label=u'mymethod')

plt.legend()  # 让图例生效
plt.xticks(x, names, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"th") #X轴标签
plt.ylabel("srcc") #Y轴标签
plt.title("th_srcc_feasure_curve") #标题
plt.savefig('/home/w509/1workspace/lee/360_fix_sort/gtprove/curve_th_srcc/unisal_curve_th_srcc.jpg')
plt.show()