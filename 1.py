#Encoding and Reconstruction of PCM
import numpy as np
import matplotlib.pyplot as plt


f=2
fs=16
ts=1/fs
t=np.arange(0,1,ts)
x_t=255*(1+np.sin(2*np.pi*f*t))
#check input
plt.stem(t,x_t)
plt.show

#quantization
u_256=[np.round(i) for i in x_t]
u_128=[np.round(i/2)*2 for i in x_t]
u_64=[np.round(i/4)*4 for i in x_t]
u_32=[np.round(i/8)*8 for i in x_t]
u_16=[np.round(i/16)*16 for i in x_t]


plt.plot(u_256)
plt.plot(u_128)
plt.plot(u_64)
plt.plot(u_32)
plt.plot(u_16)


def quantize(x_t,l):
    step=int(256/l)
    v=[int(np.round(i/step)*step) for i in x_t]
    return v
v=quantize(x_t,16)
print(v)

def encd(v,l):
    r=np.log2(l)
    t=[]
    for i in v:
        b=[int(j) for j in np.binary_repr(i)]
        t=np.append(t,b)
    return t
t=encd(v,16)
plt.stem(t)






Px=0.5*(255/2)**2
#Pn=(1/len(x_t))*np.sum(np.square(xq-x_t))


SNR=np.zeros(5)


xq_256=quantize(x_t,256)
xq_128=quantize(x_t,128)
xq_64=quantize(x_t,64)
xq_32=quantize(x_t,32)
xq_16=quantize(x_t,16)


Pn=[
    (1/len(x_t))*np.sum(np.square(xq_16-x_t)),
    (1/len(x_t))*np.sum(np.square(xq_32-x_t)),
    (1/len(x_t))*np.sum(np.square(xq_64-x_t)),
    (1/len(x_t))*np.sum(np.square(xq_128-x_t)),
    (1/len(x_t))*np.sum(np.square(xq_256-x_t))
]


for i in np.arange(5):
    SNR[i]=10*np.log10(Px/Pn[i])
    
R=np.array([4,5,6,7,8])




plt.figure(figsize=(12,8))
plt.plot(R,SNR)
plt.grid()
plt.xlabel('number of bits of encoder')
plt.ylabel('SNR')
plt.title('PCM:SNR vs number of bits')
plt.show()
