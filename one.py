import numpy as np
import matplotlib.pyplot as plt
f = 2
fs = 16
ts = 1 / fs
t = np.arange(0,1+ts,ts)
x = 255 * (1 + np.sin(2*np.pi*f*t))/2
def quantize(x, levels):
step = int(256/levels)
xq = [int(np.floor(i / step)*step) for i in x]
for i in np.arange(len(xq)):
if xq[i] == 256:
xq[i] = 255
return xq
def encode(xq,levels):
R = int(np.log2(levels))
step = 256 / levels
xe = [bin(int(i / step))[2:].zfill(R) for i in xq]
bs = []
for i in xe:
b = [int(j) for j in i]
bs += b
return [xe,bs]
levels = 32
R = np.log2(levels)
xq = quantize(x,levels)
xe,bs = encode(xq,levels)
d = int(256 / levels)
val = np.arange(0,257,d)
plt.figure(figsize=(10,7))
plt.plot(x)
plt.plot(xq)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('PCM:Original and Reconstructed Signals')
plt.legend(['Original Signal','Reconstructed Signal'])
plt.grid()
plt.savefig('Original_and_Reconstructed_Signals.jpg')
plt.show()
plt.title('PCM:Encoded bit stream')
plt.ylabel('Encoded Bit Stream')
plt.xlabel('Time')
plt.stem(bs)
plt.savefig('Encoded_bit_stream.jpg')
plt.show()
l = [8,16,32,64,128,256]
Px = 0.5 * (255 / 2) ** 2
Pn = np.zeros_like(l)
Pn = [(1 / len(x)) * np.sum(np.square(quantize(x, l[i]) - x)) for i in np.arange(len(l))
SNR = np.zeros_like(l)
R = np.log2(l)
for i in np.arange(len(l)):
SNR[i] = 10 * np.log10(Px/Pn[i])
plt.plot(R,SNR)
plt.scatter(R,SNR)
plt.grid()
plt.xlabel('Number of bits of encoder')
plt.ylabel('SNR')
plt.title('PCM: SNR vs number of bits')
plt.savefig('SNR_versus_Number_of_bits.jpg')
