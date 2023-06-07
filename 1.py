#exp 1
import numpy as np
import matplotlib.pyplot as plt

f=2.1
fs=16
t=np.arange(0,1,1/fs)
x=(255/2)*(1+np.sin(2*np.pi*f*t))

def quantize(sample,L):
    if L==256:
        return int(round(sample))
    elif L==128:
        return int(round(sample/2)*2)
    elif L==64:
        return int(round(sample/4)*4)
    elif L==32:
        return int(round(sample/8)*8)
    elif L==16:
        return int(round(sample/16)*16)
    elif L==8:
        return int(round(sample/32)*32)
    elif L==4:
        return int(round(sample/64)*64)
    elif L==2:
        return int(round(sample/128)*128)
    
def encode(sample,L):
    R=int(np.log2(L))
    return bin(quantize(sample,L))[2:].zfill(R)
def decode(code,L):
    R=int(np.log2(L))
    return quantize(int(code,2),L)

L_values=[256,128,64,32,16,8,4,2]
quantised_samples={}
encoded_samples={}

for L in L_values:
    quantised_samples[L]=np.array([quantize(sample,L) for sample in x])
    encoded_samples[L]=np.array([encode(sample,L) for sample in x])


#bit stream
encoded_concatenated = np.concatenate([encoded_samples[256]])
# Convert bit stream to binary string
bit_stream = ''.join(encoded_concatenated)

plt.stem(range(len(bit_stream)), [int(bit) for bit in bit_stream])
plt.xlim([0, len(bit_stream)])
plt.ylim([-0.1, 1.1])
plt.xlabel('Bit Index')
plt.ylabel('Bit Value')
plt.title('Bit Stream')
plt.show()



#t1=range(len(bitstream))plt.ylim(-0.1,1.1)plt.stem(t1,bitstream)plt.title('Encoded bit stream')plt.show()

decoded_samples={}
reconstructed_signals={}
for L in L_values:
    decoded_samples[L]=np.array([decode(code,L) for code in encoded_samples[L]])
    reconstructed_signals[L]=decoded_samples[L]/(L/255)
    
plt.plot(t,x,label="original signal")

plt.plot(t,reconstructed_signals[256],label="reconstructed_signals")
plt.legend()
plt.tight_layout()

#snr
px=0.5*(255/2)**2
SNR={}
for L in L_values:
    pn=np.sum(((quantised_samples[L]-x)**2)/len(x))
    SNR[L]=10*np.log10(px/pn)

R_values=[int(np.log2(L))for L in L_values] 
SNR_values=[SNR[L] for L in L_values]
plt.figure()
plt.plot(R_values,SNR_values,'-o')
plt.title('SNR VS no of bits')
plt.grid()
plt.show()


plt.scatter(reconstructed_signals[256],reconstructed_signals[256],label="reconstructed_signals")
plt.show()
