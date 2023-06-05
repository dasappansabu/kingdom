import numpy as np
import matplotlib.pyplot as plt
#Generate binary sequence of length N
def generate_binary_sequence(N):
symbols=[0, 1] #define possible symbols
b=np.random.choice(symbols,size=N) #randomly choose N bits from the symbols list
return b
#Perform QPSK modulation on input binary sequence b
def qpsk_modulation(b):
#Separate binary sequence into I and Q (in-phase and quadrature phase) components
bI=b[::2]
bQ=b[1::2]
#Map I and Q components to -1 and 1 using 2*bI-1 and 2*bQ-1
sI=2*bI-1
sQ=2*bQ-1
#Combine I and Q components into a complex-valued signal
s=(sI+1j*sQ)/np.sqrt(2)
return s
#Add AWGN channel noise to input signal s with N0 noise power
def awgn_channel(s,N0):
#Generate complex-valued noise samples with variance N0/2 for both I and Q components
wI=np.sqrt(N0/2)*np.random.randn(len(s))
wQ=np.sqrt(N0/2)*np.random.randn(len(s))
#Combine I and Q components of noise to form a complex-valued noise signal
w=(wI+1j*wQ)/np.sqrt(2)
#Add noise to input signal s
x=s+w
return x
#Perform QPSK detection on input signal x
def qpsk_detection(x):
#Perform hard decision on I and Q components separately
bIhat=(x.real>=0)
bQhat=(x.imag>=0)
#Combine detected I and Q bits into a single binary sequence
bhat=np.zeros(len(x) *2,dtype=int)
bhat[::2]=bIhat
bhat[1::2]=bQhat
return bhat
#Calculate bit error rate (BER) between input binary sequence b and detected binary sequence bhat
def calculate_BER(b,bhat):
#Count the number of bit errors and divide by the total number of bits
return np.sum(bhat!=b)/len(b)

#Calculate symbol error rate (SER) between input binary sequences bI, bQ and detected
binary sequences bIhat, bQhat
def calculate_SER(bI,bQ,bIhat,bQhat):
#Count the number of symbol errors and divide by the total number of symbols
return np.sum((bIhat!=bI)|(bQhat!=bQ))/len(bI)
#Simulate QPSK transmission with varying Eb/N0 values and return BER and SER
def simulate_QPSK_transmission(EbN0dB, N):
#determine the number of Eb/N0 values to simulate
L=len(EbN0dB)
#initialize arrays to store BER and SER values for each Eb/N0 value
BER=np.zeros(L)
SER=np.zeros(L)
#Loop over all Eb/N0 values
for i in range(L):
#generate binary sequence of length N
b=generate_binary_sequence(N)
#modulate binary sequence using QPSK
s=qpsk_modulation(b)
#determine the noise power
N0=10 **(-0.1* EbN0dB[i])
#add Gaussian noise to modulated signal
x=awgn_channel(s,N0)
#detect the received signal to obtain the estimated binary sequence
bhat=qpsk_detection(x)
#extract the in-phase and quadrature components of the original and estimated binary sequences
bI=b[::2]
bQ=b[1::2]
bIhat=bhat[::2]
bQhat=bhat[1::2]
#calculate the BER and SER for the current Eb/N0 value
BER[i]=calculate_BER(b,bhat)
SER[i]=calculate_SER(bI,bQ,bIhat,bQhat)
#Return the BER and SER arrays
return BER, SER
#Set the number of bits in the binary sequence to be twice the number of symbols
N=2*50000
#Define the range of Eb/N0 values to simulate
EbN0dB=np.arange(-10,10.5,0.5)
#Simulate QPSK transmission with varying Eb/N0 values and plot BER and SER versus Eb/N0
BER,SER=simulate_QPSK_transmission(EbN0dB,N)
#Plot BER versus Eb/N0
plt.plot(EbN0dB,BER)
plt.xlabel('Eb/N0 in dB')
plt.ylabel('BER')
plt.grid()
plt.title('BER versus SNR per bit for QPSK')
plt.show()
#Plot SER versus Eb/N0
plt.plot(EbN0dB,SER)
plt.xlabel('Eb/N0 in dB')
plt.ylabel('SER')
plt.grid()
plt.title('SER versus SNR per bit for QPSK')
plt.show()
