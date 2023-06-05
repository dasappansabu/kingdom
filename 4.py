#Expt. 4: Binary Phase Shift Keying
import numpy as np
import matplotlib.pyplot as plt
#Generate a binary sequence of length N
def generate_binary_sequence(N):
symbols=[0,1] #define possible symbols
b=np.random.choice(symbols,size=N) #randomly choose N bits from the symbols list
return b
#Perform BPSK modulation on a binary sequence b
def bpsk_modulation(b):
s=2*b-1 #map 0 to -1 and 1 to 1
return s
#Add AWGN to the signal s with power spectral density N0/2
def awgn_channel(s,N0):
w=np.sqrt(N0/2)*np.random.randn(len(s)) #generate white Gaussian noise with variance N0/2
x=s+w #add noise to the signal
return x
#Perform BPSK detection on the received signal x
def bpsk_detection(x):
return (x>=0) #if the received signal is greater than or equal to 0, decode as 1, else decode as 0
#Calculate the bit error rate(BER) between the original binary sequence b and the detected sequence bhat
def calculate_BER(b,bhat):
return np.sum(bhat!=b)/len(b) #count the number of bits that are different between the two sequences

and divide by the total number of bits
#Simulate BPSK transmission over an AWGN channel with varying Eb/N0
def simulate_BPSK_transmission(EbN0dB,N):
L=len(EbN0dB) #determine the number of SNR values to simulate
BER=np.zeros(L) #initialize the BER array to store the results
#iterate over each SNR value
for i in range(L):
#generate a random binary sequence of length N
b=generate_binary_sequence(N)
#perform BPSK modulation on the binary sequence
s=bpsk_modulation(b)
#calculate the noise power spectral density from the given Eb/N0 value
N0=10**(-0.1*EbN0dB[i])
#add AWGN to the modulated signal with the determined noise power
x=awgn_channel(s,N0)
#perform BPSK detection on the received signal
bhat=bpsk_detection(x)
#calculate the BER between the original binary sequence and the detected sequence
BER[i]=calculate_BER(b,bhat)
#return the BER results
return BER

N=10**5 #set the length of the binary sequence to simulate
EbN0dB=np.arange(-10,10.5,0.5) #define the range of Eb/N0 values to simulate
BER=simulate_BPSK_transmission(EbN0dB,N) #simulate BPSK transmission over an AWGN channel

with varying Eb/N0

#Plot the BER results as a function of Eb/N0
plt.plot(EbN0dB, BER)
plt.xlabel('Eb/N0 in dB')
plt.ylabel('BER')
plt.title('BER versus SNR per bit for BPSK')
plt.grid()
plt.show()
