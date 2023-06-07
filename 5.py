#exp 5
import numpy as np
import matplotlib.pyplot as plt


N = 2 * 50000
EbN0dB = np.arange(-10, 10.5, 0.5)
L = len(EbN0dB)
BER = np.zeros(L)
SER = np.zeros(L)


for i in range(L):
    b = np.zeros(N)
    bhat = np.zeros(N)


    # Random binary sequence
    b = np.random.randint(0,2,N)
    bI = b[::2]
    bQ = b[1::2]


    N0 = 10 ** (-0.1 * EbN0dB[i])
    wI = np.sqrt(N0 / 2) * np.random.randn(N // 2)
    wQ = np.sqrt(N0 / 2) * np.random.randn(N // 2)


    # QPSK modulation
    sI = 2 * bI - 1
    sQ = 2 * bQ - 1


    # AWGN channel
    xI = sI + wI
    xQ = sQ + wQ


    # QPSK detection
    bIhat = (xI >= 0)
    bQhat = (xQ >= 0)
    bhat[::2] = bIhat
    bhat[1::2] = bQhat


    BER[i] = np.sum(bhat != b) / N
    SER[i] = np.sum((bIhat != bI) | (bQhat != bQ)) / (N//2)


plt.plot(EbN0dB, BER)
plt.xlabel('Eb/N0 in dB')
plt.ylabel('BER')
plt.grid(True)
plt.title('BER versus SNR per bit for QPSK')
plt.show()


plt.plot(EbN0dB, SER)
plt.xlabel('Eb/N0 in dB')
plt.ylabel('SER')
plt.grid(True)
plt.title('SER versus SNR per bit for QPSK')
plt.show()
