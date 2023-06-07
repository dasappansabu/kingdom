#exp 4
import numpy as np
import matplotlib.pyplot as plt
N = 100000
EbN0dB = np.arange(-10, 10.5, 0.5)
L = len(EbN0dB)
BER = np.zeros(L)
Eb = 1
for i in range(L):
    # Random binary sequence
    b = np.random.randint(0, 2, N)
    N0 = 10**(-0.1*EbN0dB[i])
    w = np.sqrt(N0/2) * np.random.randn(N)

    # BPSK modulation
    s = np.where(b, np.sqrt(Eb), -np.sqrt(Eb))
    # AWGN channel
    x = s+w
    # BPSK detection
    bhat = (x >= 0)

    BER[i] = np.sum(bhat != b) / N

plt.plot(EbN0dB, BER)
plt.xlabel('Eb/N0 in dB')
plt.ylabel('BER')
plt.grid(True)
plt.title('BER versus SNR per bit for BPSK')
plt.show()
