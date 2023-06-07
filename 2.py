#exp 2
import numpy as np
import matplotlib.pyplot as plt

#Generate N = 10^5 samples of 4-PAM (M = 4)signal taking values in {−3, −1, 1, 3}
N = 10**5
M = 4
symbols = np.array([-3, -1, 1, 3])
d = np.random.choice(symbols, size=N)
EbN0dB_range = np.arange(-4, 14, 2)

# Upsample the samples by a factor of L = 8
L = 8
v = np.zeros(N*L)
v[::L] = d

# Generate the square-root raised cosine pulse p(t)
Ts = 1                                    # symbol interval
L = 8                                     # upsampling factor
beta = 0.3                                # roll-off factor
Nsym = 8                                  # number of symbols
t = np.linspace(-Nsym/2, Nsym/2, Nsym*L+1)          # time base
p = np.zeros(len(t))

for i in range(len(t)):
    if t[i] == 0:
        p[i] = (1 - beta + 4*beta/np.pi)
    elif abs(t[i]) == Ts/(4*beta):
        p[i] = (beta/np.sqrt(2))*((1+2/np.pi)*np.sin(np.pi/(4*beta))+ (1-2/np.pi)*np.cos(np.pi/(4*beta)))
    else:
        p[i] = (np.sin(np.pi*t[i]/Ts*(1-beta))+ 4*beta*t[i]/Ts*np.cos(np.pi*t[i]/Ts*(1+beta)))/(np.pi*t[i]/Ts*(1-(4*beta*t[i]/Ts)**2))

# Normalize pulse shape energy to one
k=np.sqrt(np.sum(p**2))
p = p / k
plt.plot(t,p)
plt.title('Root raised cosine pulse')
plt.xlabel('t')
plt.ylabel('p(t)')
plt.show()
SER_list = []
# Convolve the upsampled input stream v with the shaping pulse p
D = int(Nsym * L / 2)
s = np.convolve(v, p)
for EbN0dB in EbN0dB_range:
    # Add Gaussian noise to s to simulate transmission via a baseband AWGN channel
    EsN0dB = 10 * np.log10(np.log2(M)) + EbN0dB
    P = L * np.sum(np.abs(s)**2) / len(s)
    N0_2 = P / (2 * 10**(EsN0dB/10))
    n = np.random.normal(0, np.sqrt(N0_2), len(s))
    r = s + n
    
    # Matched filter at the receiver
    h = p[::-1]
    s_hat = np.convolve(r, h)
    # Downsampled match filter output
    v_hat = s_hat[2*D+1:len(s_hat)-2*D:L]

    # Demodulate v_hat to the nearest 4-PAM point to obtain decoded stream
    dist = np.abs(v_hat.reshape(-1,1) - symbols.reshape(1,-1))
    dec_idx = np.argmin(dist, axis=1)
    d_hat = symbols[dec_idx]

    # Compute symbol error rate
    SER = np.sum(d != d_hat) / len(d)
    SER_list.append(SER)

# Plotting

plt.stem(d[:15])
plt.title("Message Symbols")
plt.xlabel('n')
plt.ylabel('d')
plt.show()

plt.plot(s[:50*L])
plt.title("Pulse-Shaped Signal transmitted")
plt.xlabel('t')
plt.ylabel('s(t)')
plt.show()

plt.plot(r[:50*L])
plt.title("Received Signal")
plt.xlabel('t')
plt.ylabel('r(t)')
plt.show()

plt.plot(s_hat[:50*L])
plt.title("Output of Matched Filter")
plt.xlabel('t')
plt.ylabel('s_hat(t)')
plt.show()

plt.stem(v_hat[:15])
plt.title("Downsampled version of matched filter output")
plt.xlabel('n')
plt.ylabel('v_hat')
plt.show()

plt.stem(d_hat[:15])
plt.title("Decoded Symbols ")
plt.xlabel('n')
plt.ylabel('d_hat')
plt.show()

# Plot SER vs Eb/N0
plt.plot(EbN0dB_range, SER_list)
plt.title("SER versus SNR per bit")
plt.grid(True)
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Symbol Error Rate (SER)')
