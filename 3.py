#exp 3
import numpy as np
import matplotlib.pyplot as plt
# Generate N = 10^5 samples of M-PAM signal taking values in {-3, -1, 1, 3} for M = 4
N = 10**5
M_values = [2, 4]
beta_values = [0.3, 0.5, 0.8]
symbols = np.array([-3,-1, 1,3])
EbN0dB = 20
for M in M_values:
    d = np.random.choice(symbols[0:M], size=N)
    for beta in beta_values:
        # Upsample the samples by a factor of L = 8
        L = 8
        v = np.zeros(N*L)
        v[::L] = d
        # Generate the square-root raised cosine pulse p(t)
        Ts = 1 # symbol interval
        Nsym = 8 # number of symbols
        t = np.linspace(-Nsym/2, Nsym/2, Nsym*L+1) # time base
        p = np.zeros(len(t))
        for i in range(len(t)):
            if t[i] == 0:
                p[i] = (1 - beta + 4*beta/np.pi)
            elif abs(t[i]) == Ts/(4*beta):
                p[i] =(beta/np.sqrt(2))*((1+2/np.pi)* np.sin(np.pi/(4*beta)) +(1-2/np.pi)*np.cos(np.pi/(4*beta)))
            else:
                p[i] = (np.sin(np.pi*t[i]/Ts*(1-beta))+ 4*beta*t[i]/Ts*np.cos(np.pi*t[i]/Ts*(1+beta)))/(np.pi*t[i]/Ts*(1-(4*beta*t[i]/Ts)**2))
        p = p[np.nonzero(p)]
        # Normalize pulse shape energy to one
        k = np.sqrt(np.sum(p**2))
        p = p / k
        SER_list = []
        # Convolve the upsampled input stream v with the shaping pulse
        D = int(Nsym * L / 2)
        s = np.convolve(v, p)
        offset_range=np.arange(-4,5)
        
        for i in offset_range:
            
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
            v_hat = s_hat[2*D+i:len(s_hat)-2*D+i:L]
            # Demodulate v_hat to the nearest 4-PAM point to obtain decoded stream
            dist = np.abs(v_hat.reshape(-1,1) - symbols.reshape(1,-1))
            dec_idx = np.argmin(dist, axis=1)
            d_hat = symbols[dec_idx]

            # Compute symbol error rate
            SER = np.sum(d != d_hat) / len(d)
            SER_list.append(SER)

            # Plot eye diagram
        def plotEyeDiagram(s, L, offset, t,x):
            traces = np.reshape(s[offset:t*L - offset], (-1,L)).transpose()
            plt.plot(traces)
            plt.xlabel('Time [samples]')
            plt.ylabel('Amplitude')
            plt.title(f'Eye Diagram {x} for M={M}, β={beta}, EbN0dB={EbN0dB}dB')
            plt.show()
        # Plot eye diagrams with noise
        plotEyeDiagram(s_hat, 2*L, offset=int(Nsym*(L)), t=100,x='with noise')
        # Plot eye diagrams without noise
        plotEyeDiagram(s, 2*L, offset=int(Nsym*(L)), t=100,x='without noise')

        # Plot SER vs offset
        plt.plot(offset_range, SER_list, 'bo-', linewidth=2, markersize=8)
        plt.grid(True)
        plt.title(f'SER vs Offset for M={M}, β={beta}, EbN0dB={EbN0dB}dB')
        plt.xlabel('offset')
        plt.ylabel('Symbol Error Rate (SER)')
        plt.show()
        # Find the interval in which SER vs offset plot has lowest value
        min_SER_indices = np.where(SER_list == np.min(SER_list))[0]
        best_intervals = offset_range[min_SER_indices]

        print(f"For M = {M}, beta = {beta}, EbN0dB = {EbN0dB}:")
        print(f"Possible time intervals for sampling received signal:{offset_range}")
        print(f"SER values for each time interval: {SER_list}")
        print(f"The best time intervals for sampling the received signal:{best_intervals}")
