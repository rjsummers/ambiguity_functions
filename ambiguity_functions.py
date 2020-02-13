import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns


def ambgfun(x, fs, prf):
    """Calculate the ambiguity function of a radar waveform.

    This function calculates the monostatic ambiguity function of the radar
    waveform x, using FFTs to calculate the matched filter output.
    """
    ts = 1 / fs   # Sampling interval (seconds)

    nx = np.size(x)
    delay = np.arange(1-nx, nx) * ts
    n = np.size(delay)
    nfft = int(2**(np.ceil(np.log2(n))))
    m = nfft
    dshifts = np.arange(-m/2, m/2, dtype=int)
    doppler = dshifts * (fs / nfft)

    X0 = np.fft.fft(x, n=nfft)
    X = np.zeros((m, nfft), dtype=complex)
    for i in range(1, m):
        X[i, :] = np.roll(X0, dshifts[i])
    H = np.fft.fft(np.conj(np.flip(x)), n=nfft)
    # X = np.fft.fft(x[None, :] * np.exp(1j * 2 * np.pi *
    #                                    doppler[..., None] * t[None, :]),
    #                n=n, axis=1)
    # H = np.fft.fft(np.conj(np.flip(x)), n=n)
    ambig = np.fft.ifft(X * H[None, :], axis=1)
    ambig = ambig[:, 0:n]
    return ambig, delay, doppler


def ambgfun2(x, fs, prf):
    """Calculate the ambiguity function of a radar waveform.

    This function calculates the monostatic ambiguity function of the radar
    waveform x by directly calculating the correlation of the waveform with a
    doppler shifted copy of itself. This function is computationally
    inefficient compared to ambgfun2, which uses FFTs to perform the
    correlation, and should be preferred.
    """
    ts = 1 / fs  # Sampling interval (seconds)

    nx = np.size(x)
    t = np.arange(0, nx) / fs
    delay = np.arange(1-nx, nx) * ts
    n = np.size(delay)

    m = int(2**(np.ceil(np.log2(n))))
    doppler = np.arange(-m/2, m/2) * prf

    ambig = np.zeros((m, n), dtype=complex)
    for i in range(0, m):
        fd = doppler[i]
        dshift = np.exp(1j * 2 * np.pi * fd * t)
        ambig[i, :] = np.correlate(x * dshift, x, mode="full")
    return ambig, delay, doppler


sns.set()
sns.set_context("paper")

##################
# Pulse waveform #
##################

fs = 6.4e6
ts = 1/fs
tx = 1.0000e-05
prf = 10000

t = np.arange(0, tx, step=ts)

x = np.ones(np.shape(t))
ambig, delay, doppler = ambgfun(x, fs, prf)

plt.figure()
plt.pcolormesh(delay / 1e-6, doppler / 1e3, np.abs(ambig))
plt.xlabel("Delay (μs)")
plt.ylabel("Doppler (kHz)")
plt.savefig("images/pulse_ambigfun.png")

plt.figure()
plt.contour(delay / 1e-6, doppler / 1e3, np.abs(ambig))
plt.xlabel("Delay (μs)")
plt.ylabel("Doppler (kHz)")
plt.savefig("images/pulse_ambigfun_contour.png")

fig = plt.figure()
ax = fig.gca(projection='3d')
DEL, DOP = np.meshgrid(delay, doppler)
ax.plot_surface(DEL, DOP, np.abs(ambig), cmap=cm.inferno,
                linewidth=0.1)
plt.savefig("images/pulse_ambigfun_surf.png")

#########################
# Linear chirp waveform #
#########################

fs = 8e6
bw = 1e6
ts = 1/fs
tx = 15/bw
prf = 10e3

t = np.arange(0, tx, step=ts)

chirpyness = bw / tx
theta = - np.pi * bw * t + np.pi * chirpyness * t**2
x = np.exp(1j * theta)
ambig, delay, doppler = ambgfun(x, fs, prf)

plt.figure()
plt.pcolormesh(delay / 1e-6, doppler / 1e3, np.abs(ambig))
plt.xlabel("Delay (μs)")
plt.ylabel("Doppler (kHz)")
plt.savefig("images/lfm_ambigfun.png")

plt.figure()
plt.contour(delay / 1e-6, doppler / 1e3, np.abs(ambig))
plt.xlabel("Delay (μs)")
plt.ylabel("Doppler (kHz)")
plt.savefig("images/lfm_ambigfun_contour.png")

fig = plt.figure()
ax = fig.gca(projection='3d')
DEL, DOP = np.meshgrid(delay, doppler)
ax.plot_surface(DEL, DOP, np.abs(ambig), cmap=cm.inferno,
                linewidth=0.1)
plt.savefig("images/lfm_ambigfun_surf.png")

#########################
# Barker-coded waveform #
#########################

# code = [1, 1, 1, -1, 1]
code = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
x = np.zeros((117,))
for i in range(np.size(x)):
    x[i] = code[i // 9]
fs = 8e6
ambig, delay, doppler = ambgfun(x, fs, prf)

plt.figure()
plt.plot(x)
plt.savefig("images/barker_waveform.png")

plt.figure()
plt.pcolormesh(delay / 1e-6, doppler / 1e3, np.abs(ambig))
plt.xlabel("Delay (μs)")
plt.ylabel("Doppler (kHz)")
plt.savefig("images/barker_ambigfun.png")

plt.figure()
plt.contour(delay / 1e-6, doppler / 1e3, np.abs(ambig))
plt.xlabel("Delay (μs)")
plt.ylabel("Doppler (kHz)")
plt.savefig("images/barker_ambigfun_contour.png")

fig = plt.figure()
ax = fig.gca(projection='3d')
DEL, DOP = np.meshgrid(delay, doppler)
ax.plot_surface(DEL, DOP, np.abs(ambig), cmap=cm.inferno,
                linewidth=0.1)
plt.savefig("images/barker_ambigfun_surf.png")
