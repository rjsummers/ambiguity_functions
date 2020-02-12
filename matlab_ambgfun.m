clear; close all; clc;

%% Rectangular Waveform
fs = 6.4e6;
ts = 1/fs;
tx = 1.0000e-05;
prf = 1/tx;
waveform = phased.RectangularWaveform('SampleRate',fs,...
    'PRF',prf,'PulseWidth',tx);
x = waveform();
prf = waveform.PRF;
fs = waveform.SampleRate;
[afmag,delay,doppler] = ambgfun(x,waveform.SampleRate,prf);

figure()
imagesc(delay,doppler,afmag)
xlabel('Delay (seconds)')
ylabel('Doppler Shift (hertz)')

%% LFM Waveform
fs = 8e6;
bw = 1e6;
ts = 1/fs;
tx = 15/bw;
prf = 1/(tx + 1/fs);
waveform = phased.LinearFMWaveform('SampleRate',fs,...
    'SweepBandwidth',bw,'PRF',prf,'PulseWidth',tx);
x = waveform();
prf = waveform.PRF;
fs = waveform.SampleRate;
[afmag,delay,doppler] = ambgfun(x,waveform.SampleRate,prf);

figure()
imagesc(delay,doppler,afmag)
xlabel('Delay (seconds)')
ylabel('Doppler Shift (hertz)')

%% Barker Waveform
