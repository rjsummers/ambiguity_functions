fs = 1e6;
bw = 5e5;
prf = 5e3;
tx = 5e-5;
% waveform = phased.RectangularWaveform('SampleRate',fs,...
%     'PRF',prf,'PulseWidth',tx);
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