%P5  Repetition of p4.m via FFT
%    For complex v, delete "real" commands.
close all;
scrsz = get(groot,'ScreenSize');
figure('position', [50   50   4/5*scrsz(3)  4/5*scrsz(4)]); 

% Differentiation of a hat function
%% MOST INTERESTING CODE #1.
N = 24; h = 2*pi/N; x = h*(1:N)';
v = max(0, 1-abs(x-pi)/2); 
vhat = fft(v);
% Wavenumber.
k = (-N/2+1 : N/2)';

kk = [0:N/2-1    0 -N/2+1:-1]';

what = 1i*kk .* vhat;
% what(kk == N/2) = 0;
w = real(ifft(what)); 

%% MOST INTERESTING CODE #2.

clf
subplot(2,2,1), plot(x, v, '.-', 'markersize', 13);
axis([0 2*pi -0.5 1.5]), grid on, title('function')
subplot(2,2,2), plot(x, w, '.-', 'markersize', 13);
axis([0 2*pi -1 1]), grid on, title('spectral derivative')

% Differentiation of exp(sin(x))
v = exp(sin(x)); vprime = cos(x).*v;

vhat = fft(v);
what = 1i*kk .* vhat;
w = real(ifft(what));

subplot(2,2,3), plot(x, v, '.-', 'markersize', 13);
axis([0 2*pi 0 3]), grid on
subplot(2,2,4), plot(x, w, '.-', 'markersize', 13), hold on
axis([0 2*pi -2 2]), grid on
error = norm(w - vprime, inf);
text(2.2, 1.4, ['max error = ', num2str(error)]);