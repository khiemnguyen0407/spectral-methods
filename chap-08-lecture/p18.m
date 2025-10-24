% program18.m - Chebyshev differentiation via FFT
%               Compare program11.m
close all;
scrsz = get(groot,'ScreenSize');
figure('position', [100   100   4/5*scrsz(3)  4/5*scrsz(4)]); 

xx = linspace(-1,1,201);
ff = exp(xx).*sin(5*xx); clf
fs = 14;
for N = [10   20]
    
    % Plot the function.
    % x = cos(pi*(0:N)'/N);             % option 1
    x = cos(pi * linspace(0, 1, N)');   % option 2
    f = exp(x).*sin(5*x);
    subplot(2, 2, 2*(N == 20) + 1);
    plot(x, f,'.','markersize', 16), grid on
    line(xx,ff,'linewidth', 0.8, 'color', 'k');
    title(['f(x), N=' int2str(N)], 'FontSize', fs);
    
    
    % Plot the error of the derivatives.
    error = chebfft(f) - exp(x).*(sin(5*x)+5*cos(5*x));
    subplot(2, 2, 2*(N == 20) + 2);
    plot(x, error, '.', 'markersize', 16), grid on
    line(x, error, 'linewidth', 0.8, 'color', 'k');
    title(['error in f^{\prime}(x), N=', int2str(N)], 'FontSize', fs);
end