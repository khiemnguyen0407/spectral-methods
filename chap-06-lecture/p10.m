%P10  Polynomials and corresponding equipotential curves
close all;
scrsz = get(groot,'ScreenSize');
figure('position', [150   100   0.8*scrsz(3:4)]), clf;

%%
N = 16; 
fs = 14;
for i = 1:2
    if i == 1, s = 'equispaced points'; x = -1 + 2*(0:N)/N; end
    if i == 2, s = 'Chebyshev points'; x = cos(pi*(0:N)/N); end
    p = poly(x);
    % Plot p(x) over [-1,1]:
    
    y = -1: 0.005: 1;
    q = polyval(p, y);
    subplot(2,2,2*i-1)
    plot(x, 0*x, 'o', 'MarkerFaceColor', 0.5*ones(1,3)), hold on
    plot(y, q, 'k-', 'linewidth', 0.8), grid on
    set(gca, 'xtick', -1:0.5:1)
    title(s, 'FontSize', fs)
    
    % Plot equipotential curves:
    subplot(2,2,2*i)
    plot(real(x), imag(x), 'o', 'MarkerFaceColor', 0.5*ones(1,3)), hold on
    axis([-1.4 1.4 -1.12 1.12])
    xgrid = linspace(-1.4, 1.4, 101);
    ygrid = linspace(-1.12, 1.12, 101);
    [xx, yy] = meshgrid(xgrid, ygrid);
    zz = xx + 1i*yy;
    pp = polyval(p, zz); levels = 10.^(-4:0);
    contour(xx, yy, abs(pp), levels), title(s, 'FontSize', fs), colormap([0 0 0]);
end
