%P11  Chebyshev differentiation of a smooth function
close all;
scrsz = get(groot,'ScreenSize');
figure('position', [150   100   0.8*scrsz(3:4)]); 

%%
xx = -1:0.01:1;
uu = exp(xx).*sin(5*xx); clf
for N = [10 20]
    % MOST INTERESTING CODE
    %==================================
    [D, x] = cheb(N);
    u = exp(x).*sin(5*x);
    %==================================
    subplot(2,2, 2*(N == 20)+1)
    plot(x, u, '.', 'markersize', 14), grid on
    line(xx, uu, 'linewidth', 0.8, 'color','k');
    title(['u(x), N=', int2str(N)]);
    % MOST INTERESTING CODE
    %==================================
    error = D*u - exp(x).*(sin(5*x)+5*cos(5*x));
    %==================================
    subplot(2,2, 2*(N == 20)+2)
    plot(x, error, '.', 'markersize', 14), grid on
    line(x, error, 'linewidth', 0.8, 'color','k')
    title([' error in u''(x), N=' int2str(N)])
end