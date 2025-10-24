%P15  Solve eigenvalue boundary-value problem
%     u_xx = lambda*u, u(-1) = u(1) = 0
close all;
scrsz = get(groot,'ScreenSize');
figure('position', [0.025*scrsz(3), 0.02*scrsz(4), 0.95*scrsz(3), 0.90*scrsz(4)]); 

% N = 32;
% N = 48;
N = 60;
% N = 256;
[D, x] = cheb(N);
D2 = D^2;
D2 = D2(2:N,2:N);
[V, Lam] = eig(D2);
lam = diag(Lam);
[foo, ii] = sort(-lam);     % sort eigenvalues and -vectors
lam = lam(ii); V = V(:,ii); clf

ws = warning('off','all');  % Turn off warning

for j = 5:5:30              % plot 6 eigenvectors
    u = [0; V(:,j); 0];
    subplot(6, 1, j/5);
    plot(x, u, 'r.', 'markersize', 14),     grid on
    xx = linspace(-1,1, 501);
%     uu = polyval(polyfit(x,u, N), xx);
%     ue = sin(0.5*j*pi*(xx+1));
%     coef = uu(100)/ue(100);
%     uexact = coef * ue;
    uu = spline(x, u, xx);
    line(xx, uu, 'color', [0, 0, 0], 'linewidth', 1.0), axis on
    text(-0.3, 0.35, sprintf('eig %d = %20.13f*pi^2/4', j, lam(j)/pi^2*4), 'fontsize', 13);
    text(0.8, 0.35, sprintf('%4.1f ppw', 4*N/(pi*j)), 'fontsize', 13);
    axis([-1  1  -0.5  0.5])
end