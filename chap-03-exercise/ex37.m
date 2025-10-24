%EX37 -- Exercise 3.7 Solve the wave equation 
close all

N = 2^8; h = 2*pi/N;
column = [0 0.5*(-1).^(1:N-1).*cot((1:N-1)*h/2)]';
D = toeplitz(column,column([1 N:-1:2]));


x = h*(1:N)'; t = 0; dt = (5/3)/N;
c = 1/5 + sin(x-1).^2;
% Initial condition:
v = exp(-100*(x-1).^2);
% Extrapolate the set of conditions at (-dt):
vold = exp(-100*(x-0.2*dt-1).^2);
% Time-stepping by leap frog formula:
tmax = 8;
tplot = 0.15;
plotgap = round(tplot/dt); dt = tplot/plotgap;
nplots = round(tmax/tplot);
data = [v,  zeros(N,nplots)]; tdata = t;
% Wavenumbers
L = diag(c) * D;
tstart = tic;
for i = 1:nplots
    for n = 1:plotgap
        t = t + dt;
        vnew = vold - 2*dt*L*v;
        vold = v;
        v = vnew;
    end
    data(:,i+1) = v; tdata = [tdata; t]; %#ok
end
data = transpose(data);
tend = toc(tstart);
[xx, tt] = meshgrid(x,tdata);
waterfall(xx, tt, data), view(10,70)
axis([0 2*pi 0 tmax  -0.05  5]); ylabel t; zlabel u; grid off
title('Spectral Differentiation: No Numerical Dispersion Effect')
text(0.1,9.6, ['Number of grid points N = ', num2str(N)], 'FontSize', 12);
text(0.22,8.6, ['Running time = ', num2str(tend)], 'FontSize', 12);