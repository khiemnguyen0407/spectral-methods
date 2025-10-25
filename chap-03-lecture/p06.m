%P6  wave equation with variable coefficient
close all;
scrsz = get(groot,'ScreenSize');
figure('position', [50   50   4/5*scrsz(3)  4/5*scrsz(4)]); 
%% Solver using spectral differentiation.
% Grid, variable coefficient, and initial data:
N = 2^10;
h = 2*pi/N; 
x = h*(1:N); 
t = 0;
dt = 1.7/N;
% dt = 0.01;

c = 1/5 + sin(x-1).^2;
% Initial condition:
v = exp(-100*(x-1).^2);
% Extrapolate the set of conditions at (-dt):
vold = exp(-100*(x-0.2*dt-1).^2);
% Time-stepping by leap frog formula:
tmax = 50;
tplot = 0.15;
plotgap = round(tplot/dt); dt = tplot/plotgap;
nplots = round(tmax/tplot);
data = [v; zeros(nplots,N)]; tdata = t;
% Wavenumbers
k = [0:N/2-1 0 -N/2+1:-1];
tstart = tic;
for i = 1:nplots
    for n = 1:plotgap
        t = t + dt;
        v_hat = fft(v);
        w_hat = 1i*k .* v_hat;
        w = real(ifft(w_hat));
        vnew = vold - 2*dt*c.*w;
        vold = v;
        v = vnew;
    end
    data(i+1,:) = v; tdata = [tdata; t]; %#ok
end
tend = toc(tstart);
[xx, tt] = meshgrid(x,tdata);
subplot(2,2,4)
% figure(4)
tindex = 1:length(t);
waterfall(xx, tt, data), view(10,70)
colormap([0 0 0])
axis([0 2*pi 0  tmax  -0.05  5]);
ylabel t; zlabel u; grid off
title('Spectral Differentiation: No Numerical Dispersion Effect')
text(0.1,9.6, ['Number of grid points N = ', num2str(N)], 'FontSize', 12);
text(0.22,8.6, ['Running time = ', num2str(tend)], 'FontSize', 12);

%% Solver using finite difference method (N = 128, 256, 512).
Nvec = [128, 256, 512];
% Grid, variable coefficient, and initial data:
for j = 1:length(Nvec)
    N = Nvec(j); h = 2*pi/N; x = h*(1:N); t = 0; dt = 6/N^2;
    c = 0.2 + sin(x-1).^2;
    % Initial condition:
    v = exp(-100*(x-1).^2);
    % Extrapolate the set of conditions at (-dt):
    vold = exp(-100*(x-0.2*dt-1).^2);
   
    % Time-stepping by leap frog formula:
    plotgap = round(tplot/dt); dt = tplot/plotgap;
    nplots = round(tmax/tplot);
     % Stored solution.
    data = [v; zeros(nplots,N)]; tdata = t;
    
    % Time-space mesh size ratio.
    dtOverdx = dt/h;
    tstart = tic;
    for i = 1:nplots
        for n = 1:plotgap
            t = t + dt;
            vdiff = [v(2) - v(end), v(3:end) - v(1:end-2), v(1) - v(end-1)];
            vnew = vold - dtOverdx*c.*vdiff;
            vold = v;
            v = vnew;
        end
        data(i+1,:) = v; tdata = [tdata; t]; %#ok
    end
    tend = toc(tstart);
    [xx, tt] = meshgrid(x,tdata);
    subplot(2,2,j)
%     figure(j)
    waterfall(xx, tt, data), view(10,70)
    axis([0 2*pi 0 tmax  -0.05  5]); ylabel t; zlabel u; grid off
    title('Finite Difference: Numerical Dispersion Effect')
    text(1.0,9.6, ['N = ', num2str(N)]);
    text(0.22,8.6, ['Running time = ', num2str(tend)], 'FontSize', 12);
end