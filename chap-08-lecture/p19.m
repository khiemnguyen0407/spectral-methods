% program19.m - 2nd-order wave eq. on Chebyshev grid (compare p6.m)
% Time-stepping by leap frog formula.

close all;
% scrsz = get(groot,'ScreenSize');
% figure('position', [100   100   4/5*scrsz(3)  4/5*scrsz(4)]); 

N = 128; 
x = cos(pi*(0:N)/N); 
dt = 9.5/N^2;

% Initial conditions.
v = exp(-200*x.^2);
vold = exp(-200*(x-dt).^2);

tmax = 0.1;
tplot = 0.005;
plotgap = round(tplot/dt);% dt = tplot/plotgap;
nplots = round(tmax/tplot);
plotdata = [v; zeros(nplots,N+1)]; tdata = 0;
wb = waitbar(0, 'please wait...');
for i = 1:nplots
    waitbar(i/nplots,wb, ...
        ['process ... ' , num2str(round(i/nplots,2)*100), '%']);
    for n = 1:plotgap
        % INTERESTING CODE
        w = chebfft(chebfft(v))';
        w(1) = 0;
        w(N+1) = 0;
        vnew = 2*v - vold + dt^2*w;
        vold = v;
        v = vnew;
    end
    plotdata(i+1,:) = v;
    tdata = [tdata; dt*i*plotgap];      %#ok
end
% Plot results.
waterfall(x,tdata,plotdata)
axis([-1 1 0 tmax -2 2]), view(10, 70), grid off
colormap([0 0 0]), ylabel t, zlabel u, close(wb);