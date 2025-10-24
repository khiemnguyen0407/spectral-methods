% program27.m - Solve KdV eq. u_t + u u_x + u_xxx = 0 on [-pi, pi] by
%               FFT with integrating factor v = exp(-ik^3t)*u-hat.
close all;
scrsz = get(groot,'ScreenSize');
figure('position', [100   100   4/5*scrsz(3:4)]); 

%%
% Set up grid and two-soliton initial data:
N = 2^8; 
dt = 0.4/N^2; 
x = (2*pi/N)*(-N/2:N/2-1)';
A = 25; 
B = 16;
clf, drawnow
u = 3*A^2*sech(.5*(A*(x+2))).^2 + 3*B^2*sech(.5*(B*(x+1))).^2;
v = fft(u);
k = [0:N/2-1 0 -N/2+1:-1]'; ik3 = 1i*k.^3;
% Solve PDE and plot results:
tmax = 0.006; 
nplots = floor((tmax/50)/dt); 
nMax = round(tmax/dt);
udata = u;
tdata = 0; 
wb = waitbar(0,'please wait...');
for n = 1: nMax
    waitbar(n/nMax, wb, ...
        ['process ...', num2str(round(n/nMax,2)*100), '%']);
    t = n*dt; g = -.5i*dt*k;
    E = exp(dt*ik3/2); E2 = E.^2;
    a = g.*fft(real( ifft( v ) ).^2);
    b = g.*fft(real( ifft(E.*(v+a/2)) ).^2); % 4th-order
    c = g.*fft(real( ifft(E.*v + b/2) ).^2); % Runge-Kutta
    d = g.*fft(real( ifft(E2.*v+E.*c) ).^2);
    v = E2.*v + (E2.*a + 2*E.*(b+c) + d)/6;
    if mod(n, nplots) == 0
        u = real(ifft(v));
        waitbar(n/nMax)
        udata = [udata u];  %#ok<AGROW>
        tdata = [tdata t];  %#ok
    end
end

%% 
waterfall(x,tdata,udata'), 
colormap([0 0 0]), view(-20,25)
xlabel x, ylabel t, axis([-pi pi 0 tmax 0 2000]), grid off
set(gca,'ztick',[0 2000]), close(wb), pbaspect([1 1 .13])

