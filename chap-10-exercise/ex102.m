%EX103  -- Exercise 10.3: Solve the first-oder wave equation using the
% third-order Adam-Bashforth method.

close all;
scrsz = get(groot, 'ScreenSize');
figure('position', [100  100  4/5*scrsz(3)  4/5*scrsz(4)]);

N = 128;
nu = 5;
dt = 0.01/N;
beta = 0.5;
[D, x] = cheb(N);
% Initial condition.
v = exp(-60*(x - beta).^2);
vOld = exp(-60*(x + dt - beta).^2);
vOldOld = exp(-60*(x + 2*dt - beta).^2);
% % % v = exp(-60*(x - beta).^2);
% % % vNew = exp(-60*(x - dt - beta).^2);
% % % vNewNew = exp(-60*(x-2*dt - beta).^2);

% Exact solution: u(x,t) = exp(-60*(x - t - 0.5).^2);
T = 3;
nstep = round(T/dt);
nSlide = 50; 
tSlide = 1:round(nstep/nSlide):nstep+1;
nplots = length(tSlide);
t = zeros(nplots,1);


u = zeros(nplots,N+1);
u(1,:) = v;
j = 2;
wb = waitbar(0, 'please wait...');
for n = 1:nstep
    waitbar(n/nplots,wb, ...
        ['process ... ' , num2str(round(n/nstep,2)*100), '%']);
    vNew = v + (dt/12)*D*(23*v - 16*vOld + 5*vOldOld);
    vNew(1) = 0;
    vOldOld = vOld;
    vOldOld(1) = 0;
    vOld = v;
    vOld(1) = 0;
    v = vNew;
% % %     v = vNew;
% % %     vNew = vNewNew;
% % %     vNewNew = vNewNewNew;

    if ~isempty(find(n+1 == tSlide, 1))
        u(j,:) = v';
        t(j) = n*dt;
        j = j + 1;
    end
end

% Plot results.
waterfall(x, t, u);
axis([-1 1 0 T -0.1 2]), view(10, 70), grid off
colormap([0 0 0]), ylabel t, zlabel u, delete(wb);