%EX133 - Solve the second-order wave equation u_tt - u_xx = 0 with the
% boundary condition

%%
figure;
scrsz = get(groot,'ScreenSize');
figure('position', [100   100   4/5*scrsz(3:4)]); 

%% Main script
N = 2^8; 
[D, x] = cheb(N);
L = zeros(N+1,N+1);
D2 = D^2;
% BC = -D([1  N+1],[1  N+1])\D([1  N+1],2:N);
BC = -D(N+1,N+1)\D(N+1,1:N);

T = 1.0;
dt = 4/N^2;
nstep = round(T/dt);
dt = T/nstep;
nSlide = 50;
tSlide = 1:round(nstep/nSlide):nstep+1;
nplots = length(tSlide);
t = zeros(nplots,1);

% Initial conditions.
v = exp(-200*x.^2);
vold = exp(-200*(x-dt).^2);
%----------------------------------------


wb = waitbar(0, 'please wait...');
pause(1);

u = zeros(nplots,N+1);
u(1,:) = v;
j = 2;
dt2 = dt*dt;
for n = 1:nstep
    waitbar(n/nstep,wb, ...
        ['process ... ' , num2str(round(n/nstep,2)*100), '%']);
    vnew = 2*v - vold + dt^2*D2*v;
    vold = v;
    v = vnew;
    v(1) = sin(10*n*dt);        % Dirichlet BC for y = 1.
    v(N+1) = BC * v(1:N);       % Neumann BC for y = -1.
    if ~isempty(find(n+1 == tSlide,1))
        u(j,:) = v';
        t(j) = n*dt;
        j = j + 1;
    end
end
waitbar(1, wb, 'finised');
pause(1);
delete(wb);

%% Plot the wave solution.
waterfall(x, t, u)
axis([-1 1 0 T -2 2]), view(10, 70), grid off
colormap([0 0 0]),
fs = 14;
xlabel('x', 'FontSize', fs)
ylabel('t', 'FontSize', fs)
zlabel('u', 'FontSize', fs)