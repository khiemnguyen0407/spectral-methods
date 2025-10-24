% p35.m - Allen-Cahn eq. as in p34.m, but with boundary condition
% imposed explicitly ("method (II)")

%% Figure window.
close all;
scrsz = get(groot,'ScreenSize');
figure('position', [100   100   4/5*scrsz(3)  4/5*scrsz(4)]);

%% Main script.
% Differentiation matrix and initial data:
N = 2^6; 
[D,x] = cheb(N); 
D2 = D^2;
% Parameter epsilon in the Allen-Cahn equation.
eps = 0.01;
% Chosen time step.
if N <= 20
    dt = min([0.01  50*N^(-4)/eps]);
elseif N > 20 && N <= 32
    dt = min([0.01  10*N^(-4)/eps]);
elseif N >= 32
    dt = min([0.01  5*N^(-4)/eps]);
end
% Initial time.
t = 0;
% Initial condition.
v = .53*x + .47*sin(-1.5*pi*x);

% Solve PDE by Euler formula and plot results:
tmax = 100;
tplot = 2;
nplots = round(tmax/tplot);
plotgap = round(tplot/dt);
dt = tplot/plotgap;
xx = linspace(-1, 1, 101);
vv = polyval(polyfit(x,v,N),xx);
plotdata = [vv; zeros(nplots,length(xx))];
tdata = t;

wb = waitbar(0, 'please wait...');
pause(0.75);
% Loop for solution vector at all time steps.
for i = 1:nplots
    waitbar(i/nplots, wb, ['Process: ', num2str(floor(i/nplots*100)), '%']);
    for n = 1:plotgap
        % Update the current time.
        t = t + dt;
        % Update the solution using the forward Euler method.
        v = v + dt*(eps*D2*v + v - v.^3);
        % Enforce the boundary condition at the rightmost grid point.
        v(1) = 1 + sin(t/5)^2;
        % Enforce the boundary condition at the leftmost grid point.
        v(end) = -1; 
    end
    % Store the solution data only at desired time steps.
    vv = polyval(polyfit(x,v,N),xx);
    plotdata(i+1,:) = vv;
    tdata = [tdata; t]; %#ok
end

%% Visualization.
mesh(xx,tdata,plotdata), grid on, axis([-1 1 0 tmax -1 2])
view(-60,55), colormap([0 0 0]), xlabel x, ylabel t, zlabel u