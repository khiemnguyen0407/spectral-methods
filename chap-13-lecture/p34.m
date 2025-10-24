% program34.m - Allen-Cahn eq. u_t = eps*u_xx + u - u^3, u(-1)=-1, u(1)=1
%               (Compare program6.m and program32.m)

%% Figure window.
close all;
scrsz = get(groot,'ScreenSize');
figure('position', [150   150   3/5*scrsz(3:4)]);

%% Main script
% Differentiation matrix and initial data:
N = 32;
[D, x] = cheb(N);
D2 = D^2;      % use full-size matrix
% For convenience: We simply ignore the first and the last rows of the
% differentiation matrix.
D2([1 N+1],:) = zeros(2,N+1);
% Parameter epsilon in the Allen-Cahn equation.
eps = 0.01;
% Chosen time step.
if N <= 20
    dt = min([0.01, 50*N^(-4)/eps]);
elseif N > 20
    dt = min([0.01, 10*N^(-4)/eps]);
end
% Initial time.
t = 0;
% Initial condition.
v = 0.53*x + 0.47*sin(-1.5*pi*x);

% Solve PDE by Euler formula and plot results.
tmax = 100;
tplot = 2;
nplots = round(tmax/tplot);
plotgap = round(tplot/dt);
dt = tplot/plotgap;
xx = linspace(-1, 1, 101);
vv = polyval(polyfit(x,v,N),xx);
plotdata = [vv; zeros(nplots,length(xx))];
tdata = t;

% Shape functions. See program p14.m to see what this snippet means.
%------------------------------------------------------------------------
shape_val = ones(length(x), length(xx));
for j = 1: length(x)
    for k = setdiff(1:length(x), j)
        shape_val(j, :) = shape_val(j, :) .* (xx - x(k)) ./ (x(j) - x(k));
    end
end
%------------------------------------------------------------------------

wb = waitbar(0, 'please wait...');
pause(0.75);
% Loop for solution vector at all time steps.
for i = 1:nplots
    waitbar(i/nplots, wb, ['Process: ', num2str(floor(i/nplots*100)), '%']);
    for n = 1:plotgap
        % Update the current time.
        t = t + dt;
        % Update the solution using the forward Euler method.
        v = v + dt*(eps*D2*(v-x) + v - v.^3);
    end
    % Compute the numerical solution onto a finer grid for
    % visualization.
        vv = transpose(v) * shape_val; % Program p14.m to see what this means.
%     vv = polyval(polyfit(x, v, N), xx);
    plotdata(i+1,:) = vv;
    tdata = [tdata; t]; %#ok
end
pause(0.75), delete(wb); % Close the waitbar.

clf
mesh(xx,tdata,plotdata), grid on, axis([-1 1 0 tmax  1.1*min(min(vv))  1.1*max(max(vv))]);
view(-60,55), colormap([0 0 0]), xlabel x, ylabel t, zlabel u