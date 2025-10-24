%P13  Solve linear boundary-value problem
%     u_xx = f(x), u(-1) = u(1) = 0
close all;
scrsz = get(groot,'ScreenSize');
figure('position', [0.1*scrsz(3), 0.08*scrsz(4), 0.8*scrsz(3), 0.80*scrsz(4)]); 

%% Solve the boundary-value problem with f = exp(4x)
N = 64;
[D, x] = cheb(N);
D2 = D^2;
D2 = D2(2:end-1,2:end-1);           % boundary conditions
f1 = exp(4*x(2:end-1));
u = D2\f1;
u = [0; u; 0];

% Plot solution using spectral method ------------
subplot(2,2,1);
plot(x, u, 'r.', 'markersize', 12);
xx = -1:0.01:1;
uexact1 = (exp(4*xx) - sinh(4)*xx - cosh(4))/16;
line(xx, uexact1, 'linewidth', 0.8, 'color', [0 0 0]), grid on
exactgrid = (exp(4*x) - sinh(4)*x - cosh(4))/16;
title(['max err = ' num2str(norm(u - exactgrid,inf)), ...
    ' -- spectral method'], 'fontsize', fs);
axis([-1  1, -2.5  0])
fs = 14;
ylabel('u_e = (exp(4*x) - sinh(4)*x - cosh(4))/16', 'FontSize', fs)
text(0.5, -0.6, [num2str(length(u)), ' points'], 'FontSize', fs)
%-------------------------------------------------
%% Solve the boundary-value problem with f = x^3.
f2 = x(2:N).^3;
u = D2\f2;
u = [0; u; 0];

% Plot solution using spectral method ------------
subplot(2,2,3)
plot(x, u, 'r.', 'markersize', 12);
xx = linspace(-1,1,201);
uexact2 = 0.05*(xx.^5 - xx); % interpolate grid data.
line(xx, uexact2, 'linewidth', 0.8, 'color', [0 0 0]);
grid on
exactgrid = 0.05*(x.^5 - x);
title(['max err = ' num2str(norm(u - exactgrid,inf)), ' -- spectral method'], 'fontsize', 12);
ylabel('u_e = (x^5 - x)/20', 'FontSize', fs);
text(0.5, 0.015, [num2str(length(u)), ' points'], 'FontSize', fs)
%-------------------------------------------------
%% Solve the boundary-value problem using FEM.
N = N+1;
p = -1:2/(N-1):1;
h = 2/(N-1);
nDof = length(p);
nElements = length(p)-1;
k = zeros(nDof,nDof);
f1 = zeros(nDof,1);     % External source function: f(x) = exp(4x)
f2 = zeros(nDof,1);     % External source function: f(x) = x^3.
for e = 1:nElements
    k(e:e+1,e:e+1) = k(e:e+1,e:e+1) + 1/h*[1  -1; -1  1];
    func1 = @(r) -exp(4*(0.5*(1-r)*p(e) + 0.5*(1+r)*p(e+1))) .* (0.5*(1-r));
    func2 = @(r) -exp(4*(0.5*(1-r)*p(e) + 0.5*(1+r)*p(e+1))) .* (0.5*(1+r));

    f1(e:e+1) = f1(e:e+1) + h/2*[integral(func1,-1,1); integral(func2,-1,1)];
    
    func1 = @(r) -(0.5*(1-r)*p(e) + 0.5*(1+r)*p(e+1)).^3 .* (0.5*(1-r));
    func2 = @(r) -(0.5*(1-r)*p(e) + 0.5*(1+r)*p(e+1)).^3 .* (0.5*(1+r));
    f2(e:e+1) = f2(e:e+1) + h/2*[integral(func1,-1,1); integral(func2,-1,1)];
end
activeDof = 2:N-1;
u = zeros(N,1);
u(activeDof) = k(activeDof,activeDof)\f1(activeDof);

% Plot solution using FEM ------------------------
subplot(2,2,2)
plot(p, u', 'r.', 'markersize', 12)
% Exact solution for f(x) = exp(4x)
uexact1 = (exp(4*xx) - sinh(4)*xx - cosh(4))/16;
line(xx, uexact1, 'linewidth', 0.8, 'color', [0 0 0]), grid on
exactgrid = (exp(4*p) - sinh(4)*p - cosh(4))/16; exactgrid = exactgrid';
title(['max err = ' num2str(norm(u - exactgrid,inf)), ' -- FEM'], 'fontsize', fs);
axis([-1  1, -2.5  0])
ylabel('u_e = (exp(4*x) - sinh(4)*x - cosh(4))/16', 'FontSize', fs);
text(0.5, -0.6, [num2str(length(u)), ' points'], 'FontSize', fs)
%-------------------------------------------------
%% Solve the boundary-value problem with f = x^3 using FEM.
u(activeDof) = k(activeDof,activeDof)\f2(activeDof);

% Plot solution using FEM ------------------------
subplot(2,2,4)
plot(p, u', 'r.', 'markersize', 12)
line(xx, uexact2, 'linewidth', 0.8, 'color', [0 0 0]); grid on
% Exact solution for f(x) = x^3.
exactgrid = 0.05*(x.^5 - x);
title(['max err = ' num2str(norm(u - exactgrid,inf)), ' -- FEM'], 'fontsize', fs);
ylabel('u_e = (x^5 - x)/20', 'FontSize', fs);
text(0.5, 0.015, [num2str(length(u)), ' points'], 'FontSize', fs)
%-------------------------------------------------