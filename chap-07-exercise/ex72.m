%%EX72 -- Exercise 7.2: Solve the boundary value problem.
%   u_{xx} + 4 u_{x} + \exp(x) u = sin(8x).


close all;
scrsz = get(groot,'ScreenSize');
figure('position', [0.1*scrsz(3), 0.08*scrsz(4), 0.8*scrsz(3), 0.80*scrsz(4)]); 

N = 2^5;
[D, x] = cheb(N);
D2 = D^2;
D2 = D2(2:N,2:N);           % Boundary conditions
L = D2 + 4*D(2:N,2:N) + diag(exp(x(2:N)));
f = sin(8*x(2:N));
u = L\f;
u = vertcat(0, u, 0);

plot(x, u, 'r.', 'markersize', 16);
xx = linspace(-1, 1, 201);
uu = spline(x, u, xx);
line(xx, uu, 'linewidth', 0.8, 'color', [0 0 0]), grid on
text(-0.8, 0.018, [num2str(length(u)), ' points'], 'fontsize', 16)
text(-0.1, -0.002, sprintf('u(0) = %16.14f', u(abs(x) < 1e-16)), 'fontsize', 16)