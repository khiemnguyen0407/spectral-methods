% Exercise 7.4 - solve nonlinear BVP u_xx = exp(u), u(-1) = u(1) = 0
% using the Newton method instead of the fixed-point scheme.
close all;
scrsz = get(groot,'ScreenSize');
figure('position', [0.2*scrsz(3), 0.2*scrsz(4), 0.6*scrsz(3), 0.60*scrsz(4)]);

N = 16;
[D, x] = cheb(N);
D2 = D^2;
D2 = D2(2:end-1,2:end-1);
u = zeros(N-1, 1);

nIter = 200;
TOL = 1e-5;
for i = 1:nIter
    % Compute Jacobian matrix.
    J = D2 - diag(exp(u));
    % Compute increment of solution.
    du = J\(exp(u)- D2*u);
    % Update the solution.
    u = u + du;
    disp(norm(du));
    % Check the stop criterion.
    if norm(du)/norm(u) < TOL
        break
    end
end
u = [0; u; 0];
plot(x, u, 'r.', 'markersize', 16)

% Exact solution using Mathematica software.
xx = linspace(-1, 1, 201);
c1 = -1.38416; c2 = 0;
uexact = log(0.5*c1*( -1 + tanh(0.5*sqrt(c1)*abs(xx)).^2) );
line(xx, uexact , 'linewidth', 0.8, 'color', [0 0 0]), grid on
title(sprintf('no. steps = %d    u(0) = %18.14f', i, u(N/2+1)),'fontsize', 14);
text(-0.4, -0.225, ...
    ['absolute error at x = 0:  ', num2str(abs(uexact(xx == 0) - u(abs(x) < 1e-16)))],...
    'fontsize', 16);
