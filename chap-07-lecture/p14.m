% program14.m - solve nonlinear BVP u_xx = exp(u), u(-1) = u(1) = 0
%   (compare program13.m)
close all;
scrsz = get(groot,'ScreenSize');
figure('position', [0.2*scrsz(3), 0.2*scrsz(4), 0.6*scrsz(3), 0.60*scrsz(4)]); 

N = 16;
[D, x] = cheb(N);
D2 = D^2;
D2 = D2(2:N,2:N);
u = zeros(N-1,1);
change = 1; it = 0;
while change > 1e-15        % fixed-point iteration
%     changeOld = change;
    unew = D2\(exp(u));   % unew = inv(D2)*exp(u)
    change = norm(unew - u, inf);
%     if changeOld ~= 1
%         disp(change/changeOld);
%     end
    u = unew; it = it + 1;
end
% Eigenvalue of the D2 matrix.
lambda = eig(D2);
u = [0; u; 0];
% clf, subplot('position', [.1 .4 .8 .5]);
plot(x, u, 'r.', 'markersize', 16)

%% Exact solution using Mathematica software.
xx = linspace(-1,1,201);
c1 = -1.38416; c2 = 0;
uexact = log(0.5*c1*( -1 + tanh(0.5*sqrt(c1)*abs(xx)).^2) );
line(xx, uexact , 'linewidth', 0.8, 'color', [0 0 0]), grid on
title(sprintf('no. steps = %d    u(0) = %18.14f', it, u(N/2+1)),'fontsize', 14);
text(-0.4, -0.225, ...
    ['absolute error at x = 0:  ', num2str(abs(uexact(xx == 0) - u(abs(x) < 1e-16)))],...
    'fontsize', 16);

%% Plot the numerical solution using the Lagrange interpolation.
shape_val = ones(length(u), length(xx));
uu = zeros(size(xx));
for j = 1: length(u)
    for k = setdiff(1:length(u), j)
        shape_val(j, :) = shape_val(j, :) .* (xx - x(k)) ./ (x(j) - x(k));
    end
end
figure('position', [0.2*scrsz(3), 0.2*scrsz(4), 0.6*scrsz(3), 0.60*scrsz(4)]); 
plot(x, u, 'o', 'MarkerFaceColor', 0.5*ones(1,3)), hold on, grid on;
plot(xx, transpose(u) * shape_val, 'k-');