%P33 -- Solve linear BVP 
% u_xx = exp(4x)
% u'(-1) = 0.
% u(1)   = 0.

%% Figure window.
close all;
scrsz = get(groot,'ScreenSize');
figure('position', [150   150   3/5*scrsz(3:4)]);

%% Main script
% Generate the differentiation matrix.
N = 16;
[D,x] = cheb(N);
D2 = D^2;

option = 2;
switch option
    case 1 
        % This method is given by Trefethen's book.
        %--------------------------------------------------------------
        % Apply the Neumann boundary condition by using the first-order
        % differentiation matrix at the leftmost grid point.
        D2(N+1,:) = D(N+1,:);
        % For the internal grid points the second-order differentiation matrix can
        % be used as normal.
        D2 = D2(2:N+1,2:N+1);
        f = exp(4*x(2:N));
        u = D2\[f; 0];  % Solve the system for solution vector u(2:N+1).
        u = [0; u];     % Concatenate the solution to boundary value.
    case 2
        % This method (by Khiem Nguyen) applies both Neumann boundary
        % condition and Dirichlet condition in the system matrix.
        %--------------------------------------------------------------
        % Apply the Neuman boundary condition as in Option 1.
        D2(N+1, :) = D(N+1, :);
        % Apply the Dirichlet condition by equation u_{0} = 0.
        D2(1, :) = [1, zeros(1, N)];
        f = [0; exp(4*x(2:end-1)); 0];
        u = D2\f;       % Solve the system for solution vector u.
        % No need to concatenate now because we have the full solution
        % vector.
end

%% Visualization.
fs = 14; % Fonstize.
plot(x, u, 'o', 'MarkerFaceColor', 0.5*ones(1,3), 'Linewidth', 1.5);
axis([-1 1 -4 0]);
% Plot the numerical solution.
xx = linspace(-1, 1, 101);
uu = polyval(polyfit(x,u,N), xx);
line(xx, uu, 'linewidth', 0.8, 'Color', zeros(1,3)); grid on
% Exact solution.
exact = (exp(4*xx) - 4*exp(-4)*(xx-1) - exp(4))/16;
% Compute the error of the numerical solution.
title(['max err = ' num2str(norm(uu-exact,inf))], 'fontsize', fs);