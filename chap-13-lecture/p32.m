%P32  Solve boundary value problem
%   u_xx = exp(4x),
%   u(-1) = 0,
%   u(1) = 1.
% 
% We can solve this boundary value problem by using a change of variable
%   u(x) = v(x) - (1+x)/2   (1)  ==>   v_{xx} = u_{xx}.     
% Then, v(x) is the solution of the BVP with homogeneous boundary condtions
%   v_{xx} = exp(4x)
%   v(-1) = 0,
%   v(1) = 0.
% We can solve this equation by using the method discussed in Chapter 7
% (see p13.m). The original solution u(x) is obtained through the solution
% v according to Equation (1).
%
% Option 2: We may solve the original BVP directly by enforcing the
% boundary condition u(-1) = 0 and u(1) = 1 with the following algebraic
% equation:
%   u_{0} = 1,   u_{N+1} = 0.
% Recall that x_0 = 1 and x_{N+1} = -1 by using x_j = cos(j*pi/N).

%% Figure for plot
close all;
scrsz = get(groot,'ScreenSize');
figure('position', [100   100   3/5*scrsz(3:4)]);

%% Main script
% Generate the differentiation matrix.
N = 16;
[D, x] = cheb(N);
D2 = D^2;

% Option for choosing the method of solver.
option = 1;
switch option
    case 1
        D2 = D2(2:N,2:N);
        f = exp(4*x(2:N));
        u = D2\f;
        u = [0;u;0] + 0.5*(x+1);
        %-------------------------------------
    case 2
        firstRow= [1, zeros(1,N)];
        lastRow = [zeros(1,N), 1];
        D2 = D2(2:N, :);
        L = [firstRow; D2; lastRow];
        % Set up the right-hand side vector.
        f = [1; exp(4*x(2:N)); 0];
        u = L\f;
        %-------------------------------------
end
clf
subplot('position',[0.1 0.2 0.8 0.6]);
fs = 14;
plot(x, u, '.', 'markersize', 16);
xx = linspace(-1, 1, 201);
uu = polyval(polyfit(x,u,N),xx);
line(xx, uu, 'color', zeros(1,3), 'linewidth', 0.8), grid on
u_exact = (exp(4*xx) - sinh(4)*xx - cosh(4))/16 + 0.5*(xx+1);
axis([-1, 1, -2.5, 1.01])
title(['max err = ' num2str(norm(uu - u_exact,inf))],'fontsize', fs);