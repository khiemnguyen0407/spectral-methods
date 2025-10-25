%P2  convergence of periodic spectral method
close all;
scrsz = get(groot,'ScreenSize');
figure('position', [50   50   4/5*scrsz(3)  4/5*scrsz(4)]); 

% For various N, set up grid in [-pi, pi] and function u(x):
Nvec = 2.^(3:12);
clf, subplot(2,2,3);
for N = Nvec
    h = 2*pi/N;
    x = -pi + (1:N)'*h;
    u = exp(sin(x));
    uprime = cos(x).*u;
    % Construct sparse 4th-order differentiation matrix:
    e = ones(N,1);
    D =   sparse(1:N,[2:N 1], 2*e/3,N,N)...
        - sparse(1:N,[3:N 1 2], e/12,N,N);
    D = (D - D')/h;
    % Plot max(abs(D*u - uprime)):
    error = norm(D*u - uprime, inf);
    loglog(N, error, '.', 'markersize', 15), hold on;
end
grid on, xlabel N, ylabel error
title('Converge of 4th-order finite differences');
semilogy(Nvec,Nvec.^(-4), '--');
text(105, 5e-8, 'N^{-4}', 'fontsize', 18);

% For various N (even), set up grid as before:
subplot(2,2,4);
for N = 2:2:100
    h = 2*pi/N;
    x = -pi + (1:N)'*h;
    u = exp(sin(x)); 
    uprime = cos(x).*u;
    % Construct spectral differentiation matrix:
    column = [0 .5*(-1).^(1:N-1).*cot((1:N-1)*h/2)];
%     column = [0 .5*(-1).^(1:N-2).*cot((1:N-2)*h/2), 0.5*cot(h/2)];
    D = toeplitz(column, column([1 N:-1:2]));
    % Plot max(abs(D*u - uprime)):
    error = norm(D*u - uprime,inf);
    loglog(N, error, '.', 'markersize', 15,'color','k'), hold on
end
grid on, xlabel N, ylabel error
title('Convergence of spectral differentiation')

% Plot the function and its derivative.
N = 2^8; h = 2*pi/N; x = -pi + (1:N)'*h;
u = exp(sin(x));
uprime = cos(x).*u;
subplot(2,2,1)
plot(x,u,'k')
title('u = exp(sin(x))')
grid on, xlabel x, ylabel u
subplot(2,2,2)
plot(x,uprime,'k')
title('u_x = exp(sin(x))*cos(x)')
grid on, xlabel x, ylabel u_x