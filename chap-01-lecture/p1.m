%P1  Convergence of fourth-order finite differences
close all;
scrsz = get(groot,'ScreenSize');
figure('position', [100   100   4/5*scrsz(3)  4/5*scrsz(4)]); 

%% For various N, set up grid in [-pi, pi] and function u(x):
Nvec = 2.^(3:12);
clf, subplot(2,2,[3, 4]);

for N = Nvec
    %% MOST INTERESTING CODE.
    h = 2*pi/N;
    x = -pi + (1:N)'*h;
    u = exp(sin(x));
    uprime = cos(x).*u;
    
    % Construct sparse 4th-order differentiation matrix:
    e = ones(N,1);
    
    % Code from exercise class.
    c = zeros(length(x), 1);
    c([1 2 3, end-1, end]) = [0, -2/3, 1/12, -1/12, 2/3];
    % As for the Toeplitz matrix, the first element of column and row must
    % be identical. Use "help toeplitz"
    % Option 1: 
	% r = [0; flipud(c(2:end))]; 
    % Option 2:
    r = -c;
    
    D = (1/h)*toeplitz(c, r);
    %----------------------------------------
    % Trefethen book.
%     D =   sparse(1:N,[2:N 1], 2*e/3,N,N)...
%         - sparse(1:N,[3:N 1 2], e/12,N,N);
%     DDD = (D - D');
    %----------------------------------------
    % Plot max(abs(D*u - uprime)):
    error = norm(D*u - uprime, inf);
    %%
    loglog(N, error, '.', 'markersize', 15), hold on;
end
grid on, xlabel N, ylabel error
title('Converge of 4th-order finite differences');
semilogy(Nvec,Nvec.^(-4), '--');
text(105, 5e-8, 'N^{-4}', 'fontsize', 18);

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