%% Oringial function and mesh.
Nvec = 2.^(3 : 12);

%% Differentiation matrix for central difference formula
for N = Nvec
    h = 2*pi/N;
    x = -pi + (1:N)' * h;
    u = exp(sin(x));
    uprime = cos(x).*u;
    %% MOST INTERESTING CODE.
    % Construct 2nd-order differentiation matrix
    c = [0, -1/2, zeros(1,N-3), 1/2]';
    r = [c(1); flipud(c(2:end))];
    D2 = (1/h)*toeplitz(c,r);
    
    % Construct 4th-order differentiation matrix
    c = [0, -2/3, 1/12, zeros(1,N-5),  -1/12, 2/3]';
    r = [c(1); flipud(c(2:end))];
    % D4 = (1/h)*toeplitz(c,r);
    D4 = (1/h)*toeplitz(c, c([1, end:-1:2]));
    %%
    % Compute the error of the numerical derivative
    % 2nd-order approximation
    %% MOST INTERESTING CODE.
    error1 = norm(D2*u - uprime, inf);   % Inf-norm takes the maximum value.
    %% 
    subplot(2,2,2)
    loglog(N, error1, '.', 'markersize', 15),  hold on
    grid on, xlabel N, ylabel error
    title('Convergence of 2nd-order finite differences');
    semilogy(Nvec,Nvec.^(-2), 'k--');
    text(105, 3e-4, 'N^{-2}', 'fontsize', 18);
    
    % 4th-order approximation
    %% MOST INTERESTING CODE.
    error2 = norm(D4*u - uprime, inf);
    %%
    subplot(2,2,4)
    loglog(N, error2, '.', 'markersize', 15), hold on
    grid on; xlabel N; ylabel error
    title('Convergence of 4th-order finite differences');
    semilogy(Nvec,Nvec.^(-4), 'k--');
    text(105, 5e-8, 'N^{-4}', 'fontsize', 18);
    
    if N == 2^4
        R2 = D2; x2 = x; v2 = u;
        R4 = D4; x4 = x; v4 = u;
    end
end

%% 
subplot(2,2,1)
N = 2^8; h = 2*pi/N;
xx = -pi + (1:N)' * h;
f = exp(sin(xx)).*cos(xx);
plot(xx,f,'k-', x2, R2*v2,'bo'), grid on
xlabel x, ylabel u_x
title('u_x = exp(sin(x))*cos(x)')
subplot(2,2,3)
plot(xx,f,'k-', x4, R4*v4,'bo'), grid on
xlabel x, ylabel u_x
title('u_x = exp(sin(x))*cos(x)')