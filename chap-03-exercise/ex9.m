%%EX9 -- Exercise 9: 
%   
%   u(x,

close all;
scrsz = get(groot,'ScreenSize');
figure('position', [100   100   4/5*scrsz(3)  4/5*scrsz(4)]); 

% Eigenvalues of the differentiation matrix D.
Nvec = 2.^(6:9);
for i = 1:length(Nvec)
    N = Nvec(i);
    h = 2*pi/N;
    x = h*(1:N)';
    column = [0 0.5*(-1).^(1:N-1).*cot((1:N-1)*h/2)]';
    D = toeplitz(column,column([1 N:-1:2]));
    lambdaD = eig(D);
    
    % Eigenvalues of the differential operator associated with the PDE
    % u_t + c(x) u_x = 0 with c(x) = 1/5 + sin(x-1)^2.
    c = 1/5 + sin(x-1).^2;
    % Differential operator.
    L = diag(c) * D;
    % In fact, we should write this equation as u_t = -c(x) u_x and hence the
    % corresponding differential opertator should be L = -diag(c) * D. But the
    % eigenvalues of L and -L are only different from a multiplicator -1, and
    % we are interested in the modulus the such eigenvalues. Hence it suffices
    % to define L = diag(c)*D which corresponds to the study of the equation
    % u_t = c(x) u_x.
    lambdaL = eig(L);
    
    % Ratio between the modulus of eigenvalues of L and D operators.
    ratio = abs(lambdaL) ./ abs(lambdaD); % abs(lambda) computes the modulus of lambda in the complex plan.
    subplot(2,2,i), plot(ratio);
    title('Ratio between the modulus of eigenvalues of L and D');
    ylabel('|lambda_L| / |lambda_D|');
    xlabel('Indices of the eigenvalues');
    % We compute the maximum of ratio it should be around 6/5.
    text(N/4, 0.3*round(max(ratio)), ['The maximum ratio is: ', num2str(round(max(ratio),4))],...
        'fontsize', 14);
    text(N/4, 0.45*round(max(ratio)), ['N = ', num2str(N)], 'fontsize', 14);
end
