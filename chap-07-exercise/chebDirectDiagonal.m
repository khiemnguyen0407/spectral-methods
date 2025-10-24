%CHEBDIRECTDIAGONAL  Spectral differentiation using 
%       the Chebyshev collocation poitns
% 
% D = differentiation matrix
% x = Chebyshev grid

function [D, x] = chebDirectDiagonal(N)

% If N == 0, there is only one collocation point at x = 1. The derivative
% by this interpolation is obviously zero since a constant function is
% generated.
if N == 0
    D = 0;
    x = 1;
    return
end
% Chebyshev grid.
x = cos(pi*(0:N)/N)';
% Intermediate values entering the differentiation matrix.
c = [2; ones(N-1,1); 2].*(-1).^(0:N)';
X = repmat(x, 1, N+1);
dX = X - X';
% Assemble values into the differentiation matrix.
D = (c*(1./c)')./(dX+(eye(N+1)));           % off-diagonal entries
% D = D - diag(sum(D,2));                   % diagonal etries


% % Diagonal entries according to the direct formula.
a = (2*N*N + 1)/6;
D(1:N+2:end) = [a;  -0.5*x(2:end-1)./(1-x(2:end-1).^2);  -a];