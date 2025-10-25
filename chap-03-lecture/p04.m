%P4  periodic spectral differentiation using the Differentiation matrix.
close all;
scrsz = get(groot,'ScreenSize');
figure('position', [50   50   4/5*scrsz(3)  4/5*scrsz(4)]); 

% Set up grid and differentiation matrix:
%% MOST INTERESTING CODE
N = 24; h = 2*pi/N; x = h*(1:N)';
j_vector = 1:N-1;
column = [0 0.5*(-1).^j_vector.*cot(j_vector*h/2)]';
% D = toeplitz(column,column([1 N:-1:2]));
row = [column(1); flip(column(2:end))];
D = toeplitz(column, row);

%%
% Differentiation of exp(sin(x))
v = exp(sin(x));
vprime = cos(x).*v;

ii = 2:N-1;
s = [(v(2)-v(1))/h; (v(ii+1)-v(ii-1))/(2*h); (v(end)-v(end-1))/h];
subplot(3,1,1), plot(x, v, '.-k','Markersize', 9)
fs = 16;
text(3.5, 2, 'f(x) = exp(sin(x))','FontSize', fs);
axis([0, 2*pi 0 3]), grid off
title('Original function', 'FontSize', fs);

subplot(3,1,2), plot(x, s, '.r', 'markersize', 9), hold on
plot(x, vprime, '-k');
axis([0 2*pi -2 2]), grid off
error = norm(s(ii) - vprime(ii), inf);
text(1.8, 1.2, ['max error (finite difference) = ' num2str(error)],'FontSize', fs);
title('Derivative and approximation by finite-difference formula' , 'FontSize', fs);

subplot(3,1,3), plot(x,D*v, '.b', 'markersize', 9), hold on
plot(x, vprime, '-k');
axis([0 2*pi -2 2]), grid off
error = norm(D*v - vprime, inf);
text(1.8, 1.2, ['max error (spectral method) = ' num2str(error)],'FontSize', fs);
title('Derivative and approximation by spectral formula', 'FontSize', fs);