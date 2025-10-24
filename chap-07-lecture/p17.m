%P17  Helmholtz eq. u_xx + u_yy + k^2 u = f
%     on [-1, 1]x[-1,1] (compare  p16.m)

% Set up spectral grid and tensor product Helmholtz operator:
N = 24; 
[D,x] = cheb(N); 
y = x;
[xx,yy] = meshgrid(x(2:N),y(2:N));
% Strecth 2D matrices to vectors.
xx = xx(:); 
yy = yy(:);
f = exp(-10*((yy-1).^2 + (xx-0.5).^2));
D2 = D^2; 
D2 = D2(2:N,2:N); 
I = eye(N-1);
k = 9;
L = kron(I,D2) + kron(D2,I) + k^2*eye((N-1)^2);
% Solve for u, reshape to 2D grid, and plot:
u = L\f;
uu = zeros(N+1,N+1);
uu(2:N,2:N) = reshape(u,N-1,N-1);
[xx,yy] = meshgrid(x,y);
[xxx,yyy] = meshgrid(-1:1/30:1,-1:1/30:1);
uuu = interp2(xx,yy,uu,xxx,yyy,'spline');

figure(1), clf
mesh(xxx,yyy,uuu)
xlabel x, ylabel y, zlabel u
text(0.2,1,0.022, sprintf('u(0,0) = %13.11f', uu(N/2+1,N/2+1)));

figure(2), clf
[c,h] = contour(xxx,yyy,uuu); axis square
clabel(c,h)