%P16  Poisson equation on [-1,1]x[-1,1] 
%     with  u = 0 on boundary

% Set up grids and tensor product Laplacian and solve for u:
N = 32;
[D, x] = cheb(N); 
y = x;
[xx,yy] = meshgrid(x(2:N),y(2:N));
% Strecth 2D grids to 1D vectors.
xx = xx(:);
yy = yy(:);             
f = 10*sin(8*xx.*(yy-1));
D2 = D^2;
D2 = D2(2:N,2:N);
I = eye(N-1);
% L = sparse(kron(I,D2) + kron(D2,I));
L = kron(I,D2) + kron(D2,I);
figure(1), clf, spy(L), drawnow
tic, u = L\f; toc                   % solve problem and watch the clock

% Reshape long 1D results onto 2D grid:
uu = zeros(N+1,N+1); uu(2:N,2:N) = reshape(u,N-1,N-1);
[xx,yy]= meshgrid(x,y);
value = uu(N/4+1,N/4+1);

% Interpolate to finer grid annd plot:
[xxx, yyy] = meshgrid(-1:0.04:1,-1:0.04:1);
uuu = interp2(xx,yy,uu,xxx,yyy,'spline');
figure, clf, mesh(xxx,yyy,uuu), colormap([0 0 0]);
xlabel x, ylabel y, zlabel u
text(0.4,-0.3,-0.3, sprintf('u(2^{-1/2}, 2^{-1/2}) = %14.11f', value));