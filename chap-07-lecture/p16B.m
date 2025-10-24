%P16  Poisson equation on [-1,1]x[-1,1] 
%     with  u = 0 on boundary


% Set up grids and tensor product Laplacian and 
N = 3;
[D, x] = cheb(N); 
y = x;
[xx,yy] = meshgrid(x(2:N),y(2:N));
% Strecth 2D grids to 1D vectors.
xx = xx(:);
yy = yy(:);             
f = 10*sin(8*xx.*(yy-1));
D2 = D^2;
D2 = D2(2:N,2:N);
L = zeros((N-1)^2, (N-1)^2);
My = 1:N-1;
Mx = 1:N-1:(N-1)*(N-2)+1;
for i = 1:N-1
    index = My + (i-1)*(N-1);
    L(index, index) = L(index, index) + D2;
    index = Mx + i-1;
    L(index, index) = L(index, index) + D2;
end
figure, clf, spy(L), drawnow
% solve for u:
tic,
u = L\f;  toc   % solve problem and watch the clock

% Reshape long 1D results onto 2D grid:
uu = zeros(N+1,N+1); uu(2:N,2:N) = reshape(u,N-1,N-1);
[xx,yy]= meshgrid(x,y);
value = uu(N/4+1,N/4+1);

% Interpolate to finer grid annd plot:

[xxx, yyy] = meshgrid(-1:0.04:1,-1:0.04:1);
uuu = interp2(xx,yy,uu,xxx,yyy,'spline');
figure(2), clf, mesh(xxx,yyy,uuu), colormap([0 0 0]);
xlabel x, ylabel y, zlabel u
text(0.4,-0.3,-0.3, sprintf('u(2^{-1/2}, 2^{-1/2}) = %14.11f', value));