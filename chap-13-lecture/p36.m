% p36.m - Laplace eq. on [-1,1]x[-1,1] with nonzero BC's
% close all;
scrsz = get(groot,'ScreenSize');
figure('position', [100   100   4/5*scrsz(3)  4/5*scrsz(4)]); 

% Set up grid and 2D Laplacian, boundary points included:
N = 24; [D,x] = cheb(N); y = x;
[xx,yy] = meshgrid(x,y); xx = xx(:); yy = yy(:);
D2 = D^2; I = eye(N+1); L = kron(I,D2) + kron(D2,I);

% Impose boundary conditions by replacing appropriate rows of L:
%------------------------------------------------------------------
% b stores the indices corresponding to the points on the boundaries.
b = find(abs(xx)==1 | abs(yy)==1);
% Set homogeneous boundary condition for the entire boundary: u = 0. The
% nonhomogeneous boundary conditions is handled by modifying L matrix after
% this step.
L(b,:) = zeros(4*N,(N+1)^2);
% Set the nonhomogeneous Dirichlet condition by setting the components for
% the corresponding nodes to 1. The right-hand side will give the
% constraint value for these nodes.
L(b,b) = eye(4*N);

% When (xx(b) < 0) holds true, (xx(b) == 1) returns 0 so that the second
% term does not produce effect. Vice versa, when (xx(b) == 1) holds, (xx(b)
% < 0) returns 0, thus the first term is not effected.
rhs = zeros((N+1)^2,1);
% rhs(b) = (yy(b)==1).*(xx(b)<0).*sin(pi*xx(b)).^4 ...
%     + 0.2*(xx(b)==1).*sin(3*pi*yy(b));

% Alternatively, we can track down the indices of the nodes corresponding
% t
bc1 = find((yy == 1) & (xx < 0));
bc2 = find(xx == 1);
rhs(bc1) = sin(pi*xx(bc1)).^4;
rhs(bc2) = 0.2*sin(3*pi*yy(bc2));
% Solve Laplace equation, reshape to 2D, and plot:
u = L\rhs; uu = reshape(u,N+1,N+1);
[xx,yy] = meshgrid(x,y);
[xxx,yyy] = meshgrid(-1:.04:1,-1:.04:1);
uuu = interp2(xx,yy,uu,xxx,yyy,'spline');
clf, subplot('position',[.1 .4 .8 .5])
mesh(xxx,yyy,uuu), colormap([0 0 0])
axis([-1 1 -1 1 -.2 1]), view(-20,45)
text(0,.8,.4,sprintf('u(0,0) = %12.10f',uu(N/2+1,N/2+1)))