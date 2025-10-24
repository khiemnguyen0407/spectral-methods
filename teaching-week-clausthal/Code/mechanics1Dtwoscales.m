%BARMNO  A materially-nonliner-only solver for the bar problem.
%
% BARMNO solves the one-dimensional equation of bar of nonlinear elastic
% material at infinitesimal strain. The equation reads
%   -d/dx (A(X) E(x) du/dx) = f  in [a, b].
%                    u = u_D on the Dirichlet boundary.
%          (+/-1)du/dx = t_N on the Neumann boundary.
%
% Cross-section of the bar and the distributed force are given as the
% m-file <computeArea.m> and <computeForce.m>, respectively. One can define
% different area expressions and the corresponding options by using the
% variable "areaOpt" and "forceOpt"


% INFORMATION FOR PERIODIC HOMOGENIZATION.
% Radius of the RVE.
radius = 1/50;
% % Option for choosing the microstructure.
microstructureOpt = 1;

% INITIALIZATION.
% Order of interpolation for each element.
% In this script we use the same interpoatlation order for every element.
nOrder = 1;
% The problem domain is [a, b].
a = 0; b = 1;
% Mesh generation.
nElements = 20;
p = a : (b-a)/(nOrder*nElements) : b;
t = zeros(nOrder+1,nElements);
for e = 1:nElements
    t(:,e) = ((e-1)*nOrder + 1 : e*nOrder + 1)';
end

% Dirichlet boundary condition.
% load dirichlet.mat
% Prescribed degrees of freedom.
prescribedDof = unique(find(p(1,:)== 0));
prescribedVal = zeros(size(prescribedDof));

% Neumann boundary condition.
% load neumann.mat
neumann = [];

nGauss = nOrder;
% COMPUTATION OF ITERATION-INDEPENDENT VARIABLES.
% Mapping between local and global DOFs, and areas of triangular elements.
[nNodes, nElements, T] = computeMapping(p,t);
[gW, gR] = createGaussPoint(nGauss);


% Physical coordinates of Gauss points.
gP = 1/2 *(1 - gR)*a + 1/2*(1 + gR)*b;
% Generate Lagrange interpolations defined on reference interval [-1,1]
[poly, dpoly] = generateLagrangeInterpolation(-1:2/nOrder:1);
% Array of B-matrix evaluted at Gauss points of each element.
B = zeros(nElements*nGauss,size(T,2));
for e = 1:nElements
    for g = 1:nGauss
        idx = nGauss*(e-1) + g;
        L = norm(p(t(end,e)) - p(t(1,e)));
        for j = 1:nOrder+1
            B(idx,j) = polyval(dpoly(j,:),gR(g)) * 2/L;
        end
    end
end
% Number of global degrees of freedom.
gDof = nNodes;
% Active degrees of freedom.
activeDof = setdiff(1:gDof,prescribedDof);

% Allocation of Matrices Involved in Solution Loop.
%-------------------------------------------
% Tangent stiffness.
k = zeros(gDof,gDof);
% Internal force.
intF = zeros(gDof,1);
% External force.
extF = zeros(gDof,1);

% Compute External Force Vector.
% Contribution of volume force.
areaOpt = 2;        % Option for choosing user-defined cross-section area.
forceOpt = 4;       % Option for choosing user-defined force function.
for e = 1:nElements
    extF(T(e,:)) = extF(T(e,:)) + computeMechanicalForce(p(t(:,e)),poly,forceOpt,areaOpt);
end

% Contribution of Neumann boundary condition.
tractionOpt = 1;
for j = 1:size(neumann,1)
    
end

% NEWTON-RAPHSON ITERATION.
%--------------------------------
% Initial solution vector.
u = zeros(gDof,1);
u(prescribedDof) = prescribedVal;
du = zeros(gDof,1);
du(prescribedDof) = 0;
% Gradient of solution at Gauss point.
gradient = zeros(2,1,nElements*nGauss);
% Tolerance for stopping the iteration.
TOL = 1e-8;
% Maximum number of iteration.
maxIter = 50;
for iter = 1:maxIter
    % Strain at Gauss points.
    for e = 1:nElements
        for g = 1:nGauss
            idx = nGauss*(e-1) + g;
            gradient(idx) = B(idx,:) * u(T(e,:));
        end
    end
    
    % Assembly of tangent stiffness.
    k(:) = 0;
    intF(:) = 0;
    for e = 1:nElements
        for g = 1:nGauss
            idx = nGauss*(e - 1) + g;
            [dual, tangent] = ...
                computeEffectiveStiffness(gradient(idx),radius,microstructureOpt);
            gX = 0.5*(1-gR(g))*p(t(1,e)) + 0.5*(1+gR(g))*p(t(end,e));
            k(T(e,:),T(e,:)) = k(T(e,:),T(e,:)) ...
                + gW(g) * B(idx,:)' * tangent * B(idx,:) ...
                * computeCrossSectionArea(gX,areaOpt);
            intF(T(e,:)) = intF(T(e,:)) ...
                + gW(g) * dual * B(idx,:)' * computeCrossSectionArea(gX,areaOpt);
        end
    end
    
    % Incremental solution.
    residual = extF - intF;
    du = k(activeDof,activeDof)\residual(activeDof);
    u(activeDof) = u(activeDof) + du;
    if norm(du)/norm(u) < TOL
        disp(['Solver stops at step i = ', num2str(iter)]);
        break
    end
end