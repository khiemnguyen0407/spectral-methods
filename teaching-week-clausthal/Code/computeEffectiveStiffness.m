function [meanStress, Ceffective] = computeEffectiveStiffness(strain0,radius,varargin)
%COMPUTEEFFECTIVESTIFFNESS  Consistent macroscopic tangent and mean stress
% over the RVE.
%
% This function basically solves the microscopic boundary value problem
% based on the equivalent Lippmann-Schwinger equation. The microscopic
% strain is obtained as the solution of the microscopic BVP, the
% macroscopic strain is the average value of the microscopic strain over
% the RVE volume. In the same manner, the macroscopic stress is obtained by
% averaging the microscopic stress over the RVE volume.
% 
% In the strain-driven homogenization scheme, the average strain or the
% macroscopic strain is considered as given in the microscopic BVP and is
% supplied from the macroscopic-level computation. In such strain-driven
% scheme, we may interpret the microscopic BVP as a black-box constitutive
% law that establishes relation between the macroscopic strain and
% macroscopic stress. This black-box works as follows: (i) The input is the
% macroscopic strain, given in this function by the variable strain0, (ii)
% the output are macroscopic stress which can be also called mean stress
% ("mean" stands for average) and the consistent macroscopic tangent.
%
% Radius denotes the half wavelength \lambda/2. The RVE domain is defined
% in the form [-radius, radius] = (-\lambda/2, \lambda/2).


microstructureOpt = 1;
if nargin > 1
    microstructureOpt = varargin{1};
end


% Problem domain is [-radius, radius].
% radius = pi;
% Number of grid points.
N = 2^8;
% Distance between grid points.
h = radius/(N/2);
% Coordinates of collocation points.
scale = radius/pi;
x = scale * (2*pi/N) * (-N/2+1 : N/2);

% PROBLEM SETTING.
% Compute C matrix at collocation points.
switch microstructureOpt
    case 1
        C = 3/2 + sin(2*pi/radius*x);
    case 2
        C = 3/2 + sin(50*pi*x);
    otherwise
        C = 1;
end

% Wave number used in Discrete Fourier Transform.
k = [0:N/2,  -N/2+1 : -1]/scale;
% Reference Medium.
C0 = max(C)+2;
% Green operator in Fourier space.
fGamma = -1/C0;


% SOLUTION OF LIPPMANN-SCHWINGER EQUATION.
% Tolerance for stopping the iterative scheme.
TOL = 1e-10;
% Maximum number of iteration.
maxIter = 1000;
% Initialize the search variable.
strain = strain0*ones(size(x));
stress = C .* strain;

% stress = C .* (3 + strain);
% Iterative scheme resorting  modified fixed-point method.
for iter = 1:maxIter
    % Compute polarization.
%     tau = stress - C0 * strain;
    % Stress in Fourier space.
    fstress = fft(stress);
    % Convergence test.
    criterionNum = N*norm(sum(k .* fstress)) / norm(fstress(1));
    if criterionNum < TOL
        % disp(['Solver stops at step i = ', num2str(iter)]); break;
        break
    end
    % If the solution does not converge, improve the solution.
    % -------------------------------------
    %     fstrain = fGamma * fft(tau);
    
    % Updated solution in Fourier space.
    fstrain = fft(strain) + fGamma * fstress;
    fstrain(1) = N * strain0;
    % Updated solution in physical space.
    strain = ifft(fstrain);
    % Compute the stress for next step.
    stress = C .* strain;
    % Nonlinear constitutive law.
%     stress = C .* (3 + strain);
end

% SOLUTION FOR FLUCTUATIVE PART.
% Initialize the search variable.
alpha = 0*ones(size(x));
% Iterative scheme resorting to fixed point method.
for iter = 1:maxIter
    % Updated solution in Fourier space.
    falpha = ( fGamma .* fft((C - C0).*(1+alpha)) );
    % Updated solution in physical space.
    falpha(1) = 0;
    alpha = ifft(falpha);
end

% Averaged stress over the RVE.
vol = 2*radius;
meanStress = 1/vol * ( trapz(x,stress) + h*mean(stress([end, 1])) );
% Effective stiffness.
beta = C .* (1 + alpha);
Ceffective = 1/vol * ( trapz(x,beta) + h*mean(beta([end, 1])) );