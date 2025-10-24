function fe = computeMechanicalForce(p,poly,varargin)
%COMPUTEMECHANICALFORCE  Element external force derived from the given
% distributed force provided by the function "computeForce.m".
%
% This function computes the element external force contribution in the finite
% element context. That is, it evaluates the integral of the external force
% multiplied by a virtual displacement on an element. The virtual
% displacement is interpolated by polynomial that is defined by the input
% poly.


forceOpt = 1;
areaOpt = 1;
if nargin > 2
    forceOpt = varargin{1};
    if nargin > 3
        areaOpt = varargin{2};
    end
end

fe = zeros(size(poly,1),1);
for i = 1:size(poly,1)
    func = @(r) computeForce(0.5*(1-r)*p(1) + 0.5*(1+r)*p(end), forceOpt)...
        .* polyval(poly(i,:),r) .* computeCrossSectionArea(0.5*(1-r)*p(1) + 0.5*(1+r)*p(end),areaOpt);
    fe(i) = integral(func,-1,1);
end