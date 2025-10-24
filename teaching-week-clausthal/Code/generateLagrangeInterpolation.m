function [poly, varargout] = generateLagrangeInterpolation(p)
%GENERATELAGRANGEINTERPOLATION  Lagrange interpolations that are generated
% by least-squared fitting the the collocation points. The result
% interpolations are stored as polynomial in the MATLAB format.


% Loop over all possible interpolations.
np = length(p);
identity = eye(np);
poly = zeros(np,np);
dpoly = zeros(np,np-1);
for i = 1:np
    poly(i,:) = polyfit(p,identity(i,:),np-1);
end

if nargout > 1
    for i = 1:np
        dpoly(i,:) = polyder(poly(i,:));
    end
    varargout = {dpoly};
end