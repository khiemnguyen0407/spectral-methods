function [nN, nE, T] = computeMapping(p,t)
%COMPUTEMAPPING  Mapping between local and global degrees of freedom.
% 
% The results of such mapping is stored in the matrix T. T(e,:) maps the
% local DOFs within the element to the global DOFs. For example T(e,1) = 5
% maps the first node of the element "e" to the global DOF number 5. T(e,2)
% = 6 maps the second node of the element to the global DOF number 6.
%
% In addition, the function returns nN as the total number of nodes and nE
% as the total number of elements of the finite element mesh.


% Number of nodes.
nN = size(p,2);
% Number of elements.
nE = size(t,2);
% Number of DOFs per node.
nDim = size(p,1);
% T maps the local DOFs of each element to the system DOFs.
T = zeros(nE,nDim*size(t,1));
for e = 1:nE
    % Map the local DOFs within element to the global.
    T(e,:) = t(:,e)';
end