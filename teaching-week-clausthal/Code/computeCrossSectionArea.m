function A = computeCrossSectionArea(x,option)
%COMPUTEAREA  Cross-sectional area of the bar as function of coordinate x.
%
% This function returns the area of the cross-section of the bar as a
% function of x. The input "option" allows the user to choose different
% predefined function.


switch option
    case 1
        A = 2-x;
    case 2
        A = 1;
end