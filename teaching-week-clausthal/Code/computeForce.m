function f = computeForce(x,option)
%COMPUTEFORCE   Body distributed force along the bar
% 
% The given distributed force is implemented as a function of coordinate x.
% The input "option" allows users to choose the predefined functions
% conveniently.

switch option
    case 1
        f = sin(2*pi*x);
    case 2
        f = 25*sin(2*pi*x);
    case 3
        f = 5*log(x+1);
    case 4
        f = x;
    otherwise
        f = 0;
end