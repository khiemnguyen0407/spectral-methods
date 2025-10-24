% program20 - 2nd-order wave equation in 2D via FFT
%             compare program19.m

close all;
scrsz = get(groot,'ScreenSize');
% figure('position', [100   100   4/5*scrsz(3)  4/5*scrsz(4)]);

% Grid and initial data:
N = 48; x = cos(pi*(0:N)/N); y = x';
dt = 6.6/N^2;
[xx,yy] = meshgrid(x,y);
plotgap = round((1/3)/dt); dt = (1/3)/plotgap;
vv = exp(-40*((xx-0.4).^2 + yy.^2));
% The initial velocity is assumed to be almost zero and backward Euler
% formula is used for the first-time approximation. Otherwise, we should
% have special formula to compute u(x,-dt) and u(x,dt) at the first step.
vvold = vv;
% Time-stepping by leap frog formula:
[ay, ax] = meshgrid([0.56 0.06], [0.1 0.55]); clf
for n = 0:3*plotgap
    t = n*dt;
    if rem(n+0.5,plotgap) < 1 % plots at multiples of t = 1/3
        i = n/plotgap+1;
        % subplot('position', [ax(i) ay(i) 0.36 0.36]);
        figure(i)
        [xxx,yyy] = meshgrid(-1:1/16:1,-1:1/16:1);
        vvv = interp2(xx,yy,vv,xxx,yyy,'spline');
        mesh(xxx,yyy,vvv), axis([-1 1 -1 1 -0.15 1])
        colormap([0 0 0]), title(['t = ' num2str(t)], 'FontSize', 20), drawnow
    end
    uxx = zeros(N+1,N+1); uyy = zeros(N+1,N+1);
    ii = 2:N;
    for i = 2:N   % 2nd derivs wrt x in each row
        v = vv(i,:); V = [v fliplr(v(ii))];
        U = real(fft(V));
        W1 = real(ifft(1i*[0:N-1 0 1-N:-1].*U));	% diff wrt theta
        W2 = real(ifft(-[0:N 1-N:-1].^2.*U));       % diff^2 wrt theta
        uxx(i,ii) = W2(ii)./(1-x(ii).^2) - x(ii).*...
            W1(ii)./(1-x(ii).^2).^(3/2);
    end
    for j = 2:N   % 2nd derivs wrt y in each row
        v = vv(:,j); V = [v; flipud(v(ii))];
        U = real(fft(V));
        W1 = real(ifft(1i*[0:N-1 0 -N+1:-1]'.*U));  % diff wrt theta
        W2 = real(ifft(-[0:N 1-N:-1]'.^2.*U));      % diff^2 wrt theta
        uyy(ii,j) = W2(ii)./(1-y(ii).^2) - y(ii).*...
            W1(ii)./(1-y(ii).^2).^(3/2);
    end
    vvnew = 2*vv - vvold + dt^2*(uxx + uyy);
    vvold = vv; vv = vvnew;
end