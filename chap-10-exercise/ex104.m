%EX106 -- Exercise 10.6
% Solve the 

close all;

L = pi;
N = 2^8;
h = L/(N/2);
% Scale factor transforming the Fourier Transform on [-pi,pi] --> [-L,L] 
scale = L/pi;
x = scale*(2*pi/N)*(-N/2:N/2-1);
dt = 0.001;
T = 3;
nStep = round(T/dt);
nSlide = 50;
tSlide = 1:round(nStep/nSlide):nStep+1;
nplot = length(tSlide); 
t = zeros(nplot,1);
% Initial condition.
iu = zeros(1,N);
iu(x <= 0) = sin(x(x<=0).^2);
iu(x > 0) = 0;

% Solution vector.
u = zeros(nplot, N);
u(1,:) = iu;
% Wave vectors.
ko = [0:N/2-1  0 -N/2+1:-1];
ke = [0:N/2      -N/2+1:-1];
epsilon = 0.01;
E = exp(-0.5*epsilon*dt*ke.^2);
E2 = E.^2;
g = -1i*ko*dt;

% 4th-order Runge-Kutta method for the wave equation discretized in space.
uhat = fft(u(1,:));
wb = waitbar(0, 'please wait ...');
j = 2;
for n = 1:nStep
    b1 = real( ifft(uhat) );
    a1 = g.*fft( b1.^2 );
    
    b2 = real( ifft(E.*(uhat + 0.5*a1)) );
    a2 = g.*fft( b2.^2 ); 
    
    b3 = real( ifft(E.*uhat + 0.5*a2) );
    a3 = g.*fft( b3.^2 );
    
    b4 = real( ifft(E2.*uhat + E.*a3) );
    a4 = g.*fft( b4.^2 );
    
    % Update solution in Fourier space at next time step.
    uhat = E2.*uhat + (E2.*a1 + 2*E.*(a2 + a3) + a4)/6;
    
    if ~isempty(find(isnan(uhat),1))
        disp(['The solution is unstable up to time T = ', num2str(n*dt)]);
        delete(wb);
        return
    end
    if ~isempty(find(n+1==tSlide,1))
        u(j,:) = real( ifft(uhat) );
        t(j) = n*dt;
        j = j + 1;
        waitbar(n/nStep,wb, [num2str(round(n/nStep*100)) '%']);
    end
end
% If the solution is stable up the last time instant, display "success".
if j > size(u,1)
    disp(['The solution is stable up to this step T = ', num2str(t(end)) '.']);
    delete(wb);
end

%% Plot initial condition and last time step.
close all;
scrsz = get(groot,'ScreenSize');
figure('position', [100   100   4/5*scrsz(3)  4/5*scrsz(4)]); 
waterfall(x, t, u), colormap(zeros(1,3));
% axis([-pi,  pi, min(u(end,:)), max(u(end,:))]), view(10,70), grid off
xlabel x, ylabel t, zlabel u