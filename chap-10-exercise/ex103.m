%EX104 -- Exercise 10.4: Solve the wave equation using the Runge-Kutta
% method

close all;


N = 64;
[D, x]= cheb(N);
D2 = D^2;

dt = 3.5e-6;
% Initial condition.
v = zeros(N+1,1);

% Storage of solution.
T = 3.5;
nstep = round(T/dt);
nSlide = 500;
tSlide = 1:round(nstep/nSlide):nstep+1;
nplots = length(tSlide);
t = zeros(nplots,1);


u = zeros(nplots,N+1);
u(1,:) = v';
j = 2;
wb = waitbar(0, 'please wait...');
for n = 1:nstep
    waitbar(n/nplots,wb, ...
        ['process ... ' , num2str(round(n/nstep,2)*100), '%']);
    w = v;
    a = D2*w + exp(w);
    w = v + dt/2*a;
    b = D2*w + exp(w);
    w = v + dt/2*b; 
    c = D2*w + exp(w);
    w = v + dt*c;
    d = D2*w + exp(w);
    v = v + dt/6*(a + 2*b + 2*c + d);
    v([1  end]) = 0;
    if ~isempty(find(n+1 == tSlide, 1))
        u(j,:) = v';
        t(j) = n*dt;
        j = j + 1;
    end
end

%% VISUALIZE THE SOLUTION
waterfall(x,t,u), colormap([0 0 0]), view(-20,25)
xlabel x, ylabel t, grid off
ylabel t, zlabel u, close(wb);
