% Exercise 8.2. Write a program to perform a study of the relative
% efficiencies of "cheb" and "chebfft" as a function of N. Do not count the
% time by cheb to form the differentiation matrix D_N, just the time taken
% by multiply D_N by a vector.

Nvec = 2.^(8:13);
tlist = zeros(length(Nvec),2);
tic
for i = 1:length(Nvec)
    [D, x] = cheb(Nvec(i));
    u = exp(x).*sin(5*x);
    
    % Measure the time of computation by using differentiation matrix.
    tstart1 = tic;
    uprime = D*u; %#ok
    tlist(i,1) = toc(tstart1);
    
    % Measure the time of compuation by using chebfft.
    tstart2 = tic;
    uprime = chebfft(u);
    tlist(i,2) = toc(tstart2);
end