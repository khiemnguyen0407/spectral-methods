% p21.m - eigenvalues of Mathiew operator -u_xx + 2q cos(2x) u

close all;
scrsz = get(groot,'ScreenSize');
figure('position', [100   100   4/5*scrsz(3)  4/5*scrsz(4)]); 

N = 2^8; 
h = 2*pi/N;
x = h*(1:N);
D2 = toeplitz([-pi^2/(3*h^2)-1/6, -0.5*(-1).^(1:N-1)./sin(h*(1:N-1)/2).^2]);

qq = 0: 0.5 : 15; 
data = [];
for q = qq
    eigVector = sort(eig(-D2 + 2*q*diag(cos(2*x))))';
%     eigVector = sort(eig(-D2 - 2*q*diag(cosh(2*x))))';
    data = [data; eigVector(1:11)]; %#ok
end

subplot(1,2,1)
set(gca, 'colororder', [0 0 0], 'linestyleorder', '-|--'), hold on
plot(qq, data, 'linewidth', 0.8)
xlabel('q'), ylabel('\lambda')
axis([qq(1)  qq(end)  -24  32])
set(gca, 'ytick', -24:4:32)


% %% Analytical eigenvalues extracted from the book Abramowitz and Stegun.
% a0 = -qq.^2/2 + (7/128)*qq.^4 - 29/2304*qq.^6 + 68687/18874368*qq.^8;
% b1 = 1 - qq - 1/8*qq.^2 + 1/64*qq.^3 - 1/1536*qq.^4 - 11/36864*qq.^5 ...
%     + 49/589824*qq.^6 - 55/9437184*qq.^7 - 83/35389400*qq.^8;
% subplot(1,2,2),
% plot(qq, b1, 'k-', qq, b1, 'k--');