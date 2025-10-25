%P9  polynomial interpolation in equispaced and Chebyshev points

N = 32;
xx = linspace(-1, 1, 1001);
clf
for i = 1:3
    if i == 1
        s = 'equispaced points';
        x = -1 + 2*(0:N)/N;
    end
    if i == 2
        s = 'Chebyshev points';
        x = cos(pi*(0:N)/N);
    end
    if i == 3
        s = 'Gauss points';
        [x, ~] = lgwt(N-1, -1, 1);
        x = transpose(x(:));
        x = [-1, x, 1]; %#ok<AGROW>
    end
    subplot(3,1,i)
    u = 1./(1 + 16*x.^2);
    v = 1./(1 + 16*xx.^2);
    
    % One way to construct the Lagrange interpolation is to use polyfit.
%     p = polyfit(x, u, N);           % interpolation
%     q = polyval(p, y);              % evaluation of interpolant
    
    %-----------------------------------------------------
    % However, the polyfit requires inverse to obtain the polynomial
    % coefficients. When N is quite large, the polyfit ends up with
    % computing an inverse of a badly-condition matrix eventhough that
    % matrix is completely invertible. Another alternative is to construct
    % the Lagrange interpolation explicitly as follows.
    shape_val = ones(length(u), length(xx));
    for j = 1: length(u)
        for k = setdiff(1:length(u), j)
            shape_val(j, :) = shape_val(j, :) .* (xx - x(k)) ./ (x(j) - x(k));
        end
    end
    
    % Then we can compute the interpolation at coordinate vector y as
    q = u * shape_val;        % u MUST be a row vector.
    
    plot(x, u, '.', 'markersize', 13)
    line(xx, q, 'linewidth', 0.8, 'color','k');
    if (i == 1)
        axis([-1.1, 1.1, -5, 10])
    end
    axis([-1.1 1.1 -1 1.5]), title(s)
    error = norm(v - q, inf);
    text(-0.5, -0.5, ['max error = ', num2str(error)]);
end