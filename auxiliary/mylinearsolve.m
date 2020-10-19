function [alpha, beta] = mylinearsolve(X, b, n)
    % For large scale
    zeta = pcg(@myresidual, b, 1e-8, 100);
    function Ax =  myresidual(x)
        xtop = x(1:n,1);
        xbottom = x(n+1:end,1);
        Axtop = xtop + X*xbottom;
        Axbottom = X'*xtop + xbottom;
        Ax = [Axtop; Axbottom];
    end
    alpha = zeta(1:n, 1);
    beta = zeta(n+1:end, 1);
end  