function B = mod_doubly_stochastic(C, maxiter, checkperiod)
% Project a matrix to the doubly stochastic matrices (Sinkhorn's algorithm)
%
% function B = mod_doubly_stochastic(C)
%
% Given an element-wise non-negative matrix A of size nxn, returns a
% doubly-stochastic matrix B of size nxn by applying Sinkhorn's algorithm
% to A.
% 
% maxiter (optional): strictly positive integer representing the maximum 
%   number of iterations of Sinkhorn's algorithm. 
%   The default value of maxiter is n^2.
% mode (optional): Setting mode = 1 changes the behavior of the algorithm 
%   such that the input A is an n x p matrix with AA' having 
%   element-wise non-negative entries and the output B is also n x p
%   such that BB' is a doubly-stochastic matrix. The default value is 0.

% The file is based on developments in the research paper
% Philip A. Knight, "The Sinkhorn–Knopp Algorithm: Convergence and 
% Applications" in SIAM Journal on Matrix Analysis and Applications 30(1), 
% 261-275, 2008.
%
% Please cite the Manopt paper as well as the research paper.

% This file is part of Manopt: www.manopt.org.
% Original author: David Young, September 10, 2015.
% Contributors: Ahmed Douik, March 15, 2018.
% Change log:


    n = size(C, 1);
    tol = eps(n); % BM
    if ~exist('maxiter', 'var') || isempty(maxiter)
        maxiter = n^2;
    end
    if ~exist('checkperiod', 'var') || isempty(checkperiod)
        checkperiod = 100;
    end
        
    % Original
    % iter = 1;
    % d_1 = 1./sum(C);
    % d_2 = 1./(C * d_1.');
    % gap = -1;
    % while iter < maxiter
    %     iter = iter + 1;
    %     row = d_2.' * C;
    %     gap = max(abs(row .* d_1 - 1));%PJ
    %     if  gap <= tol
    %     %if  max(abs(row .* d_1 - 1)) <= tol
    %         break;
    %     end
    %     d_1 = 1./row;
    %     d_2 = 1./(C * d_1.');
    % end

    % PJ
    ones_n = ones(n,1);
    dd_1 = 1./sum(C);
    d_2 = 1./(C * dd_1.');
    d_2_prev = d_2;
    iter = 0;
    gap = Inf;
    if any(isinf(d_2)) || any(isnan(d_2))
        fprintf('Nan or Inf occured! DS projection iter %d, error %e \n', iter, gap);
    end
    
    while iter < maxiter
        iter = iter + 1;
        
        d_2 = ones_n./(C*(ones_n./(C'*d_2)));
        if any(isinf(d_2)) || any(isnan(d_2))
            fprintf('Nan or Inf occured! DS projection iter %d, error %e \n', iter, gap);
            d_2 = d_2_prev;
            break;
        end

        if mod(iter,checkperiod)==0
            row = d_2'*C;
            d_1 = ones_n'./row;
            d_2 = 1./(C * d_1.');
            row = d_2'*C;
            gap = max(abs(row .* d_1 - 1));
            if isnan(gap)
                break;
            end
            if  gap <= tol
                break;
            end
        end
        d_2_prev = d_2;
    end
    fprintf('DS projection iter %d, error %e \n', iter, gap);
    d_1 = ones_n./(C'*d_2);
    B = (d_2*d_1').*C;
end
