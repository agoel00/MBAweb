function M = mod_multinomialdoublystochasticfactory(n)
% Manifold of n-by-n doubly-stochastic matrices with positive entries.
%
% function M = mod_multinomialdoublystochasticfactory(n) % BM modified
%
% M is a Manopt manifold structure to optimize over the set of n-by-n
% matrices with (strictly) positive entries and such that the entries of
% each column and each row sum to one.
%
% Points on the manifold and tangent vectors are represented naturally as
% symmetric matrices of size n. The Riemannian metric imposed on the
% manifold is the Fisher metric, that is, if X is a point on the manifold
% and U, V are two tangent vectors:
%
%     M.inner(X, U, V) = <U, V>_X = sum(sum(U.*V./X)).
%
% The  retraction here provided is only first order. Consequently, the
% slope test in the checkhessian tool is only valid at points X where the
% gradient is zero. Furthermore, if some entries of X are very close to
% zero, this may cause numerical difficulties that can also lead to a
% failed slope test. More generally, it is important the the solution of
% the optimization problem should have positive entries, sufficiently far
% away from zero to avoid numerical issues.
%
% The file is based on developments in the research paper
% A. Douik and B. Hassibi, "Manifold Optimization Over the Set
% of Doubly Stochastic Matrices: A Second-Order Geometry"
% ArXiv:1802.02628, 2018.
%
% Link to the paper: https://arxiv.org/abs/1802.02628.
%
% Please cite the Manopt paper as well as the research paper:
% @Techreport{Douik2018Manifold,
%   Title   = {Manifold Optimization Over the Set of Doubly Stochastic
%              Matrices: {A} Second-Order Geometry},
%   Author  = {Douik, A. and Hassibi, B.},
%   Journal = {Arxiv preprint ArXiv:1802.02628},
%   Year    = {2018}
% }
%
% See also: multinomialsymmetricfactory multinomialfactory
    
    % This file is part of Manopt: www.manopt.org.
    % Original author: Ahmed Douik, March 06, 2018.
    % Contributors: Nicolas Boumal
    % Change log:
    %
    %    Apr. 24, 2018 (AD):
    %        Changed pinv() to a particular solution to the equation.
    %
    %    July 24, 2018 (AD):
    %        A bugfix related to the pinv() change, with effects in many places.
    %
    %    Sep.  6, 2018 (NB):
    %        Removed M.exp() as it was not implemented.
    
    e = ones(n, 1);
    DSmaxiter = min(2*n, 1000); % BM: DSmaxiter is set to 1000 to reduce computational overhead. 
    
    M.name = @() sprintf('%dx%d doubly-stochastic matrices with positive entries', n, n);
    
    M.dim = @() (n-1)^2;
    
    % Fisher metric
    M.inner = @iproduct;
    function ip = iproduct(X, eta, zeta)
        ip = sum((eta(:).*zeta(:))./X(:));
    end
    
    M.norm = @(X, eta) sqrt(M.inner(X, eta, eta));
    
    M.dist = @(X, Y) error('multinomialdoublystochasticfactory.dist not implemented yet.');
    
    % The manifold is not compact as a result of the choice of the metric,
    % thus any choice here is arbitrary. This is notably used to pick
    % default values of initial and maximal trust-region radius in the
    % trustregions solver.
    M.typicaldist = @() n;
    
    % Pick a random point on the manifold
    M.rand = @random;
    function X = random()
        Z = abs(randn(n, n));     % Random point in the ambient space

        X = mod_doubly_stochastic(Z, DSmaxiter); % Projection onto the manifold
    end
    
    % Pick a random vector in the tangent space at X.
    M.randvec = @randomvec;
    function eta = randomvec(X) % A random vector in the tangent space
        % A random vector in the ambient space
        Z = randn(n, n) ;
        % Projection of the vector onto the tangent space
        % A = [eye(n) X ; X' eye(n)] ;
        b = [sum(Z,2) ; sum(Z,1)'] ;
        
        [alpha, beta] = mylinearsolve(X, b, n);
        
        eta = Z - (alpha*e' + e*beta').*X ;
        % Normalizing the vector
        nrm = M.norm(X, eta);
        eta = eta / nrm;
    end
    
    % Projection of vector eta in the ambient space to the tangent space.
    M.proj = @projection;
    function etaproj = projection(X, eta) % Projection of the vector eta in the ambeint space onto the tangent space
        % A = [eye(n) X ; X' eye(n)] ;
        b = [sum(eta,2) ; sum(eta,1)'] ;
        
        [alpha, beta] = mylinearsolve(X,b,n);
        
        etaproj = eta - (alpha*e' + e*beta').*X ;
    end
    
    M.tangent = M.proj;
    M.tangent2ambient = @(X, eta) eta;
    
    % Conversion of Euclidean to Riemannian gradient
    M.egrad2rgrad = @egrad2rgrad;
    function rgrad = egrad2rgrad(X, egrad) % projection of the euclidean gradient
        mu = (X.*egrad) ;
        
        % A = [eye(n) X ; X' eye(n)] ;
        
        b = [sum(mu,2) ; sum(mu,1)'] ;
        
        [alpha, beta] = mylinearsolve(X, b, n);
        
        rgrad = mu - (alpha*e' + e*beta').*X ;
    end


    % Conversion of Euclidean to Riemannian Hessian
    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(X, egrad, ehess, eta)

        % computing the directional derivative of the Riemannian
        % gradient
        gamma = egrad.*X;
        gammadot = ehess.*X + egrad.*eta;
        
        A = [eye(n) X ; X' eye(n)];
        b = [sum(gamma, 2) ; sum(gamma, 1)'];
        bdot = [sum(gammadot, 2) ; sum(gammadot, 1)'];

        [alpha, beta] = mylinearsolve(X, b, n);

        [alphadot, betadot] = mylinearsolve(X, bdot - [eta*beta; eta'*alpha], n);
        
        S = (alpha*e' + e*beta');
        deltadot = gammadot - (alphadot*e' + e*betadot').*X- S.*eta;

        % projecting gamma
        delta = gamma - S.*X;

        % computing and projecting nabla
        nabla = deltadot - 0.5*(delta.*eta)./X;
        rhess = projection(X, nabla);
    end
    
    % First-order retraction
    M.exp = @retraction;
    M.retr = @retraction;
    function Y = retraction(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        Y = X.*exp(t*(eta./X));    
        Y = mod_doubly_stochastic(Y, DSmaxiter); % BM
        Y = max(Y, eps);
    end
    
    % Miscellaneous manifold functions
    M.hash = @(X) ['z' hashmd5(X(:))];
    M.lincomb = @matrixlincomb;
    M.zerovec = @(X) zeros(n, n);
    M.transp = @(X1, X2, d) projection(X2, d);
    M.vec = @(X, U) U(:);
    M.mat = @(X, u) reshape(u, n, n);
    M.vecmatareisometries = @() false;
    
end

