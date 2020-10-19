function [myuout, infos, options] = MBA(problem, xinit, options)
% We solve the optimization problem:
% 0.5||Yt*XXt*Y - ZZt||^2 + 0.5||Y*ZZt*Yt - XXt||^2 + 0.5 lambda*||Y||^2
%
% Y is a doubly stochastic matrix that we intend to learn.
%
%
% Please cite the Manopt paper as well as the research paper:
%     @InProceedings{mishra2011dist,
%       Title        = {Geometry-aware domain adaptation for unsupervised alignment of word embeddings},
%       Author       = {Jawanpuria, P. and Meghwanshi, M. and Mishra, B.},
%       Booktitle    = {{Accepted to the Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics}},
%       Year         = {2020}
%     }
    
    X = problem.X;
    Z = problem.Z;
    
    
    N = size(X, 1);
    
    % Local defaults for options
    localdefaults.maxiter = 250; % Max iterations.
    localdefaults.verbosity = 2; % Default: show the output.
    localdefaults.lambda = 0; % Regularization parameter
    localdefaults.tolgradnorm = 1e-6; % Absolute tolerance on Gradnorm.
    localdefaults.maxinner = 30; % Max inner iterations for the tCG step.
    localdefaults.tolrelgradnorm = 1e-10; % Gradnorm/initGradnorm tolerance.
    localdefaults.method = 'CG'; % Default solver is conjugate gradient (CG).
    localdefaults.checkperiod = 50; % Compute test every checkperiod iterations.
    localdefaults.checkgradienthessian = false; % Check gradient and Hessian correctness.
    localdefaults.computetest = true; % Compute test by default.
    localdefaults.GW_embedding_normalize = false; % Normalization proposed by GW authors.
    localdefaults.GW_cov_normalize = false; % Normalization proposed by GW authors.
    
    % Initialization
    if ~exist('xinit', 'var')
        xinit = [];
    end
    
    
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);


    % GW Normalize if needed
    if options.GW_embedding_normalize
        X = X-mean(X);
        X = bsxfun(@times, X, 1./sqrt(sum(X.^2, 2)));
        Z = Z-mean(Z);
        Z = bsxfun(@times, Z, 1./sqrt(sum(Z.^2, 2)));
    end

    % GW Normalize covariances if needed
    if options.GW_cov_normalize
        Kx = X*X';
        Kz = Z*Z';
        normalize1 = mean(1-Kx(:))
        normalize2 = mean(1-Kz(:))
        X = X./sqrt(normalize1);
        Z = Z./sqrt(normalize2);
        clear Kx Kz
    end


    BtB = Z'*Z;
    DtD = X'*X;
    normBtB = norm(BtB(:));
    normDtD = norm(DtD(:));
    
    % Doubly stochastic manifold factory.
    problem.M = mod_multinomialdoublystochasticfactory(N); % N-by-N doubly stochastic matrices
    
    % Regularization if required.
    lambda = options.lambda;
    
    % If asked to compute test accuracy.
    if options.computetest
        options.statsfun = @mystatsfun;
    end
    
    % Optimization function handles
    problem.cost = @cost;
    problem.egrad = @egrad;
    problem.ehess = @ehess;
    
    function [f, store] = cost(Y, store)
        if ~isfield(store, 'A')
            A = Y'*X;
            store.AtA = A'*A;
            store.A = A;
            
            C = Y*Z;
            store.C = C;
            store.CtC = C'*C;
            
            store.XtYZ = X'*C;
            store.BtA = (store.XtYZ)';% Z'*Y'*X;
            store.DtC = store.XtYZ;% X'*Y*Z
        end
        
        AtA = store.AtA;
        BtA = store.BtA;
        
        CtC = store.CtC;
        DtC = store.DtC;
        
        % 0.5||Yt*XXt*Y - ZZt||^2 == 0.5|| AAt  - BBt ||^2
        f = 0.5*(norm(AtA(:))^2 +  normBtB^2) - norm(BtA(:))^2 ;
        
        % 0.5||Y*ZZt*Yt - XXt||^2 == 0.5|| CCt  - DDt ||^2
        f = f ...
            + 0.5*(norm(CtC(:))^2 +  normDtD^2) - norm(DtC(:))^2;
        
        % 0.5 lambda*||Y||^2
        f = f + 0.5*lambda* norm(Y, 'fro')^2;
    end
    
    function [g, store] = egrad(Y, store)
        if ~isfield(store, 'A')
            [~, store] = cost(Y, store);
        end
        A = store.A;
        AtA = store.AtA;
        
        C = store.C;
        CtC = store.CtC;
        
        XtYZ = store.XtYZ;
        XXtYZZt = X*(XtYZ*Z');
        
        % Gradient for 0.5||Yt*XXt*Y - ZZt||^2 is
        % 2*(X*(AtA*A' - BtA'*B')).
        t1 = X*(AtA*A') - XXtYZZt;% X*(AtA*A' - BtA'*B');
        g = 2*t1 ;
        
        % Gradient for 0.5||Y*ZZt*Yt - XXt||^2
        % is 2*((C*CtC - D*DtC)*Z').
        t2 = (C*CtC)*Z' - XXtYZZt;% (C*CtC - D*DtC)*Z'
        g = g + 2*t2;
        
        % Gradient for 0.5 lambda*||Y||^2 is lambda*Y.
        g = g + lambda*Y;
    end

    symm2 = @(A) (A + A');

    function [gdot, store] = ehess(Y, Ydot, store)
        if ~isfield(store, 'A')
            [~, store] = cost(Y, store);
        end
        A = store.A;
        AtA = store.AtA;
        
        C = store.C;
        CtC = store.CtC;
        
        Adot =  Ydot'*X;
        Cdot =  Ydot*Z;
        XtYdotZ = X'*(Cdot);
        XXtYdotZZt =  X*(XtYdotZ*Z');
        
        % Gradientdot for 0.5||Yt*XXt*Y - ZZt||^2 is
        t1dot = X*(symm2(Adot'*A)*A' + AtA*Adot') - XXtYdotZZt;
        gdot = 2*t1dot ;
        
        % Gradientdot for 0.5||Y*ZZt*Yt - XXt||^2
        t2dot = (Cdot*CtC + C*symm2(Cdot'*C))*Z' - XXtYdotZZt;
        gdot = gdot + 2*t2dot;
        
        % Gradientdot for 0.5 lambda*||Y||^2
        gdot = gdot + lambda*Ydot;
    end

    
    
    function [stats, store] = mystatsfun(problem, Y, stats, store)
        if mod(stats.iter + 1, options.checkperiod) == 0
            
            W = uf(problem.X'*(Y*problem.Z));
            
            mymetrics = computeCSLSmetric(problem.Xte, problem.Xfull, problem.Zte, problem.Yte, W);
            if options.verbosity
                fprintf('Accuracy on test set: %g \n', 100*(mymetrics));
            end
            stats.accuracy = 100*mymetrics;
        else
            stats.accuracy = -1;
        end
    end
    
    if options.checkgradienthessian
        % Check correctness of gradient and Hessian
        checkgradient(problem);
        checkhessian(problem);
        pause;
    end
    
    if strcmpi('TR', options.method)
        % Riemannian trustregions
        [Yopt, ~, infos] = trustregions(problem, xinit, options);
        
    elseif strcmpi('SD', options.method)
        % Riemannian steepest descent
        [Yopt, ~, infos] = steepestdescent(problem, xinit, options);
        
    elseif strcmpi('CG', options.method)
        %         % Riemannian conjugategradients
        %         options.beta_type = 'H-S';
        %         options.linesearch = @linesearch;
        %         options.ls_contraction_factor = .2;
        %         options.ls_optimism = 1.1;
        %         options.ls_suff_decr = 1e-4;
        %         options.ls_max_steps = 25;
        [Yopt, ~, infos] = conjugategradient(problem, xinit, options);
    end
    
    % Store output
    myuout.W = uf(problem.X'*(Yopt*problem.Z));
    myuout.Y = Yopt;
    
    if options.computetest
        final_mymetrics = computeCSLSmetric(problem.Xte, problem.Xfull, problem.Zte, problem.Yte, myuout.W);
        if options.verbosity
            fprintf('Final accuracy on test set: %e \n', 100*(final_mymetrics));
        end
        myuout.accuracy = 100*final_mymetrics;
    end
    myuout.numdatapoints = N;
end