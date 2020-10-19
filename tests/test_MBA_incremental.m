function test_MBA_incremental()

    randomize_seed = 21;
    rng(randomize_seed);

    % src-tgt data
    src_lang = 'en'; % for X
    tgt_lang = 'es'; % for Z

    fprintf('Source: %s, Target: %s\n', src_lang, tgt_lang);


    % % Load sample en-es data.
    datafile = 'muse_data_en_es.mat'; % This file should be downloaded from the link supplied in README.txt.
    src_tgt_data = load(datafile);
    dataorg = src_tgt_data.data;
    

    % Vocab
    X = dataorg.X; % full source vocabulary typically around 200k.
    Z = dataorg.Z; % full target vocabulary typically around 200k.


    % Test data
    % 
    % Xtest: test source embeddings, say n words with d features. n-by-d matrix.
    % Ztest: test target embeddngs, say m words  with d features. m-by-d matrix.
    % Ytest: n-by-m sparse matrix. Entry (i,j) is 1 if x_i (ith source) and z_j (jth target) are translations.

    Xtest= dataorg.Xtest;
    Ztest = dataorg.Ztest;
    Ytest = dataorg.Ytest;


    %% Number of data points taken
    problem.Xfull = X;
    problem.Xte = Xtest;
    problem.Zte = Ztest;
    problem.Yte = Ytest;
    
        
    options.method = 'cg';
    options.checkperiod = inf; % Inf implies: basically do not compute test in between iterations
    
    
    %% Intialize
    numInitArray = [1000 2500 5000 10000 20000]; % Go from 1000 to 20 000 points.
    xinit = [];

    options.maxiter = 2000; % only for the first meta iteration.
    
    % For loop over the number of points.
    accArray = nan(length(numInitArray), 2);
    myeps = 1e-5;
    
    for ii = 1 : length(numInitArray)
        
        numInit = numInitArray(ii);
        
        fprintf('For meta iteration: %d     numInit: %d\n', ii, numInit);
        % Vocab train data
        problem.X = X(1: numInit,:);
        problem.Z = Z(1: numInit,:);
        [x, infos, options] = MBA(problem, xinit, options);
        x
        accArray(ii,:) = [numInit  x.accuracy];
        if ii < length(numInitArray)
            numInitNext = numInitArray(ii + 1);
            Y = zeros(numInitNext, numInitNext);
            Y(1:numInit, 1:numInit) = x.Y;
            xinit = mod_doubly_stochastic(Y + myeps*rand(numInitNext, numInitNext), 1000);
            options.maxiter = 200; % for subsequent runs, less iterations suffice
            clear Y x
        end

    end

    accArray
    
    proposed_src_tgt_out.x = x;
    proposed_src_tgt_out.infos = infos;
    proposed_src_tgt_out.options = options;
    proposed_src_tgt_out.accArray = accArray;
    
end
