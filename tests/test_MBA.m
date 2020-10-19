clear;
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





%% Number of data points taken for unsupervised alignment, typically we take 20k.
numInit = 20000; % 20k.
fprintf('numInit: %d\n', numInit);







% Inputs required for MBA
problem.X = X(1: numInit,:); % top 20K of source.
problem.Z = Z(1: numInit,:); % top 20k of target.
problem.Xfull = X; % full source vocabulary.

problem.Xte = Xtest;
problem.Zte = Ztest;
problem.Yte = Ytest;


% Options are not mandatory.
options.method = 'cg';
options.maxiter = 200;

xinit = [];

[x, infos, options] = MBA(problem, xinit, options);
