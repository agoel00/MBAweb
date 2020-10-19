function topIdx = cslsProxy(Xtest, X, Z, k, topk)
    %tic;
    % Assumptions: X\in\R^{m\times d} and Z\in\R^{n\times d}.
    % X(i,:) and Z(j,:) are unit norm vectors
    %meansim = 0.5*meanSimilarity(X,Z,k);%column vector
    maxsize = 1000;
    if size(Z,1) > maxsize;
        startidx = 1:maxsize:size(Z,1);
        endidx = maxsize:maxsize:size(Z,1);
        if length(startidx) == length(endidx) + 1
            endidx = [endidx, size(Z,1)];
        elseif length(startidx) == length(endidx)
            
        else
            fprintf('[cslsProxy]Problem in startidx endidx\n');
            keyboard;
        end
    else
        startidx = 1;
        endidx = size(Z,1);
    end
    m = length(startidx);
    meansim = [];
    for i=1:m
        %fprintf('CSLS: computing hubness, iter %d of %d, startIdx: %d, endidx: %d\n',i,m,startidx(i),endidx(i));
        XZt = X*Z(startidx(i):endidx(i),:)';
        meansim_i = mean(maxk(XZt,k));
        meansim = [meansim, meansim_i];
        if maxsize >= 5000
            clear XZt
        end
    end
    clear XZt
    %XtestZt = (2*Xtest)*Z';
    %cslsProx = bsxfun(@minus,XtestZt,meansim);
    %clear XtestZt
    %[~,topIdx] = maxk(cslsProx',topk);
    ZXtestt = Z*(2*Xtest)';
    cslsProxt = bsxfun(@minus,ZXtestt,meansim');
    clear ZXtestt
    [~,topIdx] = maxk(cslsProxt,topk);
    topIdx = topIdx';
    %toc
end

%function meansim = meanSimilarity(X,Z,k)
    % tic;
%    [~, distanceMat] = knnsearch(X,Z,'K',k,'Distance','cosine');
    % toc;
%    meansim = mean(1-distanceMat,2);
% end