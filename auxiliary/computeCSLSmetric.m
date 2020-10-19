function [mymetrics] = computeCSLSmetric(Xtest, X, Ztest, Ytest, W)
	tic;
	%sumYtest = sum(Ytest, 2);
    numTest = size(Ytest,1);

    L = X*W;

	Ltest = Xtest*(W);
	Rtest = (Ztest)';

	L = bsxfun(@times, L, 1./sqrt(sum(L.^2, 2)));

	Ltest = bsxfun(@times, Ltest, 1./sqrt(sum(Ltest.^2, 2)));
	Rtest = bsxfun(@times, Rtest, 1./sqrt(sum(Rtest.^2, 1)));

	ntest = size(Ltest,1);
	Jtest = cslsProxy(Ltest,L,Rtest',10,1);

	Itest = 1:ntest;
	Ypred = sparse(Itest,Jtest,ones(ntest,1),ntest,size(Rtest,2));
	Ymask = (Ypred.*Ytest);

	sumYmask = sum(Ymask,2);
	mymetrics = full(sum(sumYmask)/numTest);
	time_taken = toc;
	fprintf('Time taken by computeCSLSmetric: %e\n',time_taken);
end