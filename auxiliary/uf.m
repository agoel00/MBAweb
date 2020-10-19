function A1 = uf(A)
	threshold = 1e-8;
    [u, s, v] = svd(A,0);
    nonzeroS = diag(s) > threshold;
    A1 = u(:,nonzeroS)*(v(:,nonzeroS))';
end
