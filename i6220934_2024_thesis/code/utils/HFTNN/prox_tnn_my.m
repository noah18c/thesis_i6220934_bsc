function [X,tnn,trank] = prox_tnn_my(Y,rho,fast)
[n1,n2,n3] = size(Y);
X = zeros(n1,n2,n3);
Y = fft(Y,[],3);
tnn = 0;
trank = 0;
        
% first frontal slice
[U,S,V] = svd(Y(:,:,1),'econ');
S = diag(S);
r = length(find(S>rho));
r2=min(r,ceil(0.5*min(n1,n2)));
if r2>min(n1,n2)/3
    fast = 0;
elseif r2<min(n1,n2)/4&&fast==0
    fast = 1;
end
if r>=1
    S = S(1:r)-rho;
    X(:,:,1) = U(:,1:r)*diag(S)*V(:,1:r)';
    tnn = tnn+sum(S);
    trank = max(trank,r);
end
% i=2,...,halfn3
halfn3 = round(n3/2);
%  halfn3 = n3;
for i = 2 : halfn3+1
    if fast==0
%         tic;
        [U,S,V] = svd(Y(:,:,i),'econ');
        S = diag(S);
        r = length(find(S>rho));
        if r>=1
            S = S(1:r)-rho;
            X(:,:,i) = U(:,1:r)*diag(S)*V(:,1:r)';
            tnn = tnn+sum(S)*2;
            trank = max(trank,r);
        end
    else

        [U,S,V] = rsvd_version2(Y(:,:,i),r2,2,2,1);
        S = diag(S);
        r = length(find(S>rho));
        if r>=1
            if r>r2
                aaaa=1;
            end
            S = S(1:r)-rho;
            X(:,:,i) = U(:,1:r)*diag(S)*V(:,1:r)';
            tnn = tnn+sum(S)*2;
            trank = max(trank,r);
        end
    end
    X(:,:,n3+2-i) = conj(X(:,:,i));
end

% % if n3 is even
if mod(n3,2) == 0
    i = halfn3+1;
    [U,S,V] = svd(Y(:,:,i),'econ');
    S = diag(S);
    r = length(find(S>rho));
    if r>=1
        S = S(1:r)-rho;
        X(:,:,i) = U(:,1:r)*diag(S)*V(:,1:r)';
        tnn = tnn+sum(S);
        trank = max(trank,r);
    end
end
tnn = tnn/n3;
X = ifft(X,[],3);
end

% randomized low rank SVD using QR of B^T version 
function [U,Sigma,V] = rsvd_version2(A,k,p,q,s)
    m = size(A,1);
    n = size(A,2);
    l = k + p;

    R = randn(n,l);
    Y = A*R; % m \times n * n \times k = m \times k

    for j=1:q
        if mod(2*j-2,s) == 0
            [Y,~] = qr(Y,0);
        end
        Z = A'*Y;

        if mod(2*j-1,s) == 0
            [Z,~] = qr(Z,0);
        end
        Y = A*Z;
    end    
    [Q,~] = qr(Y,0);


    %B = Q'*A; % l \times m * m \times n = l \times n
    %Bt = B'; % n \times l
    Bt = A'*Q;

    [Qhat,Rhat] = qr(Bt,0);

    % Rhat is l \times l
%     whos Qhat Rhat

    [Uhat,Sigmahat,Vhat] = svd(Rhat,'econ');

    U = Q*Vhat;
    Sigma = Sigmahat;
    V = Qhat*Uhat;
    
    % take first k components
    U = U(:,1:k);
    Sigma = Sigma(1:k,1:k);
    V = V(:,1:k);
end
