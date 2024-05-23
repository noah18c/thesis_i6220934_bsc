function [X,obj,err,iter] = lrtc_tnn(M,omega,opts,iter1)
% References:
% Canyi Lu, Jiashi Feng, Zhouchen Lin, Shuicheng Yan
% Exact Low Tubal Rank Tensor Recovery from Gaussian Measurements
% International Joint Conference on Artificial Intelligence (IJCAI). 2018


tol = 1e-3; 
max_iter =300;
rho = 1.07;
mu = 1e-3;
max_mu = 1e10;
DEBUG = 0;

if ~exist('opts', 'var')
    opts = [];
end    
if isfield(opts, 'tol');         tol = opts.tol;              end
if isfield(opts, 'max_iter');    max_iter = opts.max_iter;    end
if isfield(opts, 'rho');         rho = opts.rho;              end
if isfield(opts, 'mu');          mu = opts.mu;                end
if isfield(opts, 'max_mu');      max_mu = opts.max_mu;        end
if isfield(opts, 'DEBUG');       DEBUG = opts.DEBUG;          end

Nway = size(M);
X = zeros(Nway);  %% 辅助变量
X(omega) = M(omega); %% 拉格朗日乘子
E = zeros(Nway);
Y = E;
iter = 0;
Error_Set = zeros(max_iter, 1);
for iter = 1 : max_iter
    Xk = X;
    Ek = E;
    % update X

     [X,tnnX] = prox_tnn_my(-E+M+Y/mu,iter1/mu,opts.frsvd); 
% [X,tnnX,trank] = prox_tnn_my1(-E+M+Y/mu,1/mu); 
    % update E
    E = M-X+Y/mu;
    E(omega) = 0;
 
    dY = M-X-E;    
    chgX = max(abs(Xk(:)-X(:)));
    chgE = max(abs(Ek(:)-E(:)));
    chg = max([chgX chgE max(abs(dY(:)))]);
    if DEBUG
        if iter == 1 || mod(iter, 50) == 0
            obj = tnnX;
            err = norm(dY(:));
            fprintf(' iter = %d, mu = %f, chg = %f, err = %e\n', iter, mu, chg, err);
        end
    end
    
    if chg < tol
        fprintf(' iter = %d, chg = %f, err = %e\n', iter, chg, err);
        break;
    end 
    Y = Y + mu*dY;
    mu = min(rho*mu,max_mu);  
    Error_Set(iter) =chg;
end
obj = tnnX;
err = norm(dY(:));

   
 