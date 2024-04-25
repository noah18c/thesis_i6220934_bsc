function out = machowskyPredict(u, y, lmax, uf, delta)
    t = delta; % time interval for which we will calculate the response
    T = length(u);

    k = 0;

    % We split the input where the past inputs goes from 1 to lmax, and
    % future goes from lmax+1 to delta
    j = T - delta -lmax+1;
    Hu = hankelizeM(u,lmax+delta);
    Hy = hankelizeM(y,lmax+delta);

    Up = Hu(1:lmax,1:j);
    Uf = Hu(lmax+1:end,1:j);
    Yp = Hy(1:lmax,1:j);
    Yf = Hy(lmax+1:end,1:j);

    fu = [zeros(lmax,1);uf(1:delta,1)];
    fy_p = zeros(lmax,1);
    out = zeros(1,t);


    while t>=k*delta
        A = [Up;Uf;Yp];
        B = [fu; fy_p];
        gk = linsolve(A,B);

        yf = Yf*gk;

        
        fu = [fu(delta+1:end,:); u(k * delta + 1:(k+1) * delta,:)];
        fy_p = [fy_p(delta+1:end,:);yf(delta+1:end,:)];

        k=k+1;
        out(k) = yf;
    end

end
