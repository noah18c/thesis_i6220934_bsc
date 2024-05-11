function out = machovskyPredict(u, y, lmax, nmax, uf, delta, t)
    T = length(u);

    k = 0;

    % We split the input where the past inputs goes from 1 to lmax, and
    % future goes from lmax+1 to delta
    j = T - delta -lmax+1;
    Hu = hankel(u(1:lmax+delta,1),u(lmax+delta:end,1));
    %hankelizeM(u,lmax+delta);
    %Hy = hankelizeM(y,lmax+delta);
    Hy = hankel(y(1:lmax+delta,1),y(lmax+delta:end,1));

    Up = Hu(1:lmax,1:j);
    Uf = Hu(lmax+1:end,1:j);
    Yp = Hy(1:lmax,1:j);
    Yf = Hy(lmax+1:end,1:j);

    fu = [zeros(lmax,1);uf(1:delta,1)];
    fy_p = zeros(lmax,1);
    out = zeros(delta, ceil(t/delta));

    while t >= k*delta

        disp("k = "+k);
        A = [Up;Uf;Yp];
        B = [fu; fy_p];
        gk = A \ B;

        yf = Yf*gk;

        out(:,k+1) = yf;
        
        fu = [fu(delta+1:end,:); u(k * delta + 1:(k+1) * delta,:)];
        fy_p = [fy_p(delta+1:end,:);yf(:,:)];

        k=k+1;
    end

end
