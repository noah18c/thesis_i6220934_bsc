function [recon_HFTNN1] = rempat1(recon_HFTNN,r1,r2,p)
%%
r=r1;
recon_HFTNN1=recon_HFTNN(:,:,r+1:p+r); %%重复取前面几层的逆




%%
% [m,n,p]=size(recon_HFTNN);
% p1=p-r1-r2;
% halfn3 = round(p1/2);
% % recon_HFTNN1=zeros(m,n.p1);
% recon_HFTNN1(:,:,1:r1)=recon_HFTNN(:,:,halfn3+1:halfn3+r1);
% recon_HFTNN1(:,:,r1+1:halfn3)=recon_HFTNN(:,:,r1+1:halfn3);
% recon_HFTNN1(:,:,halfn3+1:p1-r2)=recon_HFTNN(:,:,halfn3+r1+r2+1:p-r2);
% recon_HFTNN1(:,:,p1-r2+1:p1)=recon_HFTNN(:,:,halfn3+r1+1:halfn3+r1+r2);


%%

end