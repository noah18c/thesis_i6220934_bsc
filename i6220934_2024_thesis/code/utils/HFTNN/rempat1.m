function [recon_HFTNN1] = rempat1(recon_HFTNN,r1,r2)
%  recon_HFTNN1=recon_HFTNN(:,:,r+1:p+r);




%% 2
% [m,n,p]=size(recon_HFTNN);
% p1=p-r1-r2;
% halfn3 = round(p1/2);
% % recon_HFTNN1=zeros(m,n.p1);
% recon_HFTNN1(:,:,1:r1)=recon_HFTNN(:,:,halfn3+1:halfn3+r1);
% recon_HFTNN1(:,:,r1+1:halfn3)=recon_HFTNN(:,:,r1+1:halfn3);
% recon_HFTNN1(:,:,halfn3+1:p1-r2)=recon_HFTNN(:,:,halfn3+r1+r2+1:p-r2);
% recon_HFTNN1(:,:,p1-r2+1:p1)=recon_HFTNN(:,:,halfn3+r1+1:halfn3+r1+r2);
%% 3
[m,n,p]=size(recon_HFTNN);
p1=p-r1-r2;
recon_HFTNN1=zeros(m,n,p1);
recon_HFTNN1(:,:,1)=recon_HFTNN(:,:,11);
recon_HFTNN1(:,:,2)=recon_HFTNN(:,:,17);
recon_HFTNN1(:,:,3:10)=recon_HFTNN(:,:,3:10);
recon_HFTNN1(:,:,11:15)=recon_HFTNN(:,:,12:16);
recon_HFTNN1(:,:,16:20)=recon_HFTNN(:,:,18:22);
recon_HFTNN1(:,:,21:25)=recon_HFTNN(:,:,24:28);
recon_HFTNN1(:,:,26:29)=recon_HFTNN(:,:,30:33);
recon_HFTNN1(:,:,30)=recon_HFTNN(:,:,23);
recon_HFTNN1(:,:,31)=recon_HFTNN(:,:,29);
end