function [img,mask1,r1,r2] = rempat(img1,mask)
%% 1

[m,n,p]=size(img1);
r=round(0.1*p); %0.1-0.3
r1=r;
r2=r;
img=zeros(m,n,p+r+r);
[m1,n1,p1]=size(img);
img(:,:,1:r)=img1(:,:,1:r);
img(:,:,r+1:p+r)=img1(:,:,1:p);
img(:,:,p+r+1:p1)=img1(:,:,p-r+1:p);
mask1=rand(m,n,p1);
mask1(:,:,1:r)=mask(:,:,1:r);
mask1(:,:,r+1:p+r)=mask(:,:,1:p);
mask1(:,:,p+r+1:p1)=mask(:,:,p-r+1:p);



%% 2
% r1=4;
% r2=4;
% [m,n,p]=size(img1);
% img=zeros(m,n,p+r1+r2);
% mask1=zeros(m,n,p+r1+r2);
% [m1,n1,p1]=size(img);
% halfn3 = round(p/2);
% img(:,:,1:halfn3)=img1(:,:,1:halfn3);
% img(:,:,halfn3+1:halfn3+r1)=img1(:,:,1:r1);
% img(:,:,halfn3+r1+1:halfn3+r1+r2)=img1(:,:,p-r2+1:p);
% img(:,:,halfn3+r1+r2+1:p1)=img1(:,:,halfn3+1:p);
%  
% mask1(:,:,1:halfn3)=mask(:,:,1:halfn3);
% mask1(:,:,halfn3+1:halfn3+r1)=mask(:,:,1:r1);
% mask1(:,:,halfn3+r1+1:halfn3+r1+r2)=mask(:,:,p-r2+1:p);
% mask1(:,:,halfn3+r1+r2+1:p1)=mask(:,:,halfn3+1:p);

end