function [dimg,mask1,r1,r2] = rempat(img1,mask,a)
%% 1 重复取前面几层
% r=3;
% [m,n,p]=size(img1);
% dimg=zeros(m,n,p+r+r);
% [m1,n1,p1]=size(dimg);
% dimg(:,:,1:r)=img1(:,:,1:r);
% dimg(:,:,r+1:p+r)=img1(:,:,1:p);
% dimg(:,:,p+r+1:p1)=img1(:,:,p-r+1:p);
% 
% mask1=zeros(m,n,p1);
% mask1(:,:,1:r)=mask(:,:,1:r);
% mask1(:,:,r+1:p+r)=mask(:,:,1:p);
% mask1(:,:,p+r+1:p1)=mask(:,:,p-r+1:p);

%% 2
r=3;
r1=3;
r2=3;
[m,n,p]=size(img1);
img=zeros(m,n,p+r1+r2);
[m1,n1,p1]=size(img);
halfn3 = round(p/2);
img(:,:,1:halfn3)=img1(:,:,1:halfn3);
img(:,:,halfn3+1:halfn3+r1)=img1(:,:,1:r1);
img(:,:,halfn3+r1+1:halfn3+r1+r2)=img1(:,:,p-r2+1:p);
img(:,:,halfn3+r1+r2+1:p1)=img1(:,:,halfn3+1:p);
img(:,:,1:r)=img1(:,:,1:r);
img(:,:,r+1:p+r)=img1(:,:,1:p);
img(:,:,p+r+1:p1)=img1(:,:,p-r+1:p);
mask=rand(m,n,p1);
mask(mask<a)=1;
mask((mask~=1))=0;
dimg=img.*mask;

%% 3将中间的取出来放在两端
% r=4;
% r1=4;
% r2=4;
% [m,n,p]=size(img1);
% dimg=zeros(m,n,p+r1+r2);
% [m1,n1,p1]=size(dimg);
% dimg(:,:,1:r1)=img1(:,:,10+1:10+r1);
% dimg(:,:,r1+1:p+r1)=img1(:,:,1:p);
% dimg(:,:,p+r2+1:p1)=img1(:,:,20+1:20+r2);
% 
% mask1=zeros(m,n,p1);
% mask1(:,:,1:r)=mask(:,:,10+1:10+r1);
% mask1(:,:,r1+1:p+r1)=mask(:,:,1:p);
% mask1(:,:,p+r2+1:p1)=mask(:,:,20+1:20+r2);
end