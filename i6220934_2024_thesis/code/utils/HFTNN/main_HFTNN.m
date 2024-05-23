% This script compares low-rank tensor completion methods
% More detail can be found in [1]
% [1] Honghui Xu, Jianwei Zheng*, Xiaomin Yao, Yuchao Feng, and Shengyong Chen,.
%     Fast Tensor Nuclear Norm for Structured Low-Rank Visual Inpainting
%     IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

clear all
clc

load 'stefan.mat'
name='stefan';

for a = [0.1]; % sampling rate
%%
    [m,n,p]=size(img);
    img    = img/max(img(:));
    Img_Size 	= size(img);
    SR=a;
    mask=rand(m,n,p);
    mask(mask<a)=1;
    mask((mask~=1))=0;
    dimg=img.*mask;
    o=num2str((1-sum(mask(:))/numel(mask))*100,3)
    [dimg,mask,r1,r2] = rempat(dimg,mask);
    % [dimg,mask,r1,r2] = rempat1_1(dimg1,mask);
   

%% demo
    ex2 = 'FTNN';
    [recon_HFTNN1] = Hankel_keams(dimg,mask);
    [recon_HFTNN] = rempat1_3(recon_HFTNN1,r1,r2,p);
    %  [recon_HFTNN] = rempat1_2(recon_HFTNN1,r,p);
 
%% display results
    PSNR_HFTNN= PSNR_RGB(double(recon_HFTNN),double(img));
    PSNRvector =zeros(1,p);
    for i=1:1:p
        J=255*img(:,:,i);
        I=255* recon_HFTNN(:,:,i); 
        PSNRvector(1,i)=PSNR(J,I,m,n);
    end
    PSNR_HFTNN111  =mean(PSNRvector )
    SSIMvector =zeros(1,p);
    for i=1:1:p
        J=255*img(:,:,i);
        I=255*recon_HFTNN(:,:,i); 
        [ SSIMvector(1,i),ssim_map] = ssim(J,I);
    end
    SSIM_Hankel=mean(SSIMvector );
    disp(['PSNR_',num2str(a) '_',name,' = ',num2str(PSNR_HFTNN)]);
    disp(['SSIM_',num2str(a) ' = ',num2str(SSIM_Hankel)]);
    save(['result_' name '_' num2str(a) '_' ex2 '.mat'],'SSIMvector','recon_HFTNN');
end
