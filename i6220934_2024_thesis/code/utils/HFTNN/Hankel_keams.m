function [recon_HTNN] = Hankel_keams(dimg,mask)
%% parameter settings
    opts.DEBUG = 1;
    opts.tr    = 0;
    opts.frsvd  = 1;
    options=[]; 
    options.ReducedDim=1;
    fkmeans_opt.careful = 1; 
    iter1=0.7;
    Nimg=45;
    Nfir=7;    % should be odd
    %%
    [m,n,p]=size(dimg);
    param=struct('Nimg',Nimg,'Nfir',Nfir,'m',m,'n',n,'p',p);
    [X,omega,vid,mid,Ny,Nc,rimg] = imagetoHankel3(dimg,mask,param);
    o=num2str((1-sum(omega(:))/numel(omega))*100,3);
    omega2 = find(omega==1);
    nclusters=8;%%´ØÊý   
    X1 = reshape(X, size(X, 1)*size(X, 2)*size(X, 3), size(X, 4))';  
    [eigvector, eigvalue] = PCA(X1,options);
    X1 = X1*eigvector;
    fkmeans_opt.careful = 1;
    fprintf('Matching similar blocks...\n');
    % [idx1,~] = kmeans(X,nclusters);
    idx = fkmeans(X1, nclusters, fkmeans_opt);
tic
for k = 1:nclusters   
    nblocks = numel(find(idx==k));
    matched_blocks = X(:, :, :,idx==k);
    sz = size(matched_blocks);
    omega_blocks = omega(:, :,  :,idx==k);
    if  nblocks==1
        matched_blocks=matched_blocks; 
        omega_blocks = omega(:, :,  :,idx==k);
    else
    sz = size(matched_blocks); 
    omega_blocks = omega(:, :,  :,idx==k);
    matched_blocks=reshape(matched_blocks, sz(1),sz(2), sz(3)*sz(4)); 
    omega_blocks=reshape(omega_blocks, sz(1),sz(2), sz(3)*sz(4));
    end
    omega2 = find(omega_blocks==1); 
     [XX,iter] = lrtc_tnn1(matched_blocks,omega2,opts,iter1);
     if nblocks==1
         XX=XX;
     else
     XX=reshape(XX, sz(1),sz(2), sz(3),sz(4)); 
     end
     Xhat(:, :,  :,idx==k) = double(XX);  
end 
toc;
[Xhat2] = Hankeltoimage(Xhat,param,vid,mid,Ny,Nc,rimg);
 Xhat2(mask==1)=dimg(mask==1);
 recon_HTNN=Xhat2;
end


function [cmtx,mask_cmtx,vid,mid,Ny,Nc,rimg] = imagetoHankel3(dimg,mask,param)
Nimg    =param.Nimg;
Nfir     =param.Nfir;    % should be odd
p=param.p;   
%% select scanning positions
[dimg,vid,mid] = make_dsr_rgb(dimg,Nimg,Nfir);
if mod(Nimg,2)==0
    hNimg=Nimg/2;
else
    hNimg=(Nimg-1)/2;
end

Ny=size(dimg,1);
rimg=zeros(size(dimg));
N=length(mid(:));
maskp=padarray(mask,[hNimg,hNimg]);
Nc=size(dimg,3);
H   = @(inp) patch2hank2(inp,Nimg,Nimg,Nc,Nfir,Nfir);

%% patch based ALOHA
mask_cmtx = zeros((Nimg-Nfir+1)*(Nimg-Nfir+1),Nfir*Nfir,Nc);
cmtx      = mask_cmtx;
for iter=1:N
    idxPatch = p*(iter-1)+1:p*(iter-1)+p;
    ucur=mid(iter)-1;
    uy=mod(ucur,Ny)+1;
    ux=fix(ucur/Ny)+1;
    
    if mod(Nimg,2)==0
        roiy=uy-hNimg:uy+hNimg-1;
        roix=ux-hNimg:ux+hNimg-1;
    else
        roiy=uy-hNimg:uy+hNimg;
        roix=ux-hNimg:ux+hNimg;
    end
    
    rmask       = maskp(roiy,roix,:);
    mask_cmtx(:,:,:,iter) = H(rmask);
    rval        = dimg(roiy,roix,:);
    cmtx(:,:,:,iter)      = H(rval);

end
end 



function [recon] = Hankeltoimage(output_image,param,vid,mid,Ny,Nc,rimg)
N=length(mid(:));
Nimg=param.Nimg;
Nfir=param.Nfir;
M=param.m;
n=param.n;
p=param.p;
hNfir=(Nfir-1)/2;
if mod(Nimg,2)==0
    hNimg=Nimg/2;
else
    hNimg=(Nimg-1)/2;
end
Hi  = @(inp) hank2patch(inp,Nimg,Nimg,Nc,Nfir,Nfir);
map_count=zeros(size(rimg));
for iter=1:N
    ucur=mid(iter)-1;
    uy=mod(ucur,Ny)+1;
    ux=fix(ucur/Ny)+1;
    
    if mod(Nimg,2)==0
        roiy=uy-hNimg:uy+hNimg-1;
        roix=ux-hNimg:ux+hNimg-1;
        roiys=roiy(hNfir+1:end-hNfir);
        roixs=roix(hNfir+1:end-hNfir);
    else
        roiy=uy-hNimg:uy+hNimg;
        roix=ux-hNimg:ux+hNimg;
        roiys=roiy(hNfir+1:end-hNfir);
        roixs=roix(hNfir+1:end-hNfir);
    end
     output_image1=Hi(output_image(:,:,:,iter));
% output_image1=output_image(:,:,:,iter);
    rimg(roiys,roixs,:)       = rimg(roiys,roixs,:)+output_image1(hNfir+1:end-hNfir,hNfir+1:end-hNfir,:);
    map_count(roiys,roixs,:)  = map_count(roiys,roixs,:)+1;
 
end 
id=(map_count==0);
map_count(id)=1;
rimg_n=rimg./map_count;
recon   = reshape(rimg_n(vid),M,n,p);
end
