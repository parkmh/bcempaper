% This is an example M file to show how to implement the block circulant
% embembedding method (BCEM) introduced in "A Block Circulant Embedding
% Method for Simulation of Stationary Gaussian Random Fields on
% Block-regular Grids" by M. Park and M.V. Tretyakov. See Example 4.3.

n           = 8;
N           = ones(1,2)*n;      % number of blocks in each direction on the domain $\Omega$
m           = N*2;              % number of blocks in each direction on the extended domain $\Omega^E$
mbar        = prod(m);          % total number of blocks on $\Omega^E$
h           = 1./N;             % mesh size
cfhandle    = str2func('exp_cov');  % function handle for the covariance function

% define the location of sampling nodes in each block
dx          = h(1)*[1/4 3/4 1/4 3/4 1/2];
dy          = h(2)*[1/4 1/4 3/4 3/4 1/2];

l           = length(dx);       % number of sampling nodes in each block


%     ____                                             _ __  _
%    / __ \___  _________  ____ ___  ____  ____  _____(_) /_(_)___  ____
%   / / / / _ \/ ___/ __ \/ __ `__ \/ __ \/ __ \/ ___/ / __/ / __ \/ __ \
%  / /_/ /  __/ /__/ /_/ / / / / / / /_/ / /_/ (__  ) / /_/ / /_/ / / / /
% /_____/\___/\___/\____/_/ /_/ /_/ .___/\____/____/_/\__/_/\____/_/ /_/
%                                /_/


% compute the x and y coordinates of all sampling nodes
x_coord     = zeros(mbar*l,1);
y_coord     = zeros(mbar*l,1);
for j = 1 : m(2)
    for i = 1 : m(1)
        k = m(1)*(j-1)+i;
        x_coord(l*(k-1)+1:l*k) = h(1)*(i-1) + dx;
        y_coord(l*(k-1)+1:l*k) = h(2)*(j-1) + dy;
    end
end

% build the first block row of the covariance matrix
C           = zeros(mbar*l,mbar*l); % covariance matrix

for i  = 1 : mbar*l
    xi = x_coord(i);
    yi = y_coord(i);
    for j = 1 : mbar*l
        xj = x_coord(j);
        yj = y_coord(j);
        diff_x = abs(xi-xj);
        diff_y = abs(yi-yj);
        if diff_x > m(1)*h(1)/2
            diff_x = m(1)*h(1)-diff_x;
        end
        if diff_y > m(2)*h(2)/2
            diff_y = m(2)*h(2)-diff_y;
        end
        dist = norm([diff_x, diff_y],1);
        C(j,i) = cfhandle(dist,0.3);
    end
end
c = C(1:l,:);   % first block row of C

% Compute the block diagonal matrix Lambda
c2 = zeros(m(1),m(2),11);
c2(:,:,1) = reshape(c(1,1:l:end),m(1),m(2)); % 1:(1,1) = (2,2) = (3,3) = (4,4) = (5,5)
c2(:,:,2) = reshape(c(1,2:l:end),m(1),m(2)); % 2:(1,2)
c2(:,:,3) = reshape(c(1,3:l:end),m(1),m(2)); % 3:(1,3)
c2(:,:,4) = reshape(c(1,4:l:end),m(1),m(2)); % 4:(1,4)
c2(:,:,5) = reshape(c(1,5:l:end),m(1),m(2)); % 5:(1,5)
c2(:,:,6) = reshape(c(2,3:l:end),m(1),m(2)); % 6:(2,3)
c2(:,:,7) = reshape(c(2,4:l:end),m(1),m(2)); % 7:(2,4)
c2(:,:,8) = reshape(c(2,5:l:end),m(1),m(2)); % 8:(2,5)
c2(:,:,9) = reshape(c(3,4:l:end),m(1),m(2)); % 9:(3,4)
c2(:,:,10) = reshape(c(3,5:l:end),m(1),m(2)); % 10:(3,5)
c2(:,:,11) = reshape(c(4,5:l:end),m(1),m(2)); % 11:(4,5)


fc = fft2(c2);
L  = [real(reshape(fc(:,:,1),mbar,1)) reshape(fc(:,:,2),mbar,1) reshape(fc(:,:,3),mbar,1) reshape(fc(:,:,4),mbar,1) reshape(fc(:,:,5),mbar,1)...
    conj(reshape(fc(:,:,2),mbar,1)) real(reshape(fc(:,:,1),mbar,1)) reshape(fc(:,:,6),mbar,1) reshape(fc(:,:,7),mbar,1) reshape(fc(:,:,8),mbar,1)...
    conj(reshape(fc(:,:,3),mbar,1)) conj(reshape(fc(:,:,6),mbar,1)) real(reshape(fc(:,:,1),mbar,1)) reshape(fc(:,:,9),mbar,1) reshape(fc(:,:,10),mbar,1)...
    conj(reshape(fc(:,:,4),mbar,1)) conj(reshape(fc(:,:,7),mbar,1)) conj(reshape(fc(:,:,9),mbar,1)) real(reshape(fc(:,:,1),mbar,1)) reshape(fc(:,:,11),mbar,1)...
    conj(reshape(fc(:,:,5),mbar,1)) conj(reshape(fc(:,:,8),mbar,1)) conj(reshape(fc(:,:,10),mbar,1)) conj(reshape(fc(:,:,11),mbar,1)) real(reshape(fc(:,:,1),mbar,1))]; % Note that the diagonal entries are

iindex = zeros(size(L));
jindex = zeros(size(L));
for i = 1 : mbar
    iindex(i,:) = [1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 5 5 5 5 5] + l*(i-1);
    jindex(i,:) = [1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5] + l*(i-1);
end

Lambda = sparse(iindex(:),jindex(:),L(:));

% Compute the the square of Lambda
SL      = chol(Lambda/mbar,'lower');

%    ______                           __  _
%   / ____/__  ____  ___  _________ _/ /_(_)___  ____
%  / / __/ _ \/ __ \/ _ \/ ___/ __ `/ __/ / __ \/ __ \
% / /_/ /  __/ / / /  __/ /  / /_/ / /_/ / /_/ / / / /
% \____/\___/_/ /_/\___/_/   \__,_/\__/_/\____/_/ /_/
%

% generate a complex-valued normal random vector
Z = zeros(mbar*l,5000);
for nsamp = 1 : 5000
    x = randn(mbar*l,1) + 1i * randn(mbar*l,1);
    slx = SL*x;
    slx2 = reshape(slx,l,mbar);
    slx3 = zeros(m(1),m(2),l);
    for i = 1 : l
        slx3(:,:,i) = reshape(slx2(i,:),m(1),m(2));
    end
    
    z3 = fft2(slx3);
    z2 = [reshape(z3(:,:,1),1,mbar);reshape(z3(:,:,2),1,mbar);...
          reshape(z3(:,:,3),1,mbar);reshape(z3(:,:,4),1,mbar);...
          reshape(z3(:,:,5),1,mbar)];
    z = z2(:);
    Z(:,nsamp) = z;
end

C2 = cov(real(Z)');
C3 = cov(imag(Z)');
clims = [min([min(min(C)) min(min(C2)) min(min(C3))]) ...
         max([max(max(C)) max(max(C2)) max(max(C3))])];
     
subplot(1,3,1)
imagesc(C,clims)
title('Real covariance')
subplot(1,3,2)
imagesc(C2,clims)
title('cov(real(z))')
subplot(1,3,3)
imagesc(C3,clims)
title('cov(imag(z))')