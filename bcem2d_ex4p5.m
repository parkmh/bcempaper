% This is an example M file to show how to implement the block circulant
% embembedding method (BCEM) introduced in "A Block Circulant Embedding
% Method for Simulation of Stationary Gaussian Random Fields on
% Block-regular Grids" by M. Park and M.V. Tretyakov. See Example 4.4.

n           = 8;
N           = ones(1,2)*n;      % number of blocks in each direction on the domain $\Omega$
m           = N*2;              % number of blocks in each direction on the extended domain $\Omega^E$
mbar        = prod(m);          % total number of blocks on $\Omega^E$
h           = 1./N;             % mesh size
cfhandle    = str2func('exp_cov');  % function handle for the covariance function

% define the location of sampling nodes in each block
dx          = h(1)*[1/3 2/3];
dy          = h(2)*[1/3 2/3];

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

% Load observation data
load fielddata2d;
nobsv = size(fielddata,1);
obsv_x = fielddata(1:nobsv,1);
obsv_y = fielddata(1:nobsv,2);

% build submatrices R11, R12 and R22 of the circulant embedding matrix
c             = zeros(l,mbar*l); % covariance matrix
R12           = zeros(mbar*l,nobsv);
R22           = zeros(nobsv,nobsv);

% build the first row block of R11
for i  = 1 : mbar*l
    xi = x_coord(i);
    yi = y_coord(i);
    for j = 1 : l
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
        c(j,i) = cfhandle(dist,0.3);
    end
end

% build R12
for i  = 1 : nobsv
    xi = obsv_x(i);
    yi = obsv_y(i);
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
        R12(j,i) = cfhandle(dist,0.3);
    end
end

% build R22
for i = 1 : nobsv
    xi = obsv_x(i);
    yi = obsv_y(i);
    for j = 1 : nobsv
        xj = obsv_x(j);
        yj = obsv_y(j);
        diff_x = abs(xi-xj);
        diff_y = abs(yi-yj);
        
        dist = norm([diff_x, diff_y],1);
        R22(j,i) = cfhandle(dist,0.3);
    end
end


% Compute the block diagonal matrix Lambda
c2 = zeros(m(1),m(2),2);
c2(:,:,1) = reshape(c(1,1:l:end),m(1),m(2));
c2(:,:,2) = reshape(c(1,2:l:end),m(1),m(2));

fc = fft2(c2);

L  = [real(reshape(fc(:,:,1),mbar,1)) reshape(fc(:,:,2),mbar,1) ...
      conj(reshape(fc(:,:,2),mbar,1)) real(reshape(fc(:,:,1),mbar,1))]; % Note that the diagonal entries are

iindex = zeros(size(L));
jindex = zeros(size(L));
for i = 1 : mbar
    iindex(i,:) = [1 1 2 2] + l*(i-1);
    jindex(i,:) = [1 2 1 2] + l*(i-1);
end

Lambda = sparse(iindex(:),jindex(:),L(:));

% compute the the square of Lambda
Lambdah      = chol(Lambda/mbar,'lower');

% build the matrix K
R21_2d = zeros(m(1),m(2),nobsv,l);
for i = 1 : nobsv
    R21_2d(:,:,i,1) = reshape(R12(1:l:end,i),m(1),m(2));
    R21_2d(:,:,i,2) = reshape(R12(2:l:end,i),m(1),m(2));
end

ibfft2R12_2d = ifft2(R21_2d);

ibfft2R12 = zeros(nobsv,mbar*l);
for i = 1 : nobsv
    temp = [reshape(ibfft2R12_2d(:,:,i,1),mbar,1)';reshape(ibfft2R12_2d(:,:,i,2),mbar,1)'];
    ibfft2R12(i,:) = temp(:);   
end

K = ibfft2R12/(Lambdah');
KKH = K*K';

% build the matrix L
L = chol(real(R22-KKH),'lower');

