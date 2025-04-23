clear
close all

%% Truncation and FEM
impedance = 1;
nvec = [10,100,500,1000];
F = zeros(4,length(nvec));
ct = 1;
lamFD=cell(length(nvec),1);

for n = nvec
    n
    cfs = nlevp('acoustic_wave_1d',n,impedance); 
    lamFD{ct} = 2*pi*polyeig(cfs{:});
    mm(ct) = min(abs(lamFD{ct}));
    ct = ct+1;
end

figure
plot(real(lamFD{1}),imag(lamFD{1}),'.','markersize',15)
hold on
plot(real(lamFD{2}),imag(lamFD{2}),'.','markersize',15)
plot(real(lamFD{3}),imag(lamFD{3}),'.','markersize',15)
plot(real(lamFD{4}),imag(lamFD{4}),'.','markersize',15)
ax = gca; ax.FontSize = 18;
xlabel('$\mathrm{Re}(Lz)$','interpreter','latex','fontsize',18)
ylabel('$\mathrm{Im}(Lz)$','interpreter','latex','fontsize',18)
legend({'$n=10$','$n=100$','$n=500$','$n=1000$'},'fontsize',16,'interpreter','latex','location','northeast')
grid on

%% Compute pseudospectra
N = 100;
xpts=-2:0.02:2;    ypts=-1:0.02:0.5;
zpts=kron(xpts,ones(length(ypts),1))+1i*kron(ones(1,length(xpts)),ypts(:));    zpts=zpts(:);		% complex points where we compute pseudospectra

RES=0*zpts+1;
pf = parfor_progress(length(zpts));
pfcleanup = onCleanup(@() delete(pf));

for jj=1:length(zpts)
    L = mat_setup(zpts(jj),N,[]);% L = L(1:end-1,:);
    RES(jj) = min(svd(L));
    parfor_progress(pf);  
end

RES=reshape(RES,length(ypts),length(xpts));

%% Plot the results
close all
v=(10.^(-6:0.1:0));
figure
contourf(reshape(real(zpts),length(ypts),length(xpts)),reshape(imag(zpts),length(ypts),length(xpts)),log10(max(real(RES),min(v))),log10(v),'LineColor',[1,1,1]*0,...
    'linewidth',1,'linestyle','-','ShowText','off');
cbh=colorbar;
cbh.Ticks=log10(10.^(-20:1:0));
cbh.TickLabels=["1e-20","1e-19","1e-18","1e-17","1e-16","1e-15","1e-14","1e-13","1e-12","1e-11",...
    "1e-10","1e-9","1e-8","1e-7","1e-6","1e-5","1e-4","1e-3","1e-2","1e-1","1"];
clim([-4,0])
colormap bone
ax=gca; ax.FontSize=14;
hold on
xlabel('$\mathrm{Re}(z)$','interpreter','latex','fontsize',18)
ylabel('$\mathrm{Im}(z)$','interpreter','latex','fontsize',18)
title('$N=100$','interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=18;
box on
grid minor
set(gca,'layer','top');


%% Pseudoeigenfunctions
clear
z = 1i+2*pi;
X = 0:0.01:6;
[L,Z] = mat_setup(z,50,X);
[~,S,U] = svds(L,1,'smallest');

Y = Z*U;
Y2 = exp(1i*z*X);
Y = Y2(X==1.1)/Y(X==1.1)*Y;

figure
plot(X,Y,'linewidth',1)
hold on
plot(X,Y2,'linewidth',1)
xlabel('$x$','interpreter','latex','fontsize',18)
title('$N=50$','interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=18;
legend({'approximation','exact'},'fontsize',16,'interpreter','latex','location','northeast')








function [L,Z] = mat_setup(lam,N,X)

%% Set up matrices using Laguerre functions

NN = N+1;
D = zeros(NN,NN);
for jj = 1:NN
    D(1:jj-1,jj)= -1;
    D(jj,jj) = -1/2;
end

L = D*D + lam^2*eye(NN);

% Basis recombination

B = [zeros(1,NN-1);eye(NN-1,NN-1)];
for jj = 1:NN-1
     B(1,jj) = (-(jj+1/2)-1i*lam)/(0.5+1i*lam);
end


% %% Check
% 
% P1 = ones(1,N);
% P2 = -0.5-(0:N-1);
% 
% P = P1*1i*lam - P2;

[Q,~] = qr(B,"econ");

L = L*Q;
NN = NN - 1;
L = L(:,1:NN);


if ~isempty(X)
    X=X(:);
    LL = zeros(length(X),N+1);
    LL(:,1) = exp(-X/2);
    LL(:,2) = (1 - X).*exp(-X/2);
    for jj=2:N
        k = jj - 1;
        LL(:,jj+1) = (2*k+1 - X)./(k+1).*LL(:,jj)-k/(k+1)*LL(:,jj-1);
    end
    Z = LL*Q;
else
    Z = [];
end

end


















function [coeffs,fun,F,xcoeffs] = acoustic_wave_1d(n,z)
%ACOUSTIC_WAVE_1D   Acoustic wave problem in 1 dimension.
%  [COEFFS,FUN,F] = nlevp('acoustic_wave_1d',N,Z) constructs an N-by-N
%  quadratic matrix polynomial lambda^2*M + lambda*D + K that arises
%  from the discretization of a 1D acoustic wave equation.
%  The damping matrix has the form 2*pi*i*Z^(-1)*C, where
%  C = e_n*e_n', where e_n = [0 ... 0 1]', and the scalar parameter Z is
%  the impedance (possibly complex).
%  The default values are N = 10 and Z = 1.
%  The eigenvalues lie in the upper half of the complex plane.
%  The matrices are returned in a cell array: COEFFS = {K, D, M}.
%  FUN is a function handle to evaluate the monomials 1,lambda,lambda^2
%  and their derivatives.
%  F is the function handle K + lambda*D + lambda^2*M.
%  XCOEFFS returns the cell {1 en 1;K en' M} to exploit the low rank of D.
%  This problem has the properties pep, qep, symmetric, *-even
%  parameter-dependent, scalable, sparse, tridiagonal, banded, low-rank.

%  Reference:
%  F. Chaitin-Chatelin and M. B. van Gijzen, Analysis of parameterized
%  quadratic eigenvalue problems in computational acoustics with homotopic
%  deviation theory, Numer. Linear Algebra Appl. 13 (2006), pp. 487-512.

if nargin < 2 || isempty(z)
    z = 1; 
end
if nargin < 1 || isempty(n) 
    n = 10; 
end

h = 1/n;

e = ones(n,1);
K = spdiags([-e,2*e,-e],-1:1,n,n);
K(n,n) = 1;
K = n*K;

D = sparse(n,n,1/z,n,n);
M = speye(n); M(n,n) = 0.5; M = h*M;

coeffs = {K, 2*pi*1i*D, -(2*pi)^2*M};
fun = @(lam) nlevp_monomials(lam,2);
F =  nlevp_handleQEP(coeffs);
en = sparse(zeros(n,1)); en(n) = 1;
enz = en; enz(n) = 2*pi*1i/z;
xcoeffs1 = {1,  enz, 1};
xcoeffs2 = {coeffs{1}, en', coeffs{3}};
xcoeffs = {xcoeffs1{:}; xcoeffs2{:}};
end


    


