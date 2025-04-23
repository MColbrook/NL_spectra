clear
close all

%% Set parameters

N = 100;
a = chebfun(@(x) 0*x+1+cos(x)/2);
b = chebfun(@(x) 1.5+tanh(10*x));

% nu = 0.5;
% xpts=-100:1:20;    ypts=(-200:2:200);
% zpts=kron(xpts,ones(length(ypts),1))+1i*kron(ones(1,length(xpts)),ypts(:));    zpts=zpts(:);

% nu = 1;
% xpts=-300:2:20;    ypts=(-100:1:100);
% zpts=kron(xpts,ones(length(ypts),1))+1i*kron(ones(1,length(xpts)),ypts(:));    zpts=zpts(:);

nu = 1.5;
xpts=-50:0.5:20;    ypts=(-50:1:50);
zpts=kron(xpts,ones(length(ypts),1))+1i*kron(ones(1,length(xpts)),ypts(:));    zpts=zpts(:);

% plotting parameters
v = (10.^(-6:0.25:4));
cSCALE = [-6,4]; % scale for logarithmic epsilon

%% Build operator pieces

N = N+300;

D = leg_diffmat2(N+4);
A = leg_multmat(N+4,a,0.5);
B = leg_multmat(N+4,b,0.5);
S1 = leg_normalize(N+4, 0.5);

L1 = S1*D*D*A*D*D;
L2 = S1*D*D*B*D*D;
L0 = S1*eye(N+4);

%% Take care of boundary conditions

bc = leg_multmat(N+4,chebfun(@(x) (x.^2-1).^2),0.5);
[Q,~] = qr(S1*bc(:,1:N),"econ");
Q = diag(1./diag(S1))*Q;

% % check BCs
% b1 = ones(1,N+4);
% nn = 0:(N+3);
% mm = (-1).^nn;
% b2 = nn.*(nn+1)/2;
% 
% max(abs((mm.*b1)*Q))
% max(abs((mm.*b2)*Q))

L1 = L1*Q;
L2 = L2*Q;
L0 = L0*Q;

N = N-300;

L1 = L1(:,1:N);
L2 = L2(:,1:N);
L0 = L0(:,1:N);

c = bandwidth(L0,'lower');
c = max(c,bandwidth(L1,'lower'));
c = max(c,bandwidth(L2,'lower'));

L1 = L1(1:(N+c),1:N);
L2 = L2(1:(N+c),1:N);
L0 = L0(1:(N+c),1:N);

%% Compute pseudospectra

RES=0*zpts+1;
pf = parfor_progress(length(zpts));
pfcleanup = onCleanup(@() delete(pf));

for jj=1:length(zpts)
    L = zpts(jj)^2*L0 + L1 + zpts(jj)^nu*L2;
    RES(jj) = min(svd(L));
    parfor_progress(pf);  
end

RES=reshape(RES,length(ypts),length(xpts));

%% Numerical range bounds

RES2=0*zpts;
pf = parfor_progress(length(zpts));
pfcleanup = onCleanup(@() delete(pf));

mm = 2*sqrt(max(a./b));

for jj=1:length(zpts)
    r = max(abs(zpts(jj)),0.001); th = angle(zpts(jj))+0.00001;
    if abs((2-nu)*th)<pi
        ep = chebfun(@(x) 10*(x+1)*r+r*cos(th));
        xx = chebfun(@(x) (x+1)*1000);
        xx2 =(xx.^2+cos(th))*r;
        f = r^(nu/2) - mm*xx*sqrt(abs(cos((nu-1)*th))/(sin((2-nu)*th))^2) - r^(nu/2-1)*xx2*sqrt(2)/abs(sin((2-nu)*th));
        z = roots(f);
        if ~isempty(z)
            RES2(jj) = xx2(max(z))*r;
        end
    end
    if real(zpts(jj))>=0
        RES2(jj) = max(RES2(jj),real(zpts(jj))*r);
    end
    parfor_progress(pf);  
end

RES2=reshape(RES2,length(ypts),length(xpts));

%% Plot the results
% close all

figure
contourf(reshape(real(zpts),length(ypts),length(xpts)),reshape(imag(zpts),length(ypts),length(xpts)),log10(max(real(RES),min(v))),log10(v),'LineColor',[1,1,1]*0,...
    'linewidth',1,'linestyle','-','ShowText','off');
cbh=colorbar;
cbh.Ticks=log10(10.^(-20:1:10));
cbh.TickLabels=["1e-20","1e-19","1e-18","1e-17","1e-16","1e-15","1e-14","1e-13","1e-12","1e-11",...
    "1e-10","1e-9","1e-8","1e-7","1e-6","1e-5","1e-4","1e-3","1e-2","1e-1","1",...
    "1e1","1e2","1e3","1e4","1e5","1e6","1e7","1e8","1e9","1e10"];
clim(cSCALE)
colormap bone
ax=gca; ax.FontSize=14;
hold on
if abs(nu-1)>0.0001
    plot([min(xpts(:)),0],[0,0],'r','linewidth',2)
end
xlabel('$\mathrm{Re}(z)$','interpreter','latex','fontsize',18)
ylabel('$\mathrm{Im}(z)$','interpreter','latex','fontsize',18)
title(sprintf('Computed Pseudospectra, $\\nu=%1.2g$',nu),'interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=18;
box on
grid minor
set(gca,'layer','top');

figure
contourf(reshape(real(zpts),length(ypts),length(xpts)),reshape(imag(zpts),length(ypts),length(xpts)),log10(max(real(RES2),min(v))),log10(v),'LineColor',[1,1,1]*0,...
    'linewidth',1,'linestyle','-','ShowText','off');
cbh=colorbar;
cbh.Ticks=log10(10.^(-20:1:10));
cbh.TickLabels=["1e-20","1e-19","1e-18","1e-17","1e-16","1e-15","1e-14","1e-13","1e-12","1e-11",...
    "1e-10","1e-9","1e-8","1e-7","1e-6","1e-5","1e-4","1e-3","1e-2","1e-1","1",...
    "1e1","1e2","1e3","1e4","1e5","1e6","1e7","1e8","1e9","1e10"];
clim(cSCALE)
colormap bone
ax=gca; ax.FontSize=14;
hold on
if abs(nu-1)>0.0001
    plot([min(xpts(:)),0],[0,0],'r','linewidth',2)
end
xlabel('$\mathrm{Re}(z)$','interpreter','latex','fontsize',18)
ylabel('$\mathrm{Im}(z)$','interpreter','latex','fontsize',18)
title(sprintf('Numerical Range Bound, $\\nu=%1.2g$',nu),'interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=18;
box on
grid minor
set(gca,'layer','top');




%% Code for spectral discretisation using Legendre polynomials

function S = leg_normalize(n, lambda)
% normalisation for Legendre polynomials (used to compute L^2 residuals)
C = ones(1,n);
for ii = 1:round(2*lambda-1)
    C = C.*((1:n)+ii-1);
end
C = sqrt(((2^(1-2*lambda)*pi/(gamma(lambda)^2))./((0:n-1)+lambda)).*C);
S = spdiags(C',0,n,n);
end

function D = leg_diffmat2(n)
% Legendre differentiation matrix
D = sparse(n,n);
for jj=2:n
    nn = (jj-2):-2:0;
    D((jj-1:-2:1),jj) = 2*nn+1;
end
end

function M = leg_multmat(n, f, lambda)
% Matrix for multiplication by a chebfun f in Legendre space
a = legcoeffs(f);

% Multiplying by a scalar is easy.
if ( numel(a) == 1 )
    M = a*speye(n);
    return
end

% Prolong or truncate coefficients
if ( numel(a) < n )
    a = [a ; zeros(n - numel(a), 1)];   % Prolong
else
    a = a(1:n);                         % Truncate.
end

% Convert to C^{lam}
a = ultraS.convertmat(n, 0.5, lambda - 1) * a;

m = 2*n; 
M0 = speye(m);

d1 = [1 (2*lambda : 2*lambda + m - 2)]./ ...
    [1 (2*((lambda+1) : lambda + m - 1))];
d2 = (1:m)./(2*(lambda:lambda + m - 1));
B = [d2' zeros(m, 1) d1'];
Mx = spdiags(B,[-1 0 1], m, m);
M1 = 2*lambda*Mx;

% Construct the multiplication operator by a three-term recurrence: 
M = a(1)*M0;
M = M + a(2)*M1;
for nn = 1:length(a) - 2
    M2 = 2*(nn + lambda)/(nn + 1)*Mx*M1 - (nn + 2*lambda - 1)/(nn + 1)*M0;
    M = M + a(nn + 2)*M2;
    M0 = M1;
    M1 = M2;
    if ( abs(a(nn + 3:end)) < eps ), break, end
end
M = M(1:n, 1:n); 

end
