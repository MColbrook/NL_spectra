clear
close all

%% Set physical parameters
d1 = 0.15;
d2 = 0.5;
tau = 0.1;
r = sqrt(2);
r1 = 1;
r2 = 0.5;
a = 0;
b = 0;

%% Set numerical parameters
N = 100;
xpts=-6:0.1:1;    ypts=-7:0.1:7; % increase resolution if desired
zpts=kron(xpts,ones(length(ypts),1))+1i*kron(ones(1,length(xpts)),ypts(:));    zpts=zpts(:);

%% Find steady state
x = chebfun('x');
L = chebop(@(x,u,v) [d1*tau*diff(u,2) + tau*u.*(r1-a*u-v); d2*tau*diff(v,2) + tau*v.*(-r2+u-b*v)]);
L.lbc = @(u,v)[ u-2; v-1];
L.rbc =  @(u,v)[ u; v];
L.init = [0*x; 1+x.^2];
[u,v] = L\0;

figure
plot([u, v],'linewidth',1)
xlabel('$x$','interpreter','latex','fontsize',18)
title('$r_2=0.5$','interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=18;
legend({'prey','predator'},'fontsize',16,'interpreter','latex','location','northeast')

%% Build operator pieces

N = N+300;

D = leg_diffmat2(N+2); D2 = D*D;
U = leg_multmat(N+2,u,0.5);
V = leg_multmat(N+2,v,0.5);
S1 = leg_normalize(N+2, 0.5);
I = eye(N+2);

%% Take care of boundary conditions

bc = leg_multmat(N+2,chebfun(@(x) (x.^2-1)),0.5);
[Q,~] = qr(S1*bc(:,1:N),"econ");
Q = diag(1./diag(S1))*Q;

D2 = D2*Q;
U = U*Q;
V = V*Q;
I = I*Q;

c = bandwidth(D2,'lower');
c = max(c,bandwidth(U,'lower'));
c = max(c,bandwidth(V,'lower'));
c = max(c,bandwidth(I,'lower'));

N = N-300;
S1 = S1(1:(N+c),1:(N+c));

D2 = S1*D2(1:(N+c),1:N);
U = S1*U(1:(N+c),1:N);
V = S1*V(1:(N+c),1:N);
I = S1*I(1:(N+c),1:N);


%% Build pieces of pencil

LI = [I,0*I;0*I,I];
Ld = tau*[d1*D2,   0*I;
    0*I,   d2*D2];

L1 = tau*[r1*I-2*U*a-V,   0*I;
    0*I,   -r2*I-2*b*V+U];
L2a = tau*[0*I,   -U;
    0*I,   0*I];
L2b = tau*[0*I,   0*I;
    V,   0*I];

%% Compute pseudospectra

RES=0*zpts+1;
pf = parfor_progress(length(zpts));
pfcleanup = onCleanup(@() delete(pf));

for jj=1:length(zpts)
    L = zpts(jj)*LI - Ld - L1 -L2a*exp(-zpts(jj)*r) -L2b*exp(-zpts(jj));
    RES(jj) = min(svd(L));
    parfor_progress(pf);  
end

RES=reshape(RES,length(ypts),length(xpts));

%% Plot the results
% plotting parameters
v = (10.^(-6:0.2:4));
cSCALE = [-6,1]; % scale for logarithmic epsilon

f=figure;
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
plot([0,0],[-10,10],'r','linewidth',1)
xlabel('$\mathrm{Re}(z)$','interpreter','latex','fontsize',18)
ylabel('$\mathrm{Im}(z)$','interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=14;
box on
grid minor
set(gca,'layer','top');
axis equal
ylim([-7,7])
xlim([-6,1])
title('$r_2=0.5$','interpreter','latex','fontsize',14)
f.Position = [360.0000   50.3333  560.0000  590.6667];



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




