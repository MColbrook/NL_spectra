close all
clear

%% set parameters
V = @(x) -5*exp(-abs(x)); % potential
N = 100; 

%% Build operator

% Jacobi operator part
S = spdiags(2-(-1).^((-(N+1):N+1)'),1,2*N+3,2*N+3); DIFF = (S+S')/2;

% Potential part
V1=sparse(diag(V(-(N+1):N+1)));
V2=V1.^2;

A0 = -DIFF+2*speye(size(DIFF))-V2;
A0 = (A0+A0')/2;
A1 = 2*V1;
A2 = -speye(size(DIFF));

bb = 1; % bandwidth

%% Finite section poly eigenvalue problem

[X,e] = polyeig(A0(1+bb:end-bb,1+bb:end-bb),A1(1+bb:end-bb,1+bb:end-bb),A2(1+bb:end-bb,1+bb:end-bb));
R =  A0(:,1+bb:end-bb)*X + A1(:,1+bb:end-bb)*X.*transpose(e) + A2(:,1+bb:end-bb)*X.*transpose(e.^2);
R = vecnorm(R)./vecnorm(X); % residual
%% Pseudospectra

xpts=-4:0.05:2.4;    ypts=-1:0.05:1;
zpts=kron(xpts,ones(length(ypts),1))+1i*kron(ones(1,length(xpts)),ypts(:));    zpts=zpts(:);		% complex points where we compute pseudospectra

RES=0*zpts+1;

pf = parfor_progress(length(zpts));
pfcleanup = onCleanup(@() delete(pf));

for jj=1:length(zpts)
    B = A0+A1*zpts(jj)+A2*zpts(jj)^2;
    B = (B(:,1+bb:end-bb));
    RES(jj) = svds(B,1,'smallest','MaxIterations',100000);
    parfor_progress(pf);  
end

RES=reshape(RES,length(ypts),length(xpts));

%% Plot the results
v=(10.^(-20:0.1:2));
f=figure;
contourf(reshape(real(zpts),length(ypts),length(xpts)),reshape(imag(zpts),length(ypts),length(xpts)),log10(max(real(RES),min(v))),log10(v),'LineColor',[1,1,1]*0.3,...
    'linewidth',1,'linestyle','-','ShowText','off');
cbh=colorbar;
cbh.Ticks=log10(10.^(-20:1:0));
cbh.TickLabels=["1e-20","1e-19","1e-18","1e-17","1e-16","1e-15","1e-14","1e-13","1e-12","1e-11",...
    "1e-10","1e-9","1e-8","1e-7","1e-6","1e-5","1e-4","1e-3","1e-2","1e-1","1"];
clim([-3,0])
colormap(magma)
colormap bone
ax=gca; ax.FontSize=14;
hold on
plot(real(e(R<0.1)),imag(e(R<0.1)),'.g','markersize',12)
plot(real(e(R>0.1)),imag(e(R>0.1)),'.r','markersize',16)

xlabel('$\mathrm{Re}(z)$','interpreter','latex')
ylabel('$\mathrm{Im}(z)$','interpreter','latex')
ax=gca; ax.FontSize=18;
box on
grid minor
set(gca,'layer','top');
f.Position=[160.0000   97.6667  560.0000*2  420.0000];
axis([min(xpts(:)),max(xpts(:)),min(ypts(:)),max(ypts(:))])

x1 = [-1.2,-1.415];
y1 = [0.8 0.05];

quiver( x1(1),y1(1),x1(2)-x1(1),y1(2)-y1(1),0,'m','linewidth',5,'MaxHeadSize' ,5)

x1 = -0.05 + [1.2,1.415];
y1 = [0.8 0.05];
quiver( x1(1),y1(1),x1(2)-x1(1),y1(2)-y1(1),0,'m','linewidth',5,'MaxHeadSize' ,5)







