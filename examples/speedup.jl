### =======================================================================
### = speedup.jl
### = Alex Turner
### = 11/18/2013
### =----------------------------------------------------------------------
### = NOTES:
### =  ( 1) Warm's up julia, then tests the scaleup.
### =----------------------------------------------------------------------
### = SUBFUNCTIONS:
### =  ( 1) warm_up :: Warms up the processors.
### =----------------------------------------------------------------------
### = INPUTS:
### =  (  ) N/A
### =----------------------------------------------------------------------
### = OUTPUTS:
### =  (  ) N/A
### =======================================================================


### ========================================
### Load the libraries
### ========================================

### Load the developer function script
using SVDapprox
using PyPlot


### ========================================
### Warmup function
### ========================================

### Warmup function
function warm_up(ProcsUse::Integer,Nx::Integer,Ny::Integer)
    ### Warmup
    if nprocs() > ProcsUse
       rmprocs(procs()[end-(nprocs()-ProcsUse-1):end])
    else
       addprocs(ProcsUse-nprocs())
    end
    svd_approx(rand(Nx,Ny))
end


### ========================================
### Run the test
### ========================================

### Test
@printf("\nRand matrices (sharp eigenvalue dropoff)\n")
procs_Use   = 8;
mat_size    = [10,100,1000,10000]
Nx, Ny      = (2000,2000)
par_time_d  = zeros(length(mat_size),1)
par_time_s  = zeros(length(mat_size),1)
serial_time = zeros(length(mat_size),1)
i           = 1

# Warmup
collect(warm_up(8,300,300));

# Loop over sizes
for N in mat_size
   collect(warm_up(p,300,300));
   @printf("   Iter %i/%i procs = %i, m = n = %i\n",i,length(procs_Use),nprocs(),N)
   # AbstractMatrix
   tA = time();
   (U,S,V) = svd_approx(rand(N,N));
   par_time_s[i]  = time() - tA;
   # DArray
   tB = time();
   (U,S,V) = svd_approx(drand(N,N));
   par_time_d[i]  = time() - tB;
    # Serial
   tC = time();
   (U,S,V) = svd(rand(N,N));
   par_time_d[i]  = time() - tC;
   i += 1
end


### ========================================
### Plot
### ========================================

# Hard coded from a test
mat_size    = [10,100,1000,10000]
serial_time = [0.000106359,0.0033429,0.567297418,468.731458991];
par_time_s  = [0.064359209,0.161196264,2.969376102,64.6724925];
par_time_d  = [0.093101816,0.55615837,22.02425065,175.7230821];

# Plot
fig = figure()
p1 = loglog(mat_size.^2,serial_time,"bo-")
p2 = loglog(mat_size.^2,par_time_s,"ro-")
p3 = loglog(mat_size.^2,par_time_d,"go-")
xlab = xlabel(L"Matrix Size, $mn$")
ylab = ylabel(L"Walltime [s]")
setp(p1,"linewidth",3,"MarkerSize",10)
setp(p2,"linewidth",3,"MarkerSize",10)
setp(p3,"linewidth",3,"MarkerSize",10)
legend([p1,p2,p3],loc=2,[L"''svd''",
                         L"''svd_approx'' (AbstractMatrix)",
                         L"''svd_approx'' (DArray)"])
setp(xlab,"FontSize",20)
setp(ylab,"FontSize",20)
savefig("speedup.png")

### =======================================================================
### =                            E   N   D                                =
### =======================================================================
