### =======================================================================
### = error.jl
### = Alex Turner
### = 11/18/2013
### =----------------------------------------------------------------------
### = NOTES:
### =  ( 1) Warm's up julia, then tests the error in the algorithm.
### =----------------------------------------------------------------------
### = SUBFUNCTIONS:
### =  ( 1) run_test :: Runs the approximate SVD and real SVD.
### =----------------------------------------------------------------------
### = INPUTS:
### =  (  ) N/A
### =----------------------------------------------------------------------
### = OUTPUTS:
### =  (  ) N/A
### =======================================================================


### ========================================
### Parameters
### ========================================

### Specify the size of "A" and number of processors
Nx, Ny   = 300, 300
ProcsUse = 4
kUSE     = [2:2:100]


### ========================================
### Functions and libraries
### ========================================

### Load the developer function script
using SVDapprox
using PyPlot

### Runs the approximate SVD with a test matrix
function run_test(A::Union(AbstractMatrix,DArray),k::Integer)

   # Approximate SVD
   t0 = time()
   (Uk,Sk,Vk) = svd_approx(A,k)
   tA = time() - t0
   B = Uk*diagm(Sk)*Vk'

   # Compute the true SVD
   As = try
         convert(Array, A)
       catch
         A
       end
   t0 = time()
   (U,S,V) = svd(As)
   tB = time() - t0

   ### Error Analysis
   err_Approx = (normfro(As - B) / normfro(As))
   err_True   = (normfro(As - U[:,1:k]*diagm(S[1:k])*V[:,1:k]') / normfro(As))
   per_Diff   = (abs(err_True-err_Approx)/err_True*100)
   #@printf("Rank Approximation:    %i/%i\n",k,length(S))
   #@printf("Approximate SVD error: %5.2f\n",err_Approx)
   #@printf("True error:            %5.2f\n",err_True)
   #@printf("Percent Difference:    %5.2f%%\n",per_Diff)

   ### Return the percent difference
   return per_Diff

end


### ========================================
### Warmup and compute the error
### ========================================

### Warmup on four processors
@printf("Warming Up\n\n")
if nprocs() > ProcsUse
   rmprocs(procs()[end-(nprocs()-ProcsUse-1):end])
else
   addprocs(ProcsUse-nprocs())
end
A = rand(Nx,Ny)
collect(svd_approx(rand(Nx,Ny),20,4));
collect(svd_approx(drand(Nx,Ny),20,4));

### Test
@printf("\nRand/Randn matrices (sharp/smooth eigenvalue dropoff)\n")
rand_error  = zeros(length(kUSE),1)
randn_error = zeros(length(kUSE),1)
i           = 1
for k in kUSE
   @printf("   Iter %i/%i: k-rank = %i\n",i,length(kUSE),k)
   rand_error[i]  = run_test(rand(Nx,Ny),k);
   randn_error[i] = run_test(randn(Nx,Ny),k);
   i += 1
end


### ========================================
### Plot
### ========================================

### Plot Rand
# Error
p1 = plot(kUSE,rand_error,"b-")
xlab = xlabel(L"Rank, $k$")
ylab = ylabel(L"Percent Error, $\left|\frac{\epsilon_T - \epsilon_A}{\epsilon_T}\right|\times\,100$")
setp(p1,"linewidth",4)
setp(xlab,"FontSize",20)
setp(ylab,"FontSize",20)
ylim(0,6)
# Singular value spectrum
(U,S,V) = svd(rand(Nx,Ny))
xvals = [1:maximum(kUSE)]
yvals = S[1:maximum(kUSE)]
p1 = plot(xvals,yvals,"bo-")
xlab = xlabel(L"$i$")
ylab = ylabel(L"$\sigma_i$")
setp(p1,"linewidth",4,"MarkerSize",6)
setp(xlab,"FontSize",20)
setp(ylab,"FontSize",20)

### Plot Randn
# Error
p1 = plot(kUSE,randn_error,"b-")
xlab = xlabel(L"Rank, $k$")
ylab = ylabel(L"Percent Error, $\left|\frac{\epsilon_T - \epsilon_A}{\epsilon_T}\right|\times\,100$")
setp(p1,"linewidth",4)
setp(xlab,"FontSize",20)
setp(ylab,"FontSize",20)
ylim(0,6)
# Singular value spectrum
(U,S,V) = svd(randn(Nx,Ny))
xvals = [1:maximum(kUSE)]
yvals = S[1:maximum(kUSE)]
p1 = plot(xvals,yvals,"bo-")
xlab = xlabel(L"$i$")
ylab = ylabel(L"$\sigma_i$")
setp(p1,"linewidth",4,"MarkerSize",6)
setp(xlab,"FontSize",20)
setp(ylab,"FontSize",20)


### =======================================================================
### =                            E   N   D                                =
### =======================================================================
