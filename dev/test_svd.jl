### =======================================================================
### = test_svd.jl
### = Alex Turner
### = 11/18/2013
### =----------------------------------------------------------------------
### = NOTES:
### =  ( 1) Warm's up julia, then tests 4 matrices.
### =----------------------------------------------------------------------
### = SUBFUNCTIONS:
### =  ( 1) run_test :: Runs the approximate SVD with a test matrix.
### =----------------------------------------------------------------------
### = INPUTS:
### =  (  ) N/A
### =----------------------------------------------------------------------
### = OUTPUTS:
### =  (  ) N/A
### =======================================================================


### Load the developer function script
include("svd_approx.jl")

### Runs the approximate SVD with a test matrix
function run_test(A::Union(AbstractMatrix,DArray))

   # Approximate SVD
   @ time begin
      (Uk,Sk,Vk) = svd_approx(A)
   end
   B = Uk*diagm(Sk)*Vk'

   # Compute the true SVD
   As = try
         convert(Array, A)
       catch
         A
       end
   k = length(Sk)
   @ time begin
      (U,S,V) = svd(As)
   end

   ### Error Analysis
   err_Approx = (normfro(As - B) / normfro(As))
   err_True   = (normfro(As - U[:,1:k]*diagm(S[1:k])*V[:,1:k]') / normfro(As))
   @printf("Rank Approximation:    %i/%i\n",k,length(S))
   @printf("Approximate SVD error: %5.2f\n",err_Approx)
   @printf("True error:            %5.2f\n",err_True)
   @printf("Percent Difference:    %5.2f%%\n",(abs(err_True-err_Approx)/err_True*100))

end

### Specify the size of "A" and number of processors
Nx, Ny   = 1000, 1000
ProcsUse = 4

### Warmup on four processors
@printf("Warming Up\n\n")
if nprocs() > ProcsUse
   rmprocs(procs()[end-(nprocs()-ProcsUse-1):end])
else
   addprocs(ProcsUse-nprocs())
end
A = rand(Nx,Ny)
collect(svd_approx(rand(Nx,Ny),20,4,5,1e-3,false));
collect(svd_approx(drand(Nx,Ny),20,4,5,1e-3,false));

### Test with shared rand matrix
@printf("\nShared rand matrix (sharp eigenvalue dropoff)\n")
run_test(rand(Nx,Ny));

### Test with a shared randn matrix
@printf("\nShared randn matrix (smooth eigenvalue dropoff)\n")
run_test(randn(Nx,Ny));

### Test with distributed rand matrix
@printf("\nDArray rand matrix (sharp eigenvalue dropoff)\n")
run_test(drand(Nx,Ny));

### Test with a distributed randn matrix
@printf("\nDArray randn matrix (smooth eigenvalue dropoff)\n")
run_test(drandn(Nx,Ny));

### =======================================================================
### =                            E   N   D                                =
### =======================================================================
