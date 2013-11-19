### =======================================================================
### = compute_svd_dist.jl
### = Alex Turner
### = 10/15/2013
### =----------------------------------------------------------------------
### = NOTES:
### =  ( 1) Actual code to comptue the approximate svd.
### =  ( 2) Called by svd_approx.jl.
### =  ( 3) Computes the k-rank approximation of A using the SVD.
### =  ( 4) Uses distributed arrays.
### =----------------------------------------------------------------------
### = SUBFUNCTIONS:
### =  ( 1) compute_svd  :: Actually compute the svd.
### =  ( 2) rand_cols    :: Randomly choose new columns from A.
### =  ( 3) run_orth     :: Computes an orthornomal set.
### =  ( 4) compute_norm :: Compute the norm of B from the compact version.
### =  ( 5) compute_B    :: Compute B using the orthonormal vectors and A.
### =  ( 6) compute_G    :: Compute G given the orthonormal vectors and A.
### =  ( 7) eig_G        :: Compute the eigenvectors/values of G.
### =  ( 8) svd_G        :: Compute the SVD of G given the eigenvectors.
### =  ( 9) par_MATMUL   :: Does a local matmul with a DArray.
### =----------------------------------------------------------------------
### = INPUTS:
### =  ( 1) m   :: Number of rows.
### =  ( 2) n   :: Number of columns.
### =  ( 3) A   :: An [m x n] matrix.
### =  ( 4) k   :: Rank of the approximation.
### =  ( 5) l   :: Number of additional columns to sample at each iteration.
### =  ( 7) N   :: Maximum number of iterations.
### =  ( 8) tol :: Tolerance for exiting the iterative process.
### =  ( 9) pro :: Profile the code?
### =----------------------------------------------------------------------
### = OUTPUTS:
### =  ( 1) U   :: The k largest left-singular vectors [m x k].
### =  ( 2) S   :: The k largest singular values [k x 1].
### =  ( 3) V   :: The k largest right-singular vectors [n x k].
### =======================================================================


### Define the level-2/level-3 BLAS and LAPACK routines
import Base.LinAlg: BLAS.gemv, BLAS.gemm, LAPACK.geqrf!, LAPACK.orgqr!
DGEMV,  DGEMM  = Base.LinAlg.BLAS.gemv,     Base.LinAlg.BLAS.gemm
DGEQRF, DORGQR = Base.LinAlg.LAPACK.geqrf!, Base.LinAlg.LAPACK.orgqr!


### Function to compute the SVD
function compute_svd_dist(m::Integer,n::Integer,A::DArray,k::Integer,l::Integer,N::Integer,tol::Real,pro::Bool)

   ### Keep track of the number of times we've called a column
   tot_time  = time()
   cols      = [1:n]
   cols_used = 0*cols

   ### Randomly choose k columns without repeating
   t0 = time()
   cols_curr = rand_cols(cols, cols_used, k)
   t1 = time() - t0;

   ### Get the orthonormal set
   t0 = time()
   A_tmp = @parallel (hcat) for i in cols[cols_curr] ; A[:,i]; end
   X = run_orth(A_tmp)
   t2 = time() - t0;

   ### Compute B0
   t0 = time()
   (Bo_x, Bo_y) = compute_B(X, A, k)
   t3 = time() - t0;

   ### Compute the norm of B0
   t0 = time()
   old_norm = compute_norm(Bo_x,Bo_y,k)
   t4 = time() - t0;

   ### Profile the code
   if pro
      tot=(t1+t2+t3+t4)/100
      t1=t1/tot;t2=t2/tot;t3=t3/tot;t4=t4/tot
      @printf("ITER  0: rand_cols    - %5.2f%s\n",t1,'%')
      @printf("ITER  0: run_orth     - %5.2f%s\n",t2,'%')
      @printf("ITER  0: compute_B    - %5.2f%s\n",t3,'%')
      @printf("ITER  0: compute_norm - %5.2f%s\n",t4,'%')
      @printf("ITER  0: %12.8f seconds\n\n",tot*100)
   end

   ### Begin iterating
   iter = true
   IJ   = 1
   while iter == true

      ### Randomly choose l columns without repeating
      t0 = time()
      cols_curr = rand_cols(cols, cols_used, l)
      t5 = time() - t0;

      ### Get the orthonormal set
      t0 = time()
      A_tmp = @parallel (hcat) for i in cols[cols_curr] ; A[:,i]; end
      X = run_orth(hcat(X,A_tmp))
      t6 = time() - t0;

      ### Construct G
      t0 = time()
      G = compute_G(X, A, k+l)
      t7 = time() - t0;

      ### Get the eigenvectors and eigenvalues of G
      t0 = time()
      (O, lam) = eig_G(G, k, l)
      t8 = time() - t0;

      ### Get the SVD
      t0 = time()
      (U, S) = svd_G(X, O, lam)
      t9 = time() - t0;

      ### Use the SVD to get our new B
      t0 = time()
      (Bx, By) = compute_B(U, A, k)
      t10 = time() - t0;

      ### Get the relative improvement
      t0 = time()
      new_norm  = compute_norm(Bx,By,k)
      t11 = time() - t0;
      rel_error = old_norm / new_norm

      ### Profile the code
      if pro
         tot=(t5+t6+t7+t8+t9+t10+t11)/100
         t5=t5/tot;t6=t6/tot;t7=t7/tot;t8=t8/tot
         t9=t9/tot;t10 = t10/tot;t11=t11/tot
         @printf("ITER %2d: rand_cols    - %5.2f%s\n",IJ,t5,'%')
         @printf("ITER %2d: run_orth     - %5.2f%s\n",IJ,t6,'%')
         @printf("ITER %2d: compute_G    - %5.2f%s\n",IJ,t7,'%')
         @printf("ITER %2d: eig_G        - %5.2f%s\n",IJ,t8,'%')
         @printf("ITER %2d: svd_G        - %5.2f%s\n",IJ,t9,'%')
         @printf("ITER %2d: compute_B    - %5.2f%s\n",IJ,t10,'%')
         @printf("ITER %2d: compute_norm - %5.2f%s\n",IJ,t11,'%')
         @printf("ITER %2d: %12.8f seconds\n\n",IJ,tot*100)
      end

      ### Exit conditions
      if (IJ >= N) | (rel_error > (1 - tol))
         iter = false
         V    = [By' ./ S]' # Compute the right singular vectors
         if pro
            @printf("\nExited at iter %2d in %3.2f seconds\n\n",IJ,(time()-tot_time))
         end
         return (U, S, V)
      else
         X, Bo_x, Bo_y, old_norm = Bx, Bx, By, new_norm
         IJ += 1
      end

   end

end


### Function to randomly select k columns without repeating
function rand_cols(cols::Vector, cols_used::Vector, k::Integer)

   ### Initialize the boolean array
   cols_curr = bool(0*cols)

   ### Select the columns
   for i in 1:k
      tmp_cols        = cols[map(x -> x == minimum(cols_used), cols_used)]
      ind             = tmp_cols[rand(1:length(tmp_cols))]
      cols_used[ind] += 1
      cols_curr[ind]  = true
   end

   ### Return the boolean array
   return cols_curr

end


### Function to compute an orthonormal set from a set of vectors
function run_orth(cols::AbstractMatrix)

   ### Return the orthonormal set and extract "Q" (DGEQRF/DORGQR = 3x faster than qr)
   # (Q,_) = qr(cols)
   (Q,tau) = DGEQRF(cols)
   Q       = DORGQR(Q,tau)
   return Q

end


### Function to compute the norm of B without constructing it
function compute_norm(x::AbstractMatrix, y::AbstractMatrix, k::Integer)

   ### There are k summations so we can break this up into k computations
   err = @parallel (+) for i in 1:k
      normfro(x)*normfro(y)
   end

   ### Return the parts of the matrix
   return (err)

end


### Function to compute B in parallel
function compute_B(Bx::AbstractMatrix, A::DArray, k::Integer)

   ### Do a local matmul on each processor
   By = zeros(size(A,2),k)
   for i in 1:k
      By[:,i] = mapreduce( fetch, +, { @spawnat p par_MATMUL('T', A, Bx[:,i]) for p = procs(A) } )
   end

   ### Return the parts of the matrix
   return (Bx, By)

end


### Function to compute G in parallel
function compute_G(X::AbstractMatrix, A::DArray, p::Integer)

   ### Do a local matmul on each processor
   Y = zeros(size(A,2),p)
   for i in 1:p
      Y[:,i] = mapreduce( fetch, +, { @spawnat pp par_MATMUL('T', A, X[:,i]) for pp = procs(A) } )
   end

   ### Fill G
   G = @parallel (hcat) for i in 1:p
      DGEMV('T',Y,Y[:,i])
   end

   ### Return G
   return G

end


### Function to compute the eigenvalues and orthonormal eigenvectors of G
function eig_G(G::AbstractMatrix, k::Integer, l::Integer)

   ### Size of G
   p = k + l

   ### Compute the eigenvalues and eigenvectors
   (lam, O) = eig(G)

   ### Make sure the vectors are orthonormal
   O = @parallel (hcat) for i in 1:p
      O[:,i] = O[:,i] ./ sqrt( dot( O[:,i], O[:,i] ) )
   end

   ### Sort them based on the size of the eigenvalue
   ind = sortperm(lam, rev=true)
   lam = lam[ind]
   O   = O[:,ind]

   ### Return the k largest eigenvalues and orthonormal eigenvectors
   return (O[:,1:k], lam[1:k])

end


### Function to compute the SVD of G
function svd_G(X::AbstractMatrix, O::AbstractMatrix, lam::Vector)

   ### Get the left-singular vectors and singular values
   U   = DGEMM('N','N',float64(X),float64(O))
   lam = sqrt(lam)

   ### Return the left-singular vectors and singular values
   return (U, lam)

end


### Function to do the local matrix multiplication
function par_MATMUL(TRANS::Char, A::DArray, B::Vector)

   ### Get the indices and subarray
   if TRANS == 'T'
      (indR,indC), B_out = myindexes(A), zeros(size(A,2),1)
   else
      (indC,indR), B_out = myindexes(A), zeros(size(A,1),1)
   end

   ### Matrix multiplication
   B_out[indC] = DGEMV(TRANS, float64(localpart(A)), float64(B[indR]))

   ### Return our contribution
   return B_out
   
end

### =======================================================================
### =                            E   N   D                                =
### =======================================================================
