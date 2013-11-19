### =======================================================================
### = compute_svd.jl
### = Alex Turner
### = 10/15/2013
### =----------------------------------------------------------------------
### = INPUTS:
### =  ( 1) m   :: Number of rows.
### =  ( 2) n   :: Number of columns.
### =  ( 3) A   :: An [m x n] matrix.
### =  ( 4) k   :: Rank of the approximation.
### =  ( 5) l   :: Number of additional columns to sample at each iteration.
### =  ( 7) N   :: Maximum number of iterations.
### =  ( 8) tol :: Tolerance for exiting the iterative process.
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

### ========================================
### APPROXIMATE SVD ALGORITHM
### ========================================
### Using DArrays
function compute_svd(m::Integer,n::Integer,A::DArray,k::Integer,l::Integer,N::Integer,tol::Real)
   iter, IJ     = true, 1
   cols         = [1:n]
   cols_used    = 0*cols
   cols_curr    = rand_cols(cols, cols_used, k)
   A_tmp        = @parallel (hcat) for i in cols[cols_curr] ; A[:,i]; end
   X            = run_orth(A_tmp)
   (Bo_x, Bo_y) = compute_B(X, A, k)
   old_norm     = compute_norm(Bo_x,Bo_y,k)

   while iter == true
      cols_curr = rand_cols(cols, cols_used, l)
      A_tmp     = @parallel (hcat) for i in cols[cols_curr] ; A[:,i]; end
      X         = run_orth(hcat(X,A_tmp))
      G         = compute_G(X, A, k+l)
      (O, lam)  = eig_G(G, k, l)
      (U, S)    = svd_G(X, O, lam)
      (Bx, By)  = compute_B(U, A, k)
      new_norm  = compute_norm(Bx,By,k)
      rel_error = old_norm / new_norm

      if (IJ >= N) | (rel_error > (1 - tol))
         iter = false
         V    = [By' ./ S]' # Compute the right singular vectors
         return (U, S, V)
      else
         X, Bo_x, Bo_y, old_norm = Bx, Bx, By, new_norm
         IJ += 1
      end
   end
end

### Using Abstract Matrices
function compute_svd(m::Integer,n::Integer,A::AbstractMatrix,k::Integer,l::Integer,N::Integer,tol::Real)
   iter, IJ     = true, 1
   cols         = [1:n]
   cols_used    = 0*cols
   cols_curr    = rand_cols(cols, cols_used, k)
   X            = run_orth(A[:,cols_curr])
   (Bo_x, Bo_y) = compute_B(X, A, k)
   old_norm     = compute_norm(Bo_x,Bo_y,k)
   
   while iter == true
      cols_curr = rand_cols(cols, cols_used, l)
      X         = run_orth(hcat(X,A[:,cols_curr]))
      G         = compute_G(X, A, k+l)
      (O, lam)  = eig_G(G, k, l)
      (U, S)    = svd_G(X, O, lam)
      (Bx, By)  = compute_B(U, A, k)
      new_norm  = compute_norm(Bx,By,k)
      rel_error = old_norm / new_norm
      
      if (IJ >= N) | (rel_error > (1 - tol))
         iter = false
         V    = [By' ./ S]' # Compute the right singular vectors
         return (U, S, V)
      else   
         X, Bo_x, Bo_y, old_norm = Bx, Bx, By, new_norm
         IJ += 1
      end    
   end    
end

### ========================================
### UTILITY FUNCTIONS
### ========================================
### Function to randomly select k columns without repeating
function rand_cols(cols::Vector, cols_used::Vector, k::Integer)
   cols_curr = bool(0*cols)
   for i in 1:k
      tmp_cols        = cols[map(x -> x == minimum(cols_used), cols_used)]
      ind             = tmp_cols[rand(1:length(tmp_cols))]
      cols_used[ind] += 1
      cols_curr[ind]  = true
   end
   return cols_curr
end

### Function to compute an orthonormal set from a set of vectors
function run_orth(cols::AbstractMatrix)
   (Q,tau) = DGEQRF(cols)
   Q       = DORGQR(Q,tau)
   return Q
end

### Function to compute the norm of B without constructing it
function compute_norm(x::AbstractMatrix, y::AbstractMatrix, k::Integer)
   err = @parallel (+) for i in 1:k
      normfro(x)*normfro(y)
   end
   return err
end

### Function to compute B in parallel for a DArray
function compute_B(Bx::AbstractMatrix, A::DArray, k::Integer)
   By = zeros(size(A,2),k)
   for i in 1:k
      By[:,i] = mapreduce( fetch, +, { @spawnat p par_MATMUL('T', A, Bx[:,i]) for p = procs(A) } )
   end
   return (Bx, By)
end

### Function to compute B in parallel for an AbstractMatrix
function compute_B(Bx::AbstractMatrix, A::AbstractMatrix, k::Integer)
   By = @parallel (hcat) for i in 1:k
      DGEMV('T',A,Bx[:,i])
   end
   return (Bx, By)
end

### Function to compute G in parallel for a DArray
function compute_G(X::AbstractMatrix, A::DArray, p::Integer)
   Y = zeros(size(A,2),p)
   for i in 1:p
      Y[:,i] = mapreduce( fetch, +, { @spawnat pp par_MATMUL('T', A, X[:,i]) for pp = procs(A) } )
   end
   G = @parallel (hcat) for i in 1:p
      DGEMV('T',Y,Y[:,i])
   end
   return G
end

### Function to compute G in parallel for an AbstractMatrix
function compute_G(X::AbstractMatrix, A::AbstractMatrix, p::Integer)
   Y = @parallel (hcat) for i in 1:p
      DGEMV('T',A,X[:,i])
   end
   G = @parallel (hcat) for i in 1:p
      DGEMV('T',Y,Y[:,i])
   end
   return G
end

### Function to compute the eigenvalues and orthonormal eigenvectors of G
function eig_G(G::AbstractMatrix, k::Integer, l::Integer)
   (lam, O) = eig(G)
   O = @parallel (hcat) for i in 1:k+l
      O[:,i] ./ sqrt( dot( O[:,i], O[:,i] ) )
   end
   ind = sortperm(lam, rev=true)
   lam = lam[ind]
   O   = O[:,ind]
   return (O[:,1:k], lam[1:k])
end

### Function to compute the SVD of G
function svd_G(X::AbstractMatrix, O::AbstractMatrix, lam::Vector)
   U   = DGEMM('N','N',float64(X),float64(O))
   lam = sqrt(lam)
   return (U, lam)
end

### Function to do the local matrix multiplication
function par_MATMUL(TRANS::Char, A::DArray, B::Vector)
   if TRANS == 'T'
      (indR,indC), B_out = myindexes(A), zeros(size(A,2),1)
   else
      (indC,indR), B_out = myindexes(A), zeros(size(A,1),1)
   end
   B_out[indC] = DGEMV(TRANS, float64(localpart(A)), float64(B[indR]))
   return B_out
end
