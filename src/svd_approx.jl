### =======================================================================
### = svd_approx.jl
### = Alex Turner
### = 10/15/2013
### =----------------------------------------------------------------------
### = INPUTS:
### =  ( 1) A    :: An [m x n] matrix.
### =  ( 2) k*   :: Rank of the approximation.
### =  ( 3) l*   :: Number of columns to sample at each iteration.
### =  ( 4) N*   :: Maximum number of iterations.
### =  ( 5) tol* :: Tolerance for exiting the iterative process.
### =  ( *) Starred arguments are optional.
### =----------------------------------------------------------------------
### = OUTPUTS:
### =  ( 1) U :: The k largest left-singular vectors [m x k].
### =  ( 2) S :: The k largest singular values [k x 1].
### =  ( 3) V :: The k largest right-singular vectors [n x k].
### =======================================================================

### Functions for default parameters
d_N, d_tol, maxK, facA, facB = 20, 5e-4, 200, 0.08, 0.125
function def_k(A::Union(AbstractMatrix,DArray))
   return int( round( min( minimum(size(A)) * facA, maxK ) ) )
end
function def_l(k::Integer)
   return int( ceil( k * facB ) )
end

### All possible calls of the driver function, sets options if unspecified
svd_approx(A::AbstractMatrix) = svd_approx(A,def_k(A),def_l(def_k(A)))
svd_approx(A::AbstractMatrix,k::Integer) = svd_approx(A,k,def_l(k))
svd_approx(A::AbstractMatrix,k::Integer,l::Integer) = svd_approx(A,k,l)
svd_approx(A::AbstractMatrix,k::Integer,l::Integer,N::Integer) = svd_approx(A,k,l,N,d_tol)
svd_approx(A::AbstractMatrix,k::Integer,l::Integer,N::Integer,tol::Real) = svd_approx(A,k,l,N,tol)
svd_approx(A::DArray) = svd_approx(A,def_k(A),def_l(def_k(A)))
svd_approx(A::DArray,k::Integer) = svd_approx(A,k,def_l(k))
svd_approx(A::DArray,k::Integer,l::Integer) = svd_approx(A,k,l)
svd_approx(A::DArray,k::Integer,l::Integer,N::Integer) = svd_approx(A,k,l,N,d_tol)
svd_approx(A::DArray,k::Integer,l::Integer,N::Integer,tol::Real) = svd_approx(A,k,l,N,tol)

### Driver function
require(Pkg.dir("SVDapprox","src","compute_svd.jl"))
function svd_approx(A::AbstractMatrix,k::Integer,l::Integer,N::Integer=d_N,tol::Real=d_tol)
   (m,n) = size(A)
   return compute_svd(m, n, A, k, l, N, tol)
end
function svd_approx(A::DArray,k::Integer,l::Integer,N::Integer=d_N,tol::Real=d_tol)
   (m,n) = size(A)
   return compute_svd(m, n, A, k, l, N, tol)
end
