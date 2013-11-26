### =======================================================================
### = svd_approx.jl
### = Alex Turner
### = 10/15/2013
### =----------------------------------------------------------------------
### = NOTES:
### =  ( 1) Driver function for the approximate SVD.
### =  ( 2) All possible calls to the approximate SVD.
### =  ( 3) Sets default tolerances if none are specified.
### =  ( 4) Only required input is "A".
### =  ( 5) SVD is actually computed in "compute_svd.jl".
### =  ( 6) Will use a different parallel matmul for shared vs distributed.
### =----------------------------------------------------------------------
### = SUBFUNCTIONS:
### =  ( 1) svd_approx :: Driver function that is called.
### =  ( 2) def_k      :: Sets the rank approxmation based on "A".
### =  ( 3) def_l      :: Set the number of columns to read based on "k".
### =----------------------------------------------------------------------
### = INPUTS:
### =  ( 1) A    :: An [m x n] matrix.
### =  ( 2) k*   :: Rank of the approximation.
### =  ( 3) l*   :: Number of columns to sample at each iteration.
### =  ( 4) N*   :: Maximum number of iterations.
### =  ( 5) tol* :: Tolerance for exiting the iterative process.
### =  ( 6) pro* :: Profile the code?
### =  ( *) Starred arguments are optional.
### =----------------------------------------------------------------------
### = OUTPUTS:
### =  ( 1) U :: The k largest left-singular vectors [m x k].
### =  ( 2) S :: The k largest singular values [k x 1].
### =  ( 3) V :: The k largest right-singular vectors [n x k].
### =======================================================================


### Functions for default parameters
d_N,  d_tol, d_pro = 20,  5e-4, true
maxK, facA,  facB  = 200, 0.08, 0.125
function def_k(A::Union(AbstractMatrix,DArray))
   return int( round( min( minimum(size(A)) * facA, maxK ) ) )
end
function def_l(k::Integer)
   return int( ceil( k * facB ) )
end


### All possible calls of the driver function, sets options if unspecified
# Abstract Matrix
svd_approx(A::AbstractMatrix) = svd_approx(A,def_k(A),def_l(def_k(A)))
svd_approx(A::AbstractMatrix,k::Integer) = svd_approx(A,k,def_l(k))
svd_approx(A::AbstractMatrix,k::Integer,l::Integer) = svd_approx(A,k,l)
svd_approx(A::AbstractMatrix,k::Integer,l::Integer,N::Integer) = svd_approx(A,k,l,N,d_tol,d_pro)
svd_approx(A::AbstractMatrix,k::Integer,l::Integer,N::Integer,tol::Real) = svd_approx(A,k,l,N,tol,d_pro)
svd_approx(A::AbstractMatrix,k::Integer,l::Integer,N::Integer,tol::Real,pro::Bool) = svd_approx(A,k,l,N,tol,pro)
# DArray
svd_approx(A::DArray) = svd_approx(A,def_k(A),def_l(def_k(A)))
svd_approx(A::DArray,k::Integer) = svd_approx(A,k,def_l(k))
svd_approx(A::DArray,k::Integer,l::Integer) = svd_approx(A,k,l)
svd_approx(A::DArray,k::Integer,l::Integer,N::Integer) = svd_approx(A,k,l,N,d_tol,d_pro)
svd_approx(A::DArray,k::Integer,l::Integer,N::Integer,tol::Real) = svd_approx(A,k,l,N,tol,d_pro)
svd_approx(A::DArray,k::Integer,l::Integer,N::Integer,tol::Real,pro::Bool) = svd_approx(A,k,l,N,tol,pro)


### Driver functions
# Abstract Matrix
function svd_approx(A::AbstractMatrix,k::Integer,l::Integer,N::Integer=d_N,tol::Real=d_tol,pro::Bool=d_pro)
   require("compute_svd_shared.jl")
   (m,n) = size(A)
   return compute_svd_shared(m, n, A, k, l, N, tol, pro)
end
# DArray
function svd_approx(A::DArray,k::Integer,l::Integer,N::Integer=d_N,tol::Real=d_tol,pro::Bool=d_pro)
   require("compute_svd_dist.jl")
   (m,n) = size(A)
   return compute_svd_dist(m, n, A, k, l, N, tol, pro)
end


### =======================================================================
### =                            E   N   D                                =
### =======================================================================
