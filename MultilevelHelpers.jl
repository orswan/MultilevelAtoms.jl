# MultilevelHelpers.jl
# Requires: QuantumOptics

import SparseArrays.spdiagm
import LinearAlgebra.diagm, LinearAlgebra.rank
import WignerSymbols.clebschgordan
import RationalRoots

################################ State structs #######################################

#------------------ Multiplets --------------------------

RationalOrInt = (Union{T,R} where T<:Integer where R<:Rational)

struct Multiplet <: Basis
    shape::Vector{Int}
    r::RationalOrInt          # Total angular momentum
    q::Dict                   # Dictionary of other quantum numbers
    mu::Real                  # Magnetic moment
    name::String              # Name of multiplet
    Multiplet(r,q,mu,name) = new(Int[2r+1],r,q,mu,name)
end

LNames = ["S","P","D","F","G","H","I","K","L","M","N"]
function lsString(S::RationalOrInt,L::Int,J::RationalOrInt)
    L+1 <= length(LNames) ? string(Integer(2S+1))*"'"*string(LNames[L+1])*"'"*string(J) : string(Integer(2S+1))*"'"*string(L)*"'"*string(J)
end

function Multiplet(S::RationalOrInt,L::Int,J::RationalOrInt)
    Multiplet(J,Dict(:S=>S,:L=>L),muMoment(S,L,J),lsString(S,L,J))
end

function mLevelGamma(gamma::Number,j₁::Multiplet,m₁::RationalOrInt,j₂::Multiplet,m₂::RationalOrInt,q::Int)
    # j₁, m₁ are lower level parameters, j₂, m₂ upper level parameters, and q is the light polarization.
    return gamma * clebschgordan(j₁, m₁, 1, q, j₂, m₂)^2
end

#------------------- Atoms ---------------------------------

abstract type AbstractAtom <: Basis end

struct AtomBasis <: AbstractAtom
    shape::Vector{Int}
    multiplets::Vector{Multiplet}
    AtomBasis(multiplets::Vector{Multiplet}) = new(sum(m.shape for m in multiplets),multiplets)
end

struct Atom0D <: AbstractAtom
    shape::Vector{Int}
    multiplets::Vector{Multiplet}
    energies::Vector       # Energies of each multiplet in absense of external fields
    gammas::Matrix         # Lifetimes γ between pairs of multiplets.  <s'||d||s>^2 = γ_{s',s} * (3 \epsilon_0 \hbar \lambda^3)/8pi^2
    units::String
    Atom0D(multiplets::Vector{Multiplet},energies::Vector,gammas::Matrix,units::String) = 
        new(sum(m.shape for m in multiplets),multiplets,energies,gammas,units)
end

struct BoostedAtom1D <: AbstractAtom
    shape::Vector{Int}
    multiplets::Vector{Multiplet}
    energies::Vector       # Energies of each multiplet in absense of external fields
    gammas::Matrix         # Lifetimes γ between pairs of multiplets.  <s'||d||s>^2 = γ_{s',s} * (3 \epsilon_0 \hbar \lambda^3)/8pi^2
    beta::Number           # v/c
    units::String
    BoostedAtom1D(multiplets::Vector{Multiplet},energies::Vector,gammas::Matrix,beta::Number,units::String) = 
        new(sum(m.shape for m in multiplets),multiplets,energies,gammas,beta,units)
end

struct Atom1D <: AbstractAtom
    shape::Vector{Int}
    momentum::MomentumBasis
    multiplets::Vector{Multiplet}
    energies::Vector       # Energies of each multiplet in absense of external fields
    gammas::Matrix         # Lifetimes γ between pairs of multiplets.  <s'||d||s>^2 = γ_{s',s} * (3 \epsilon_0 \hbar \lambda^3)/8pi^2
    units::String
    Atom1D(momentum::MomentumBasis,multiplets::Vector{Multiplet},energies::Vector,gammas::Matrix,units::String) = 
        new(sum(m.shape for m in multiplets),momentum,multiplets,energies,gammas,units)
end

function subAttribute(x::AbstractArray,indices::Vector)
    N = ndims(x)
    getindex(x,(indices for i in 1:N)...)
end
subAttribute(x,indices::Vector) = x       # Default case for non-arrays.

function subAtom(a::T,indices::Vector) where T<:AbstractAtom
    all( i -> (i isa Integer) && 1<=i<=length(a.multiplets) , indices) || error("Invalid indices.")
    T( (subAttribute(getfield(a,s),indices) for s in fieldnames(T)[2:end])... )
end

#------------------- Atom basis indices ---------------------

function atomStateIndex(a::AbstractAtom,multipletIdx::Int,mstate::RationalOrInt)
    return convert(Int, sum( (m.shape[1] for m in a.multiplets[1:multipletIdx-1]) ,init = 0) + mstate + a.multiplets[multipletIdx].r + 1)
end

atomStateIndex(a::AbstractAtom,name::String,mstate::RationalOrInt) = atomStateIndex(a, findfirst(m->m.name==name,a.multiplets), mstate)

function atomMultipletIndices(a::AbstractAtom,multipletIdx::Int)
    mp = a.multiplets[multipletIdx]
    return atomStateIndex(a,multipletIdx,-mp.r):atomStateIndex(a,multipletIdx,mp.r)
end

atomMultipletIndices(a::AbstractAtom,name::String) = atomMultipletIndices(a, findfirst(m->m.name==name,a.multiplets))

function atomIndexMultiplet(a::AbstractAtom,i::Integer)
    # Returns (multiplet_index, m_index, m) where the first is the index of the multiplet in a.multiplets,
    # the second is the index of the m state in the corresponding multiplet, and the third is the m state eigenvalue. 
    i <= a.shape[1] && i > 0 || error("Index $(i) out of Atom index range $(1:a.shape[1]).")
    idxs = cumsum(m.shape[1] for m in a.multiplets)
    plet = findfirst(i .<= idxs)
    m = a.multiplets[plet].r - (idxs[plet] - i)
    mIdx = m + 1 + a.multiplets[plet].r
    return plet, mIdx, m
end

#------------------- Lasers and couplings --------------------

struct Laser
    a::AbstractAtom
    nearLevels::Vector{Tuple{Int,Int}}
    amplitude::Number
    polarization::Tuple{Number,Number,Number}
    frequency::Number
end

struct Coupling
    a::AbstractAtom
    lower::Int
    upper::Int
    reducedRabi::Number
    polarization::Tuple{Number,Number,Number}
    detuning::Number
end

########################### Vector operators ###########################################################

# Vector operator helper functions

function vecOpSparseData(T::DataType,l::Multiplet,u::Multiplet,q::Int)
    # For spherical component q of a vector operator and a lower and upper multiplet, 
    # this function returns the indices of the lower and upper multiplet which are coupled
    # and the corresponding Clebsch-Gordan coefficients.  These are returned as three iterables,
    # I, J, C, with I[i] coupled to J[i] with coefficient C[i].
    idxl = (max(-l.r,-u.r-q):min(l.r,u.r-q)) .+ (1+l.r)
    idxu = (max(-u.r,-l.r+q):min(u.r,l.r+q)) .+ (1+u.r)
    coef = Vector{T}([clebschgordan(l.r,m,1,q,u.r) for m in (-l.r:l.r)[idxl]])
    return idxl, idxu, coef
end
vecOpSparseData(l::Multiplet,u::Multiplet,q::Int) = vecOpSparseData(RationalRoots.RationalRoot{BigInt},l,u,q)
# Example: I, J, C = vecOpSparseData(Sr88.multiplets[1],Sr88.multiplets[2],1)

function atomVectorOperator(a::AbstractAtom,lower::Int,upper::Int,q::Int)
    # Constructs a symmetric sparse matrix of level Clebsch-Gordan couplings. 
    # q = -1, 0, 1 indexes a spherical vector component of a vector operator. 
    I, J, C = vecOpSparseData(a.multiplets[lower],a.multiplets[upper],q)
    return sparse([atomMultipletIndices(a,lower)[I]... , atomMultipletIndices(a,upper)[J]...], [atomMultipletIndices(a,upper)[J]... , atomMultipletIndices(a,lower)[I]...], vcat(C,C),a.shape[1],a.shape[1])
end

# Example: atomVectorOperator(Sr88,1,4,0)


########################### Magnetic structure #########################################################

gs = −2.00231930436256                                           # Electron g-factor
lande(S,L,J) = ( ( J*(J+1) - S*(S+1) + L*(L+1) ) + gs*( J*(J+1) + S*(S+1) - L*(L+1) ) )/(2*J*(J+1))
muB = 1.400                                                      # Units of h MHz/Gauss
muMoment(S,L,J) = iszero(J) ? 0.0 : -muB * lande(S,L,J) * J      # Magnetic moment (LS coupling?)

function zeeman(a::AbstractAtom,B::Number)
    spdiagm( [(B * a.multiplets[atomIndexMultiplet(a,i)[1]].mu * atomIndexMultiplet(a,i)[3] for i=1:a.shape[1])...] )
end

########################### Strontium 88 ##################################################################

sr88_1S0 = Multiplet(0,0,0)
sr88_1P1 = Multiplet(0,1,1)
sr88_3P0 = Multiplet(1,1,0)
sr88_3P1 = Multiplet(1,1,1)
sr88_3P2 = Multiplet(1,1,2)
sr88_3S1 = Multiplet(1,0,1)

sr88_energies = [0,650.503499,429.22800422987365,434.829121311,446.647242704,(434.829121311+435.731497)] * 1e6    # In MHz
sr88_gammas = [0 2.01e8 0 4.69e4 1.2e-2 0 ; 0 0 0 0 0 0 ; 0 0 0 0 0 8.9e6 ; 0 0 0 0 0 2.7e7 ; 0 0 0 0 0 4.2e7 ; 0 0 0 0 0 0] .* (1e-6)
Sr88 = Atom0D([sr88_1S0,sr88_1P1,sr88_3P0,sr88_3P1,sr88_3P2,sr88_3S1],sr88_energies,sr88_gammas,"MHz")

########################## Time-reducible couplings ##############################################################

#------------------------- Coupling graph helpers ----------------------------

function adjacencyMatrix(edges::Vector{Tuple{Int,Int}})
    m = max((e[i] for e in edges, i in (1,2))...)
    A = zeros(Int,m,m)
    for e in edges
        A[e[1],e[2]] += 1
        A[e[2],e[1]] += 1
    end
    return A
end

function graphLaplacian(edges::Vector{Tuple{Int,Int}})
    A = adjacencyMatrix(edges)
    degrees = [sum(A,dims=1)...]
    return diagm(degrees) - A
end

function isForest(edges::Vector{Tuple{Int,Int}})
    L = graphLaplacian(edges)
    return tr(L) == rank(L)*2
end

function ditreeRank(es::Vector{Tuple{I,I,R}},n::Integer;returnClusters=false) where {I<:Integer, R<:Real}
    clusters = [Set(i) for i=1:n]
    clusterIdx = [1:n...]
    ranks = zeros(n)
    for e in es
        ci1 = clusterIdx[e[1]]
        ci2 = clusterIdx[e[2]]
        r1 = ranks[e[1]]
        r2 = ranks[e[2]]
        for i in clusters[ci2]
            ranks[i] += e[3]+r1-r2
            clusterIdx[i] = ci1
        end
        clusters[ci1] = union(clusters[ci1],clusters[ci2])
        clusters[ci2] = Set()
    end
    if returnClusters
        return ranks, clusters
    else
        return ranks
    end
end

function ditreeRank(es::Vector{Tuple{I,I,R}};returnClusters=false) where {I<:Integer, R<:Real}
    n = max((e[1] for e in es)...,(e[2] for e in es)...)
    ditreeRank1(es,n;returnClusters)
end

#------------------------ Hamiltonian matrix constructors ---------------------

# Time independent interaction Hamiltonians and Lindbladians.
# tr in the names stands for time reducible

function trCouplingMatrix(c::Coupling,diags=1)
    if any(p==1 for p in c.polarization) && sum(c.polarization)==1
        q = findfirst(p==1 for p in c.polarization) - 2
        HInt = c.reducedRabi * atomVectorOperator(c.a,c.lower,c.upper,q)
        HDetuning = sparse(atomMultipletIndices(c.a,c.upper),atomMultipletIndices(c.a,c.upper),c.detuning,c.a.shape[1],c.a.shape[1])
        return HInt + HDetuning * diags
    else
        Hm = c.reducedRabi * atomVectorOperator(c.a,c.lower,c.upper,-1) * c.polarization[1]
        H0 = c.reducedRabi * atomVectorOperator(c.a,c.lower,c.upper,0) * c.polarization[2]
        Hp = c.reducedRabi * atomVectorOperator(c.a,c.lower,c.upper,1) * c.polarization[3]
        HDetuning = sparse(atomMultipletIndices(c.a,c.upper),atomMultipletIndices(c.a,c.upper),c.detuning,c.a.shape[1],c.a.shape[1])
        return Hm + H0 + Hp + HDetuning * diags
    end
end

function multipletDiagonal(a::AbstractAtom,ds::Vector)
    length(ds) == length(a.multiplets) || error("Diagonal vector and multiplet vector must have same length.")
    return spdiagm(vcat((fill(ds[i],a.multiplets[i].shape[1]) for i in 1:length(ds))...))
end

function trTree(vc::Vector{Coupling})
    edges = [(c.lower,c.upper) for c in vc]
    isForest(edges) || error("Coupling topology has loops.  Cannot eliminate time dependence.")
    all(c.a == vc[1].a for c in vc) ? ab = vc[1].a : error("All couplings must be for the same atom.")
    offDiag = sum(trCouplingMatrix(c,0) for c in vc)
    
    diedges = [(c.lower,c.upper,c.detuning) for c in vc]
    HDetuning = multipletDiagonal(ab,ditreeRank(diedges,length(ab.multiplets)))
    return offDiag + HDetuning
end


####################### Lindblad operators ########################################################################


function lindblads(a::AbstractAtom; combine::Bool=false)
    Js = []
    γs = []
    for i=1:size(a.gammas,1), j=1:size(a.gammas,1)
        if ~iszero(a.gammas[i,j])
            lower = a.multiplets[i]
            upper = a.multiplets[j]
            for m=-lower.r:lower.r
                for q=-1:1
                    if (m+q) in -upper.r:upper.r
                        cg = clebschgordan(lower.r, m, 1, q, upper.r)
                        if ~iszero(cg)
                            idxl = atomStateIndex(a,i,m)
                            idxu = atomStateIndex(a,j,m+q)
                            J = sparse([idxl],[idxu],[cg],a.shape[1],a.shape[1])
                            push!(Js,J)
                            push!(γs,a.gammas[i,j])
                        end
                    end
                end
            end
        end
    end
    combine ? [Js[i]*sqrt(γs[i]) for i=1:length(Js)] : (Js, γs)
end


####################### Plotting and displaying ###################################################################

function pletPlot(a::AbstractAtom,tout::Vector,pops::Vector{Operator{T,T, Matrix{ComplexF64}}}) where T<:Basis
    p = plot()
    for plet=1:length(a.multiplets)
        for i in atomMultipletIndices(A,plet)
            tmp,mIdx,m = atomIndexMultiplet(A,i)
            plot!(p,tout,[real(pop.data[i,i]) for pop in pops], label = A.multiplets[plet].name * ", m=$(m)", linewidth=2.5, thickness_scaling=1, linestyle=[:solid, :dash, :dot, :dashdot, :dashdotdot][((plet-1) % 5)+1])
        end
    end
    p
end

printFull(M::AbstractMatrix) = show(IOContext(stdout, :limit=>false), MIME"text/plain"(),M |> Matrix)