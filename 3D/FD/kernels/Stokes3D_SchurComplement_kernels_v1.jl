@parallel_indices (i,j,k) function UpdatePressure_SchurComplement!( P, bP, ∇V, rheo, γ )
    if i<=size(P,1)-2 && j<=size(P,2)-2 && k<=size(P,3)-2
        P[i+1,j+1,k+1] += γ.*(bP[i+1,j+1,k+1] - ∇V[i+1,j+1,k+1])
    end
    return nothing
end

@parallel_indices (i,j,k) function ComputeStress_SchurComplement!( τ, ε̇, ∇V, rheo, γ )
    if i<=size(ε̇.xx,1)-2 && j<=size(ε̇.xx,2)-2 && k<=size(ε̇.xx,3)-2
        ηve_c_x_2 = 2.0* rheo.ηve_c[i+1,j+1,k+1]
        γ_x_∇V    = γ*∇V[i+1,j+1,k+1]
        τ.xx[i+1,j+1,k+1] = ηve_c_x_2 * ε̇.xx[i+1,j+1,k+1] + γ_x_∇V
        τ.yy[i+1,j+1,k+1] = ηve_c_x_2 * ε̇.yy[i+1,j+1,k+1] + γ_x_∇V
        τ.zz[i+1,j+1,k+1] = ηve_c_x_2 * ε̇.zz[i+1,j+1,k+1] + γ_x_∇V
    end
    if i<=size(ε̇.xy,1) && j<=size(ε̇.xy,2) && k<=size(ε̇.xy,3)-2
        τ.xy[i,j,k+1] = 2.0 * rheo.ηve_xy[i,j,k] * ε̇.xy[i,j,k+1]
    end
    if i<=size(ε̇.xz,1) && j<=size(ε̇.xz,2)-2 && k<=size(ε̇.xz,3)
       τ.xz[i,j+1,k] =  2.0 * rheo.ηve_xz[i,j,k] * ε̇.xz[i,j+1,k]
    end
    if i<=size(ε̇.yz,1)-2 && j<=size(ε̇.yz,2) && k<=size(ε̇.yz,3)
        τ.yz[i+1,j,k] = 2.0 * rheo.ηve_yz[i,j,k] * ε̇.yz[i+1,j,k]
    end
    return nothing
end

@parallel_indices (i,j,k) function ComputeResiduals_SchurComplement!( R, τ, b, D, Δ, rhs )
    if i<=size(R.x,1) && j<=size(R.x,2)-2 && k<=size(R.x,3)-2
        if i>1 && i<size(R.x,1) # avoid Dirichlets
            R.x[i,j+1,k+1]  = rhs*b.x[i,j+1,k+1]
            R.x[i,j+1,k+1] += (τ.xx[i+1,j+1,k+1] - τ.xx[i,j+1,k+1]) / Δ.x
            R.x[i,j+1,k+1] += (τ.xy[i,j+1,k+1] - τ.xy[i,j,k+1]) / Δ.y
            R.x[i,j+1,k+1] += (τ.xz[i,j+1,k+1] - τ.xz[i,j+1,k]) / Δ.z
            R.x[i,j+1,k+1] *= -1. 
        end
    end
    if i<=size(R.y,1)-2 && j<=size(R.y,2) && k<=size(R.y,3)-2
        if j>1 && j<size(R.y,2) # avoid Dirichlets
            R.y[i+1,j,k+1]  = rhs*b.y[i+1,j,k+1]
            R.y[i+1,j,k+1] += (τ.yy[i+1,j+1,k+1] - τ.yy[i+1,j,k+1]) / Δ.y
            R.y[i+1,j,k+1] += (τ.xy[i+1,j,k+1] - τ.xy[i,j,k+1]) / Δ.x
            R.y[i+1,j,k+1] += (τ.yz[i+1,j,k+1] - τ.yz[i+1,j,k]) / Δ.z
            R.y[i+1,j,k+1] *= -1.
        end
    end
    if i<=size(R.z,1)-2 && j<=size(R.z,2)-2 && k<=size(R.z,3)
        if k>1 && k<size(R.z,3) # avoid Dirichlets
            R.z[i,j+1,k+1]  = rhs*b.z[i,j+1,k+1]
            R.z[i,j+1,k+1] += (τ.zz[i+1,j+1,k+1] - τ.zz[i+1,j+1,k]) / Δ.z
            R.z[i,j+1,k+1] += (τ.xz[i+1,j+1,k] - τ.xz[i,j+1,k]) / Δ.x
            R.z[i,j+1,k+1] += (τ.yz[i+1,j+1,k] - τ.yz[i+1,j,k]) / Δ.y
            R.z[i,j+1,k+1] *= -1.
        end
    end
    return nothing
end

@parallel_indices (i,j,k) function ComputeResiduals_SchurComplement_v2!( R, b, τ, P, Δ )
    if i<=size(R.x,1) && j<=size(R.x,2)-2 && k<=size(R.x,3)-2
        if i>1 && i<size(R.x,1) # avoid Dirichlets
            R.x[i,j+1,k+1]  = (τ.xx[i+1,j+1,k+1] - τ.xx[i,j+1,k+1]) / Δ.x
            R.x[i,j+1,k+1] -= (   P[i+1,j+1,k+1] -    P[i,j+1,k+1]) / Δ.x
            R.x[i,j+1,k+1] += (τ.xy[i,j+1,k+1] - τ.xy[i,j,k+1]) / Δ.y
            R.x[i,j+1,k+1] += (τ.xz[i,j+1,k+1] - τ.xz[i,j+1,k]) / Δ.z
            R.x[i,j+1,k+1] *= -1. 
        end
    end
    if i<=size(R.y,1)-2 && j<=size(R.y,2) && k<=size(R.y,3)-2
        if j>1 && j<size(R.y,2) # avoid Dirichlets
            R.y[i+1,j,k+1]  = b.y[i+1,j,k+1]
            R.y[i+1,j,k+1] += (τ.yy[i+1,j+1,k+1] - τ.yy[i+1,j,k+1]) / Δ.y
            R.y[i+1,j,k+1] -= (   P[i+1,j+1,k+1] -    P[i+1,j,k+1]) / Δ.y
            R.y[i+1,j,k+1] += (τ.xy[i+1,j,k+1] - τ.xy[i,j,k+1]) / Δ.x
            R.y[i+1,j,k+1] += (τ.yz[i+1,j,k+1] - τ.yz[i+1,j,k]) / Δ.z
            R.y[i+1,j,k+1] *= -1.
        end
    end
    if i<=size(R.z,1)-2 && j<=size(R.z,2)-2 && k<=size(R.z,3)
        if k>1 && k<size(R.z,3) # avoid Dirichlets
            R.z[i+1,j+1,k]  = (τ.zz[i+1,j+1,k+1] - τ.zz[i+1,j+1,k]) / Δ.z
            R.z[i+1,j+1,k] -= (   P[i+1,j+1,k+1] -    P[i+1,j+1,k]) / Δ.z
            R.z[i+1,j+1,k] += (τ.xz[i+1,j+1,k] - τ.xz[i,j+1,k]) / Δ.x
            R.z[i+1,j+1,k] += (τ.yz[i+1,j+1,k] - τ.yz[i+1,j,k]) / Δ.y
            R.z[i+1,j+1,k] *= -1.
        end
    end
    return nothing
end

@parallel_indices (i,j,k) function ComputeResiduals_SchurComplement_v3!( R, b, τ, P, Δ )
    if i<=size(R.x,1) && j<=size(R.x,2)-2 && k<=size(R.x,3)-2
        if i>1 && i<size(R.x,1) # avoid Dirichlets
            R.x[i,j+1,k+1]  = (τ.xx[i+1,j+1,k+1] - τ.xx[i,j+1,k+1]) / Δ.x
            R.x[i,j+1,k+1] -= (   P[i+1,j+1,k+1] -    P[i,j+1,k+1]) / Δ.x
            R.x[i,j+1,k+1] += (τ.xy[i,j+1,k+1] - τ.xy[i,j,k+1]) / Δ.y
            R.x[i,j+1,k+1] += (τ.xz[i,j+1,k+1] - τ.xz[i,j+1,k]) / Δ.z
            R.x[i,j+1,k+1] *= -1. 
        end
    end
    if i<=size(R.y,1)-2 && j<=size(R.y,2) && k<=size(R.y,3)-2
        if j>1 && j<size(R.y,2) # avoid Dirichlets
            R.y[i+1,j,k+1]  = b.y[i,j,k]
            R.y[i+1,j,k+1] += (τ.yy[i+1,j+1,k+1] - τ.yy[i+1,j,k+1]) / Δ.y
            R.y[i+1,j,k+1] -= (   P[i+1,j+1,k+1] -    P[i+1,j,k+1]) / Δ.y
            R.y[i+1,j,k+1] += (τ.xy[i+1,j,k+1] - τ.xy[i,j,k+1]) / Δ.x
            R.y[i+1,j,k+1] += (τ.yz[i+1,j,k+1] - τ.yz[i+1,j,k]) / Δ.z
            R.y[i+1,j,k+1] *= -1.
        end
    end
    if i<=size(R.z,1)-2 && j<=size(R.z,2)-2 && k<=size(R.z,3)
        if k>1 && k<size(R.z,3) # avoid Dirichlets
            R.z[i+1,j+1,k]  = (τ.zz[i+1,j+1,k+1] - τ.zz[i+1,j+1,k]) / Δ.z
            R.z[i+1,j+1,k] -= (   P[i+1,j+1,k+1] -    P[i+1,j+1,k]) / Δ.z
            R.z[i+1,j+1,k] += (τ.xz[i+1,j+1,k] - τ.xz[i,j+1,k]) / Δ.x
            R.z[i+1,j+1,k] += (τ.yz[i+1,j+1,k] - τ.yz[i+1,j,k]) / Δ.y
            R.z[i+1,j+1,k] *= -1.
        end
    end
    return nothing
end

@parallel_indices (i,j,k) function ComputeResiduals_MatVec!( R, τ, Δ )
    if i<=size(R.x,1) && j<=size(R.x,2)-2 && k<=size(R.x,3)-2
        if i>1 && i<size(R.x,1) # avoid Dirichlets
            R.x[i,j+1,k+1]  = (τ.xx[i+1,j+1,k+1] - τ.xx[i,j+1,k+1]) / Δ.x
            R.x[i,j+1,k+1] += (τ.xy[i,j+1,k+1] - τ.xy[i,j,k+1]) / Δ.y
            R.x[i,j+1,k+1] += (τ.xz[i,j+1,k+1] - τ.xz[i,j+1,k]) / Δ.z
        end
    end
    if i<=size(R.y,1)-2 && j<=size(R.y,2) && k<=size(R.y,3)-2
        if j>1 && j<size(R.y,2) # avoid Dirichlets
            R.y[i+1,j,k+1]  = (τ.yy[i+1,j+1,k+1] - τ.yy[i+1,j,k+1]) / Δ.y
            R.y[i+1,j,k+1] += (τ.xy[i+1,j,k+1] - τ.xy[i,j,k+1]) / Δ.x
            R.y[i+1,j,k+1] += (τ.yz[i+1,j,k+1] - τ.yz[i+1,j,k]) / Δ.z
        end
    end
    if i<=size(R.z,1)-2 && j<=size(R.z,2)-2 && k<=size(R.z,3)
        if k>1 && k<size(R.z,3) # avoid Dirichlets
            R.z[i+1,j+1,k]  = (τ.zz[i+1,j+1,k+1] - τ.zz[i+1,j+1,k]) / Δ.z
            R.z[i+1,j+1,k] += (τ.xz[i+1,j+1,k] - τ.xz[i,j+1,k]) / Δ.x
            R.z[i+1,j+1,k] += (τ.yz[i+1,j+1,k] - τ.yz[i+1,j,k]) / Δ.y
        end
    end
    return nothing
end