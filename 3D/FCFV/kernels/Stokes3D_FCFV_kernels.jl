@parallel_indices (i,j,k) function interp8!(η, ηv)
    if i<size(η,1) && j<size(η,2) && k<size(η,3)
        # η[i,j,k] = (1/8*(ηv[i,j,k] + ηv[i+1,j,k] + ηv[i,j+1,k] + ηv[i+1,j+1,k] + ηv[i,j,k+1] + ηv[i+1,j,k+1] + ηv[i,j+1,k+1] + ηv[i+1,j+1,k+1]))
        η[i,j,k] = 1/(1/8*(1/ηv[i,j,k] + 1/ηv[i+1,j,k] + 1/ηv[i,j+1,k] + 1/ηv[i+1,j+1,k] + 1/ηv[i,j,k+1] + 1/ηv[i+1,j,k+1] + 1/ηv[i,j+1,k+1] + 1/ηv[i+1,j+1,k+1]))
    end
    return nothing
end

@parallel_indices (I) function copy_R2!(R_it, R)
    if I ≤ length(R_it.x.data)
        R_it.x.data[I] = R.x.data[I]
    end
    if I ≤ length(R_it.y.data)
        R_it.y.data[I] = R.y.data[I]
    end
    if I ≤ length(R_it.z.data)
        R_it.z.data[I] = R.z.data[I]
    end
    return nothing
end

@parallel_indices (I...) function fill_tensor_array!(A, T)

    local_tensor = @cell T[I...]
    for i in 1:3, j in 1:3
        @inbounds A[i,j,I...] = Float32(local_tensor[i,j])
    end

    nothing
end

@parallel_indices (i,j,k) function Set_Dirichlets!( V, σ, η, ηv, ph, verts, cents )

    xc = (0.333, 0.153, 0.079, -0.217, 0.328, -0.337, -0.152, -0.039, -0.463, 0.280, -0.209, 0.302, -0.159, -0.356, -0.047, -0.289, -0.482, -0.145, 0.440, -0.471, 0.091, -0.402, -0.198, -0.114, -0.332, 0.337, 0.326, 0.148, -0.144, -0.394, 0.293, -0.479, 0.478, -0.495, -0.064, -0.474, -0.227, 0.323, -0.293, -0.364, 0.298, 0.462, -0.084, -0.381, -0.123, 0.477, -0.430, -0.134, 0.353, 0.218,)
    yc = (0.123, -0.427, -0.367, -0.020, -0.128, 0.244, -0.347, 0.359, 0.289, 0.260, 0.470, -0.241, -0.379, -0.048, -0.326, -0.273, 0.327, -0.190, 0.226, 0.265, -0.407, 0.069, -0.122, 0.403, -0.051, -0.037, 0.296, 0.138, -0.041, -0.197, -0.456, -0.123, -0.103, 0.189, 0.386, 0.403, 0.025, -0.293, -0.331, 0.014, -0.073, -0.020, 0.479, 0.448, 0.103, 0.380, 0.403, -0.081, 0.131, 0.146,)
    zc = (0.436, 0.431, 0.318, 0.319, 0.302, 0.123, -0.493, 0.077, -0.317, -0.373, 0.245, 0.099, 0.434, 0.309, -0.434, 0.458, 0.168, 0.389, -0.178, 0.311, 0.242, -0.425, -0.371, 0.414, 0.374, 0.339, -0.209, 0.047, 0.090, 0.060, -0.114, -0.069, -0.098, -0.309, 0.034, -0.387, 0.456, 0.446, 0.225, 0.270, 0.390, -0.047, -0.000, 0.218, -0.014, 0.386, -0.313, -0.222, -0.090, -0.143,)
    rc = (0.134, 0.149, 0.053, 0.030, 0.103, 0.114, 0.045, 0.112, 0.110, 0.131, 0.110, 0.080, 0.009, 0.129, 0.113, 0.083, 0.033, 0.111, 0.054, 0.056, 0.013, 0.137, 0.035, 0.091, 0.014, 0.133, 0.104, 0.086, 0.094, 0.109, 0.026, 0.033, 0.046, 0.086, 0.098, 0.030, 0.048, 0.008, 0.072, 0.096, 0.079, 0.036, 0.133, 0.008, 0.039, 0.082, 0.027, 0.018, 0.095, 0.002,)
    phase = (
        2, 3, 3, 3, 2, 2, 3, 2, 2, 3,
        3, 2, 3, 3, 3, 2, 2, 3, 2, 2,
        3, 2, 3, 3, 3, 2, 2, 3, 2, 2,
        3, 2, 3, 3, 3, 2, 2, 3, 2, 2,
        2, 3, 3, 3, 2, 2, 3, 2, 2, 3
    )

    eta = (1e0, 1e-2, 1e2)

    all_dofs = false
    # if BC_type2 == 1 #:SimpleShear
    #     params = (mm=1e0, mc=1e2, rc=0.2, gr=1., er=0.) # Simple shear
    # elseif BC_type2 == 2#:PureShear
        params = (mm=1e0, mc=1e2, rc=0.2, gr=0., er=1.) # Pure shear
    # end

    if i<=size(V.x, 1) && j<=size(V.x, 2) && k<=size(V.x, 3) 
        if ph.x[i,j,k]==1 || ph.x[i,j,k]>=2 || all_dofs
            X    = @SVector([verts.x[i]; cents.y[j]; cents.z[k]]) 
            T_Cell = eltype(V.x)
            V.x[i,j,k] = T_Cell( 1. .* @SVector[verts.x[i]; 0.; 0.] )
        end
    end
    if i<=size(V.y, 1) && j<=size(V.y, 2) && k<=size(V.y, 3) 
        if ph.y[i,j,k]==1 || ph.y[i,j,k]>=2 || all_dofs
            X = @SVector([cents.x[i]; verts.y[j]; cents.z[k]])
            T_Cell = eltype(V.y)
            V.y[i,j,k] = T_Cell( -1.0 .* @SVector[0; verts.y[j]; 0.]  )
        end
    end
    if i<=size(V.z, 1) && j<=size(V.z, 2) && k<=size(V.z, 3) 
        if ph.z[i,j,k]==1 || ph.z[i,j,k]>=2 || all_dofs
            X = @SVector([cents.x[i]; cents.y[j]; verts.z[k]])
            T_Cell = eltype(V.z)
            V.z[i,j,k] = T_Cell( -0.0 .* @SVector[0.; 0.; verts.z[k]]  )
        end
    end

    if i<=size(V.e, 1) && j<=size(V.e, 2) && k<=size(V.e, 3) 
        # Source term
        η[i,j,k] = 1.0
        x, y, z = cents.x[i], cents.y[j], cents.z[k]
        for ii in eachindex(xc)
            if ( (x-xc[ii])^2 + (y-yc[ii])^2 + (z-zc[ii])^2 ) < (rc[ii])^2
                η[i,j,k] = eta[phase[ii]]   
            end
        end
    end

    if i<=size(ηv, 1) && j<=size(ηv, 2) && k<=size(ηv, 3) 
        # Source term
        ηv[i,j,k] = 1.0
        x, y, z = verts.x[i], verts.y[j], verts.z[k]
        for ii in eachindex(xc)
            if ( (x-xc[ii])^2 + (y-yc[ii])^2 + (z-zc[ii])^2 ) < (rc[ii])^2
                ηv[i,j,k] = eta[phase[ii]]
            end
        end
    end

    return nothing
end

function V_τ_elem_local!( V̂, b, α, Ω_ijk, τe, N_el, Γ_el, η, SC, γ )  

        Ve_ijk = b ./ α
        ε̇_ijk  = @SMatrix zeros(3,3)

        for ifac=1:6 # need to extent to 6 for 3D
            # reduce few memreads
            Γi        = Γ_el[ifac]
            @views Ni = SVector{3, Float64}(N_el[ifac,:])
            NiT       = Ni'
            V̂i        = V̂[ifac,:]
            V̂iT       = V̂i'
            # computations
            Ve_ijk     = Ve_ijk .+ (Γi * τe / α) .* V̂i
            ε̇_ijk      = ε̇_ijk  .- (Γi / Ω_ijk * 0.5) * (Ni * V̂iT .+ V̂i * NiT)
        end

        ∇V_ijk    = tr(ε̇_ijk)
        comp      = γ * ∇V_ijk * SC
        τ_ijk     = (2 * η) .* (ε̇_ijk .- 1/3*@SMatrix [
            ∇V_ijk 0e0    0e0
            0e0    ∇V_ijk 0e0
            0e0    0e0    ∇V_ijk
        ]) .+ @SMatrix [
            comp 0e0  0e0
            0e0  comp 0e0
            0e0  0e0  comp
        ]
    return τ_ijk, Ve_ijk, ε̇_ijk, ∇V_ijk
end

@inline Stokes(V, V̂, Vbc, P, τ, n, Γ, τe, Xi, σ, a) =  -a .* Γ .* ((n'*τ)' .+ P*n .+ τe.*(V .- V̂) - Xi*σ*n ) .+ (1.0 .- a).*(V̂ .- (1.0 .- a).*Vbc)

function FaceResidual( V̂, Vdir, P, b, α, Ω, τe, N, Γ, η, SC, γ, X_Neu, σBC, a, i_face )
    τ, Ve, _, _   = V_τ_elem_local!( V̂, b, α, Ω, τe, N, Γ, η, SC, γ ) 
    R       = Stokes(Ve, V̂[i_face,:], Vdir[i_face,:], P, τ, N[i_face,:], Γ[i_face], τe, X_Neu, σBC, a) 
    return R, τ, Ve 
end

@inline function FaceResidualDerivative2!( D, G, face, V̂, Vdir, P, b, α, Ω, τ, N, Γ, η, SC, γ, X_Neu, σBC, a )
    
    eps = 1e-6

    dRdV = @MMatrix zeros(3,3)
    grad = @SVector zeros(3)

    for iface=1:6

        for idim=1:3
                        
            Vref      = V̂[iface,idim]
            pert      = Vref*eps + 1e-13

            V̂[iface,idim] = Vref - pert
            R_m,_,_ = FaceResidual( V̂, Vdir, P, b, α, Ω, τ, N, Γ, η, SC, γ, X_Neu, σBC, a, face )

            V̂[iface,idim] = Vref + pert
            R_p,_,_ = FaceResidual( V̂, Vdir, P, b, α, Ω, τ, N, Γ, η, SC, γ, X_Neu, σBC, a, face )

            V̂[iface,idim] = Vref

            grad = @. (R_p - R_m)/(2*pert) 
            
            dRdV[:,idim] = grad

        end       

        if iface == face
            D .= diag(dRdV)
        end

        G .+= sum(abs.(dRdV), dims=2)

    end
end

function FaceResidualDerivative2(face, V̂, Vdir, P, b, α, Ω, τ, N, Γ, η, SC, γ, X_Neu, σBC, a )
    
    eps  = 1e-6
    D    = @SVector zeros(3)
    G    = @SVector zeros(3)
    dRdV = @MMatrix zeros(3,3)
    grad = @SVector zeros(3)

    V̂mut = MMatrix(V̂)
    for iface=1:6

        for idim=1:3
                        
            Vref      = V̂[iface,idim]
            pert      = Vref*eps + 1e-13

            V̂mut[iface,idim] = Vref - pert
            R_m,_,_ = FaceResidual( SMatrix(V̂mut), Vdir, P, b, α, Ω, τ, N, Γ, η, SC, γ, X_Neu, σBC, a, face )

            V̂mut[iface,idim] = Vref + pert
            R_p,_,_ = FaceResidual( SMatrix(V̂mut), Vdir, P, b, α, Ω, τ, N, Γ, η, SC, γ, X_Neu, σBC, a, face )

            V̂mut[iface,idim] = Vref

            grad = @. (R_p - R_m)/(2*pert)

            dRdV[:,idim] .= grad
            
        end

        if iface == face
            D = diag(SMatrix(dRdV))
        end

        G = G .+ sum(abs.(SMatrix(dRdV)), dims=2)

    end

    return D, G
end

@parallel_indices (i,j,k) function FCFV_iter_params!( D, G, R, b, η, V, P, τ, σ, ph, c, Γ, Ω, N, SC, γ, PC )

    V̂    = @MMatrix zeros(6,3)
    Vdir = @MMatrix zeros(6,3)

    if i<=size(R.x, 1) && j<=size(R.x, 2) && k<=size(R.x, 3) 
        if ph.x[i,j,k] != 1
            Vx       = @cell V.x[i,j,k] 
            σBC      = @cell σ.x[i,j,k]
            X_Neu, a = 0.0, @SVector[1; 1; 1]
            
            # R_W      = @MVector zeros(3)
            # R_E      = @MVector zeros(3)
            D_W      = @MVector zeros(3)
            G_W      = @MVector zeros(3)
            D_E      = @MVector zeros(3)
            G_E      = @MVector zeros(3)

            ph.x[i,j,k]>=2 && ( X_Neu  = 1.0 )  
            ph.x[i,j,k]==3 && (a  = @SVector[0; 1; 1]) # free slip
            
            if i>1 
                N_el = @cell N[i-1,j,k]
                Γ_el = @cell Γ[i-1,j,k]
                b_el = @cell b[i-1,j,k]
                α_el =    c.α[i-1,j,k]
                Ω_el =      Ω[i-1,j,k]
                τ_el =   c.τe[i-1,j,k]
                η_el =      η[i-1,j,k]
                P_el =      P[i-1,j,k]

                V̂      .= 0e0 #@MMatrix zeros(6,3)
                V̂[1,:] .= @cell(V.x[i-1,j,k])
                V̂[2,:] .= @cell(V.x[i-0,j,k])
                V̂[3,:] .= @cell(V.y[i-1,j+0,k])
                V̂[4,:] .= @cell(V.y[i-1,j+1,k])
                V̂[5,:] .= @cell(V.z[i-1,j,k+0])
                V̂[6,:] .= @cell(V.z[i-1,j,k+1])

                Vdir      .= 0e0 #@MMatrix zeros(6,3)
                Vdir[1,:] .= @cell(V.x[i-1,j,k])
                Vdir[2,:] .= @cell(V.x[i-0,j,k])
                Vdir[3,:] .= @cell(V.y[i-1,j+0,k])
                Vdir[4,:] .= @cell(V.y[i-1,j+1,k])
                Vdir[5,:] .= @cell(V.z[i-1,j,k+0])
                Vdir[6,:] .= @cell(V.z[i-1,j,k+1])

                # R_Wnew, τ_W, Ve_W = FaceResidual( SMatrix(V̂), SMatrix(Vdir), P_el, b_el, α_el, Ω_el, τ_el, N_el, Γ_el, η_el, SC, γ, X_Neu, σBC, a, 2 )
                D_Wnew, G_Wnew = FaceResidualDerivative2(2, SMatrix(V̂), SMatrix(Vdir),  P_el, b_el, α_el, Ω_el, τ_el, N_el, Γ_el, η_el, SC, γ, X_Neu, σBC, a )

                D_W .= D_Wnew
                G_W .= G_Wnew
                # R_W .= R_Wnew
                # Res_x(V̂) = FaceResidual( V̂, P_el, b_el, α_el, Ω_el, τ_el, N_el, Γ_el, η_el, SC, γ, X_Neu, σBC, a, i_face )[1]
                # @show autodiff(Enzyme.Reverse, Res_x, Duplicated(V̂, dV̂))
                # @show Enzyme.autodiff_deferred(Enzyme.Reverse, Const(Res), Active, Duplicated(V̂,dV̂))
            end

            if i<size(R.x, 1) 
                N_el = @cell N[i-0,j,k]
                Γ_el = @cell Γ[i-0,j,k]
                b_el = @cell b[i-0,j,k]
                α_el =     c.α[i-0,j,k]
                Ω_el =       Ω[i-0,j,k]
                τ_el =    c.τe[i-0,j,k]
                η_el =       η[i-0,j,k]
                P_el =       P[i-0,j,k]

                V̂      .= 0e0 #@MMatrix zeros(6,3)
                V̂[1,:] .= @cell(V.x[i-0,j+0,k+0])
                V̂[2,:] .= @cell(V.x[i+1,j+0,k+0])
                V̂[3,:] .= @cell(V.y[i-0,j+0,k+0])
                V̂[4,:] .= @cell(V.y[i-0,j+1,k+0])
                V̂[5,:] .= @cell(V.z[i-0,j+0,k+0])
                V̂[6,:] .= @cell(V.z[i-0,j+0,k+1])

                Vdir  = copy(V̂)
                D_Enew, G_Enew = FaceResidualDerivative2(2, SMatrix(V̂), SMatrix(Vdir),  P_el, b_el, α_el, Ω_el, τ_el, N_el, Γ_el, η_el, SC, γ, X_Neu, σBC, a )
                
                # R_Enew,_,_ = FaceResidual( SMatrix(V̂), SMatrix(Vdir), P_el, b_el, α_el, Ω_el, τ_el, N_el, Γ_el, η_el, SC, γ, X_Neu, σBC, a, 1 )

                D_E .= D_Enew
                G_E .= G_Enew
             end
         
            # setcell!(R.x, SVector(R_W) .+ SVector(R_E), i,j,k)
            PC && setcell!(D.x, (SVector(D_W) .+ SVector(D_E)), i,j,k)
            setcell!(G.x, (SVector(G_W) .+ SVector(G_E)), i,j,k)
        end
    end
    if i<=size(R.y, 1) && j<=size(R.y, 2) && k<=size(R.y, 3) 
        if ph.y[i,j,k] != 1
            Vy   = @cell V.y[i,j,k]
            σBC  = @cell σ.y[i,j,k]
            X_Neu, a = 0.0, @SVector[1; 1; 1]
            # R_S      = @SVector zeros(3)
            # R_N      = @SVector zeros(3)
            D_S      = @MVector zeros(3)
            G_S      = @MVector zeros(3)
            D_N      = @MVector zeros(3)
            G_N      = @MVector zeros(3)
            ph.y[i,j,k]>=2 && ( X_Neu  = 1.0 )  
            ph.y[i,j,k]==3 && ( a = @SVector[1; 0; 1]) # free slip
            if j>1    
                N_el = @cell N[i,j-1,k]
                Γ_el = @cell Γ[i,j-1,k]
                b_el = @cell b[i,j-1,k]
                α_el =     c.α[i,j-1,k]
                Ω_el =       Ω[i,j-1,k]
                τ_el =    c.τe[i,j-1,k]
                η_el =       η[i,j-1,k]
                P_el =       P[i,j-1,k]

                V̂      .= 0e0 #@MMatrix zeros(6,3)
                # V̂       = @MMatrix zeros(6,3)
                V̂[1,:] .= @cell(V.x[i+0,j-1,k+0])
                V̂[2,:] .= @cell(V.x[i+1,j-1,k+0])
                V̂[3,:] .= @cell(V.y[i+0,j-1,k+0])
                V̂[4,:] .= @cell(V.y[i+0,j+0,k+0])
                V̂[5,:] .= @cell(V.z[i+0,j-1,k+0])
                V̂[6,:] .= @cell(V.z[i+0,j-1,k+1])

                Vdir  = copy(V̂)
                # FaceResidualDerivative2!( D_S, G_S, 4, V̂, Vdir,  P_el, b_el, α_el, Ω_el, τ_el, N_el, Γ_el, η_el, SC, γ, X_Neu, σBC, a )
                D_Snew, G_Snew = FaceResidualDerivative2( 4, SMatrix(V̂), SMatrix(Vdir),  P_el, b_el, α_el, Ω_el, τ_el, N_el, Γ_el, η_el, SC, γ, X_Neu, σBC, a )

                D_S .= D_Snew
                G_S .= G_Snew

                # R_S,_,_ = FaceResidual( V̂, Vdir, P_el, b_el, α_el, Ω_el, τ_el, N_el, Γ_el, η_el, SC, γ, X_Neu, σBC, a, 4 )

            end
            if j<size(R.y, 2) 
                N_el = @cell N[i,j-0,k]
                Γ_el = @cell Γ[i,j-0,k]
                b_el = @cell b[i,j-0,k]
                α_el =     c.α[i,j-0,k]
                Ω_el =       Ω[i,j-0,k]
                τ_el =    c.τe[i,j-0,k]
                η_el =       η[i,j-0,k]
                P_el =       P[i,j-0,k]

                V̂      .= 0e0 #@MMatrix zeros(6,3)
                V̂[1,:] .= @cell(V.x[i+0,j+0,k+0])
                V̂[2,:] .= @cell(V.x[i+1,j+0,k+0])
                V̂[3,:] .= @cell(V.y[i+0,j+0,k+0])
                V̂[4,:] .= @cell(V.y[i+0,j+1,k+0])
                V̂[5,:] .= @cell(V.z[i+0,j+0,k+0])
                V̂[6,:] .= @cell(V.z[i+0,j+0,k+1])
                Vdir  = copy(V̂)

                # FaceResidualDerivative2!( D_N, G_N, 3, V̂, Vdir,  P_el, b_el, α_el, Ω_el, τ_el, N_el, Γ_el, η_el, SC, γ, X_Neu, σBC, a )
                D_Nnew, G_Nnew = FaceResidualDerivative2(3, SMatrix(V̂), SMatrix(Vdir),  P_el, b_el, α_el, Ω_el, τ_el, N_el, Γ_el, η_el, SC, γ, X_Neu, σBC, a )
                # R_N,_,_ = FaceResidual( V̂, Vdir, P_el, b_el, α_el, Ω_el, τ_el, N_el, Γ_el, η_el, SC, γ, X_Neu, σBC, a, 3 )
                D_N .= D_Nnew
                G_N .= G_Nnew
            end
            # setcell!(R.y, R_S .+ R_N, i,j,k)
            PC && setcell!(D.y, (SVector(D_S) .+ SVector(D_N)), i,j,k)
            setcell!(G.y, (SVector(G_S) .+ SVector(G_N)), i,j,k)
        end
    end
    if i<=size(R.z, 1) && j<=size(R.z, 2) && k<=size(R.z, 3) 
        if ph.z[i,j,k] != 1
            Vz   = @cell V.z[i,j,k]
            σBC  = @cell σ.z[i,j,k]
            X_Neu, a = 0.0, @SVector[1; 1; 1]
            # R_B      = @SVector zeros(3)
            # R_F      = @SVector zeros(3)
            D_B      = @MVector zeros(3)
            G_B      = @MVector zeros(3)
            D_F      = @MVector zeros(3)
            G_F      = @MVector zeros(3)
            ph.z[i,j,k]>=2 && ( X_Neu  = 1.0 )  
            ph.z[i,j,k]==3 && ( a  = @SVector[1; 1; 0])

            if k>1      
                N_el = @cell N[i,j,k-1]
                Γ_el = @cell Γ[i,j,k-1]
                b_el = @cell b[i,j,k-1]
                α_el =     c.α[i,j,k-1]
                Ω_el =       Ω[i,j,k-1]
                τ_el =    c.τe[i,j,k-1]
                η_el =       η[i,j,k-1]
                P_el =       P[i,j,k-1]

                V̂      .= 0e0 #@MMatrix zeros(6,3)
                V̂[1,:] .= @cell(V.x[i+0,j+0,k-1])
                V̂[2,:] .= @cell(V.x[i+1,j+0,k-1])
                V̂[3,:] .= @cell(V.y[i+0,j+0,k-1])
                V̂[4,:] .= @cell(V.y[i+0,j+1,k-1])
                V̂[5,:] .= @cell(V.z[i+0,j+0,k-1])
                V̂[6,:] .= @cell(V.z[i+0,j+0,k+0])

                Vdir  = copy(V̂)

                D_Bnew, G_Bnew = FaceResidualDerivative2(6, SMatrix(V̂), SMatrix(Vdir),  P_el, b_el, α_el, Ω_el, τ_el, N_el, Γ_el, η_el, SC, γ, X_Neu, σBC, a )
                # # FaceResidualDerivative2!( D_B, G_B, 6, V̂, Vdir,  P_el, b_el, α_el, Ω_el, τ_el, N_el, Γ_el, η_el, SC, γ, X_Neu, σBC, a )
                # # R_B,_,_ = FaceResidual( V̂, Vdir, P_el, b_el, α_el, Ω_el, τ_el, N_el, Γ_el, η_el, SC, γ, X_Neu, σBC, a, 6 )
                D_B .= D_Bnew
                G_B .= G_Bnew
            end

            if k<size(R.z, 3)
                N_el = @cell N[i,j,k-0]
                Γ_el = @cell Γ[i,j,k-0]
                b_el = @cell b[i,j,k-0]
                α_el =     c.α[i,j,k-0]
                Ω_el =       Ω[i,j,k-0]
                τ_el =    c.τe[i,j,k-0]
                η_el =       η[i,j,k-0]
                P_el =       P[i,j,k-0]

                V̂      .= 0e0 # @MMatrix zeros(6,3)
                V̂[1,:] .= @cell(V.x[i+0,j+0,k+0])
                V̂[2,:] .= @cell(V.x[i+1,j+0,k+0])
                V̂[3,:] .= @cell(V.y[i+0,j+0,k+0])
                V̂[4,:] .= @cell(V.y[i+0,j+1,k+0])
                V̂[5,:] .= @cell(V.z[i+0,j+0,k+0])
                V̂[6,:] .= @cell(V.z[i+0,j+0,k+1])

                Vdir  = copy(V̂)

                D_Fnew, G_Fnew = FaceResidualDerivative2( 5, SMatrix(V̂), SMatrix(Vdir),  P_el, b_el, α_el, Ω_el, τ_el, N_el, Γ_el, η_el, SC, γ, X_Neu, σBC, a )
                # FaceResidualDerivative2!( D_F, G_F, 5, V̂, Vdir,  P_el, b_el, α_el, Ω_el, τ_el, N_el, Γ_el, η_el, SC, γ, X_Neu, σBC, a )
                # R_F,_,_ = FaceResidual( V̂, Vdir, P_el, b_el, α_el, Ω_el, τ_el, N_el, Γ_el, η_el, SC, γ, X_Neu, σBC, a, 5 )
                D_F .= D_Fnew
                G_F .= G_Fnew
            end

            # setcell!(R.z, R_B .+ R_F, i,j,k)
            PC && setcell!(D.z, (SVector(D_B) .+ SVector(D_F)), i,j,k)
            setcell!(G.z, (SVector(G_B) .+ SVector(G_F)), i,j,k)
        end
    end

    return nothing
end

@parallel_indices (I) function copy_R2!(R_it, R)
    if I ≤ length(R_it.x.data)
        R_it.x.data[I] = R.x.data[I]
    end
    if I ≤ length(R_it.y.data)
        R_it.y.data[I] = R.y.data[I]
    end
    if I ≤ length(R_it.z.data)
        R_it.z.data[I] = R.z.data[I]
    end
    return nothing
end

@parallel_indices (I) function UpdateRatesFields4!(V, D, ∂V∂τe, R, a1, a2, h)

    if I ≤ length(V.x.data)
        ∂V∂τe.x.data[I] = a1.x * ∂V∂τe.x.data[I] - a2.x * R.x.data[I] / D.x.data[I]
        V.x.data[I]    += h.x * ∂V∂τe.x.data[I]
    end
    if I ≤ length(V.y.data)
        ∂V∂τe.y.data[I] = a1.y * ∂V∂τe.y.data[I] - a2.y * R.y.data[I] / D.y.data[I]
        V.y.data[I]    += h.y * ∂V∂τe.y.data[I]
    end
    if I ≤ length(V.z.data)
        ∂V∂τe.z.data[I] = a1.z * ∂V∂τe.z.data[I] - a2.z * R.z.data[I] / D.z.data[I]
        V.z.data[I]    += h.z * ∂V∂τe.z.data[I]
    end
    
    return nothing
end

@parallel_indices (i,j,k) function UpdatePressure_SchurComplement!( P, bP, ∇V, ε̇, γ )
    if i<=size(P,1) && j<=size(P,2) && k<=size(P,3) 
        P[i,j,k] += γ.*(bP[i,j,k] + ∇V[i,j,k])
    end
    return nothing
end

@parallel_indices (i,j,k) function FCFV_coeffs!( c, b, T, ph, Γ, Ω, N )
    if i<=size(c.α, 1) && j<=size(c.α, 2) && k<=size(c.α, 3) 

        # Face types 
        c.α[i,j,k] = 0.0
        c.β[i,j,k] = b[i,j,k]*Ω[i,j,k]

        Γ_el = @cell Γ[i,j,k]

        for ifac=1:6
            c.α[i,j,k] += Γ_el[ifac]*c.τe[i,j,k]
        end
    end
    return nothing
end

@parallel_indices (i,j,k) function V_τ_elem!( 
    V, ε̇, ∇V, η, τ, b, c, ph, Γ, Ω, N, SC, γ 
)  
    if i<=size(V.e, 1) && j<=size(V.e, 2) && k<=size(V.e, 3) 

        V̂ = SMatrix{6,3, Float64}([@cell(V.x[i,j,k])'; @cell(V.x[i+1,j,k])'; @cell(V.y[i,j,k])'; @cell(V.y[i,j+1,k])'; @cell(V.z[i,j,k])'; @cell(V.z[i,j,k+1])'] )        
        τ_el, V_el, ε̇_el, div_el   = V_τ_elem_local!( V̂, b[i,j,k], c.α[i,j,k], Ω[i,j,k], c.τe[i,j,k], (@cell N[i,j,k]), (@cell Γ[i,j,k]), η[i,j,k], SC, γ ) 
        
        ∇V[i,j,k]          = div_el
        @cell τ[i, j, k]   = τ_el
        @cell V.e[i, j, k] = V_el
        @cell ε̇[i, j, k]   = ε̇_el
    end
    return nothing
end

@parallel_indices (i,j,k) function ResidualStokes!( R, V, σ, P, τ, ph, c, Γ, Ω, N )

    if i<=size(R.x, 1) && j<=size(R.x, 2) && k<=size(R.x, 3) 
        if ph.x[i,j,k]!=1
            Vx       = @cell V.x[i,j,k]
            Vdir     = @cell V.x[i,j,k]  
            σBC      = @cell σ.x[i,j,k]
            X_Neu, a = 0.0, @SVector[1; 1; 1]
            R_W      = @SVector zeros(3)
            R_E      = @SVector zeros(3)
            ph.x[i,j,k]>=2 && ( X_Neu  = 1.0 )  
            ph.x[i,j,k]==3 && (a  = @SVector[0; 1; 1]) # free slip
            if i>1 
                N_W = @cell N[i-1,j,k]
                Γ_W = @cell Γ[i-1,j,k]
                R_W = Stokes(@cell(V.e[i-1,j,k]), Vx, Vdir, P[i-1,j,k], @cell(τ[i-1,j,k]), N_W[2,:], Γ_W[2], c.τe[i-1,j,k], X_Neu, σBC, a) 
            end
            if i<size(R.x, 1) 
                N_E = @cell N[i-0,j,k]
                Γ_E = @cell Γ[i-0,j,k]
                R_E = Stokes(@cell(V.e[i-0,j,k]), Vx, Vdir, P[i-0,j,k], @cell(τ[i-0,j,k]), N_E[1,:], Γ_E[1], c.τe[i-0,j,k], X_Neu, σBC, a) 
            end
            setcell!(R.x, R_W .+ R_E, i,j,k)
        end
    end
    if i<=size(R.y, 1) && j<=size(R.y, 2) && k<=size(R.y, 3) 
        if ph.y[i,j,k]!=1
            Vy   = @cell V.y[i,j,k]
            Vdir = @cell V.y[i,j,k]
            σBC  = @cell σ.y[i,j,k]
            X_Neu, a = 0.0, @SVector[1; 1; 1]
            R_S      = @SVector zeros(3)
            R_N      = @SVector zeros(3)
            ph.y[i,j,k]>=2 && ( X_Neu  = 1.0 )  
            ph.y[i,j,k]==3 && ( a = @SVector[1; 0; 1]) # free slip
            if j>1    
                N_S = @cell N[i,j-1,k]
                Γ_S = @cell Γ[i,j-1,k]
                R_S = Stokes(@cell(V.e[i,j-1,k]), Vy, Vdir, P[i,j-1,k], @cell(τ[i, j-1, k]), N_S[4,:], Γ_S[4], c.τe[i,j-1,k], X_Neu, σBC, a) 
            end
            if j<size(R.y, 2) 
                N_N = @cell N[i,j-0,k]
                Γ_N = @cell Γ[i,j-0,k]
                R_N = Stokes(@cell(V.e[i,j-0,k]), Vy, Vdir, P[i,j-0,k], @cell(τ[i, j-0, k]), N_N[3,:], Γ_N[3], c.τe[i,j-0,k], X_Neu, σBC, a) 
            end
            setcell!(R.y, R_S .+ R_N, i,j,k)
        end
    end
    if i<=size(R.z, 1) && j<=size(R.z, 2) && k<=size(R.z, 3) 
        if ph.z[i,j,k]!=1
            Vz   = @cell V.z[i,j,k]
            Vdir = @cell V.z[i,j,k]
            σBC  = @cell σ.z[i,j,k]
            X_Neu, a, Vbc = 0.0, @SVector[1; 1; 1], @SVector[0.; 0.; 0.]
            R_B      = @SVector zeros(3)
            R_F      = @SVector zeros(3)
            ph.z[i,j,k]>=2 && ( X_Neu  = 1.0 )  
            ph.z[i,j,k]==3 && ( a  = @SVector[1; 1; 0]; Vbc = (1 .- a) .* Vz)
            if k>1      
                N_B = @cell N[i,j,k-1]
                Γ_B = @cell Γ[i,j,k-1]      
                R_B = Stokes(@cell(V.e[i,j,k-1]), Vz, Vdir, P[i,j,k-1], @cell(τ[i, j, k-1]), N_B[6,:], Γ_B[6], c.τe[i,j,k-1], X_Neu, σBC, a) 
            end
            if k<size(R.z, 3)
                N_F = @cell N[i,j,k-0]
                Γ_F = @cell Γ[i,j,k-0] 
                R_F = Stokes(@cell(V.e[i,j,k-0]), Vz, Vdir, P[i,j,k-0], @cell(τ[i, j, k-0]), N_F[5,:], Γ_F[5], c.τe[i,j,k-0], X_Neu, σBC, a) 
            end
            setcell!(R.z, R_B .+ R_F, i,j,k)
        end
    end
    return nothing
end

@parallel_indices (i,j,k) function Set_Γ_N!( Γ, N, Δ )

    if i<=size(Γ, 1) && j<=size(Γ, 2) && k<=size(Γ, 3)
        Nmat = @SMatrix([-1 0 0; 1 0 0; 0 -1 0; 0 1 0; 0 0 -1; 0 0 1])
        Γvec = @SVector([Δ.y*Δ.z, Δ.y*Δ.z, Δ.x*Δ.z, Δ.x*Δ.z, Δ.x*Δ.y, Δ.x*Δ.y])
        setcell!(Γ, Γvec, i,j,k)
        setcell!(N, Nmat, i,j,k)
    end
    return nothing
end