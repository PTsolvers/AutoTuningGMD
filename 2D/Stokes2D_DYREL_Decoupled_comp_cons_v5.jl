# Achtung: α and β labels are interverted w.r.t to the manuscript  

# Initialisation
using Printf, Statistics, LinearAlgebra, JLD2
import Plots as pt
# Macros
@views    av(A) = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])
@views av_xa(A) =  0.5*(A[1:end-1,:].+A[2:end,:])
@views av_ya(A) =  0.5*(A[:,1:end-1].+A[:,2:end])
@views av4_harm(A) = 1.0./( 0.25.*(1.0./A[1:end-1,1:end-1].+1.0./A[2:end,1:end-1].+1.0./A[1:end-1,2:end].+1.0./A[2:end,2:end]) ) 

# Advective term V·∇X at cell centers, consistent with the discrete identity ∇·(ρV) = ρ∇·V + V·∇ρ
# obtained with arithmetic face averages of ρ: each face contributes 0.5*(X_neighbour - X_center)*V_face/Δ
# Boundary faces contribute nothing (density at boundary faces = density of adjacent cell)
@views function advection!(V∇X, gx, gy, X, Vx, Vy, Δx, Δy)
    gx[2:end-1,:] .= (X[2:end,:] .- X[1:end-1,:])./Δx .* Vx[2:end-1,2:end-1]
    gy[:,2:end-1] .= (X[:,2:end] .- X[:,1:end-1])./Δy .* Vy[2:end-1,2:end-1]
    V∇X           .= 0.5.*(gx[1:end-1,:] .+ gx[2:end,:] .+ gy[:,1:end-1] .+ gy[:,2:end])
end

# can be replaced by AD
function Gershgorin_Stokes2D_SchurComplement(ηc, ηv, γ, Δx, Δy, ncx  ,ncy)
        
    ηN    = ones(ncx-1, ncy)
    ηS    = ones(ncx-1, ncy)
    ηN[:,1:end-1] .= ηv[2:end-1,2:end-1]
    ηS[:,2:end-0] .= ηv[2:end-1,2:end-1]
    ηW    = ηc[1:end-1,:]
    ηE    = ηc[2:end-0,:]
    ebW   = γ[1:end-1,:] 
    ebE   = γ[2:end-0,:] 
    Cxx   = ones(ncx-1, ncy)
    Cxy   = ones(ncx-1, ncy)
    @. Cxx = abs.(ηN ./ Δy .^ 2) + abs.(ηS ./ Δy .^ 2) + abs.(ebE ./ Δx .^ 2 + (4 // 3) * ηE ./ Δx .^ 2) + abs.(ebW ./ Δx .^ 2 + (4 // 3) * ηW ./ Δx .^ 2) + abs.(-(-ηN ./ Δy - ηS ./ Δy) ./ Δy + (ebE ./ Δx + ebW ./ Δx) ./ Δx + ((4 // 3) * ηE ./ Δx + (4 // 3) * ηW ./ Δx) ./ Δx)
    @. Cxy = abs.(ebE ./ (Δx .* Δy) - 2 // 3 * ηE ./ (Δx .* Δy) + ηN ./ (Δx .* Δy)) + abs.(ebE ./ (Δx .* Δy) - 2 // 3 * ηE ./ (Δx .* Δy) + ηS ./ (Δx .* Δy)) + abs.(ebW ./ (Δx .* Δy) + ηN ./ (Δx .* Δy) - 2 // 3 * ηW ./ (Δx .* Δy)) + abs.(ebW ./ (Δx .* Δy) + ηS ./ (Δx .* Δy) - 2 // 3 * ηW ./ (Δx .* Δy))
    
    Dx  = ones(ncx-1, ncy)
    @. Dx .= -(-ηN ./ Δy - ηS ./ Δy) ./ Δy + (ebE ./ Δx + ebW ./ Δx) ./ Δx + ((4 // 3) * ηE ./ Δx + (4 // 3) * ηW ./ Δx) ./ Δx

    ηE    = ones(ncx, ncy-1)
    ηW    = ones(ncx, ncy-1)
    ηE[1:end-1,:] .= ηv[2:end-1,2:end-1]
    ηW[2:end-0,:] .= ηv[2:end-1,2:end-1]
    ηS    = ηc[:,1:end-1]
    ηN    = ηc[:,2:end-0]
    ebS  = γ[:,1:end-1] 
    ebN  = γ[:,2:end-0] 
    Cyy  = ones(ncx, ncy-1)
    Cyx  = ones(ncx, ncy-1)
    @. Cyy = abs.(ηE ./ Δx .^ 2) + abs.(ηW ./ Δx .^ 2) + abs.(ebN ./ Δy .^ 2 + (4 // 3) * ηN ./ Δy .^ 2) + abs.(ebS ./ Δy .^ 2 + (4 // 3) * ηS ./ Δy .^ 2) + abs.((ebN ./ Δy + ebS ./ Δy) ./ Δy + ((4 // 3) * ηN ./ Δy + (4 // 3) * ηS ./ Δy) ./ Δy - (-ηE ./ Δx - ηW ./ Δx) ./ Δx)
    @. Cyx = abs.(ebN ./ (Δx .* Δy) + ηE ./ (Δx .* Δy) - 2 // 3 * ηN ./ (Δx .* Δy)) + abs.(ebN ./ (Δx .* Δy) - 2 // 3 * ηN ./ (Δx .* Δy) + ηW ./ (Δx .* Δy)) + abs.(ebS ./ (Δx .* Δy) + ηE ./ (Δx .* Δy) - 2 // 3 * ηS ./ (Δx .* Δy)) + abs.(ebS ./ (Δx .* Δy) - 2 // 3 * ηS ./ (Δx .* Δy) + ηW ./ (Δx .* Δy))

    Dy  = ones(ncx, ncy-1)
    @. Dy .= (ebN ./ Δy + ebS ./ Δy) ./ Δy + ((4 // 3) * ηN ./ Δy + (4 // 3) * ηS ./ Δy) ./ Δy - (-ηE ./ Δx - ηW ./ Δx) ./ Δx

    λmaxVx = 1.0./Dx .* (Cxx .+ Cxy)
    λmaxVy = 1.0./Dy .* (Cyx .+ Cyy)

    return Dx, Dy, λmaxVx, λmaxVy
end

# 2D Stokes routine
@views function Stokes2D(n; form=1, geometry=:layer, verbose=false)

    # Continuity equation:
    # form 1 - non-conservative: 1/K*DP/Dt     = -∇·V
    # form 2 - non-conservative: D(ln(ρ))/Dt   = -∇·V
    # form 3 - conservative:     ∂ρ/∂t         = -∇·(ρV)
    # with ρ = ρref*exp(P/K) and D/Dt = ∂/∂t + V·∇ (Eulerian grid -> advection term needed in forms 1 & 2)

    # Physics
    Lx, Ly   = 1.0, 1.0     # domain size
    η0       = 1.0          # matrix viscosity
    εbg      =-1.0          # background strain-rate (compression along x)
    ρref     = 1.0
    K0       = 1e4          # bulk modulus (uniform)
    comp     = true            
    if geometry == :layer   # Duretz et al. (2018), Fig. A1b / Moulas et al. (2018)
        ηi   = 1e1          # layer viscosity
        h    = 0.3*Ly       # layer thickness (layer parallel to compression direction)
    elseif geometry == :circle
        ηi   = 1e2          # inclusion viscosity
        radi = 0.2          # inclusion radius
    end
    tM       = ηi/K0        # Maxwell time of the layer/inclusion
    Δt       = tM/10        # physical time step
    nt       = 120          # number of time steps -> t_end = 12 tM

    one_iter = false
      
    # Numerics
    ncx, ncy = n*30, n*30   # numerical grid resolution (multiple of 10: layer interfaces coincide with cell faces)
    ϵ        = 1e-8         # tolerance
    iterMax  = 1e5          # max number of iters
    nout     = 100          # check frequency
    c_fact   = 0.5          # damping factor
    dτ_local = true         # helps a little bit sometimes, sometimes not! 
    γfact    = 60           # penalty: multiplier to the arithmetic mean of η
    rel_drop = 1e-3         # relative drop of velocity residual per PH iteration
    # Preprocessing
    Δx, Δy  = Lx/ncx, Ly/ncy
    # Array initialisation
    ρ       =  ones(ncx  ,ncy  )
    ρx      =  ones(ncx+1,ncy  )
    ρy      =  ones(ncx  ,ncy+1)
    ρ0      =  ones(ncx  ,ncy  )
    Pt      = zeros(ncx  ,ncy  )
    Pt0     = zeros(ncx  ,ncy  )
    ∇V      = zeros(ncx  ,ncy  )
    ∇ρV     = zeros(ncx  ,ncy  )
    V∇X     = zeros(ncx  ,ncy  )
    gx      = zeros(ncx+1,ncy  )
    gy      = zeros(ncx  ,ncy+1)
    Vx      = zeros(ncx+1,ncy+2) 
    Vy      = zeros(ncx+2,ncy+1)
    dVx     = zeros(ncx-1,ncy  )
    dVy     = zeros(ncx  ,ncy-1)
    Exx     = zeros(ncx  ,ncy  )
    Eyy     = zeros(ncx  ,ncy  )
    Exy     = zeros(ncx+1,ncy+1)
    Txx     = zeros(ncx  ,ncy  )
    Tyy     = zeros(ncx  ,ncy  )
    Txy     = zeros(ncx+1,ncy+1)
    Rx      = zeros(ncx-1,ncy  )
    Ry      = zeros(ncx  ,ncy-1)
    Rp      = zeros(ncx  ,ncy  )
    Rx0     = zeros(ncx-1,ncy  )
    Ry0     = zeros(ncx  ,ncy-1)
    dVxdτ   = zeros(ncx-1,ncy  )
    dVydτ   = zeros(ncx  ,ncy-1)
    βVx     = zeros(ncx-1,ncy  )
    βVy     = zeros(ncx  ,ncy-1)
    cVx     = zeros(ncx-1,ncy  )  # this disappears is dτ is not local
    cVy     = zeros(ncx  ,ncy-1)  # this disappears is dτ is not local
    αVx     = zeros(ncx-1,ncy  )  # this disappears is dτ is not local
    αVy     = zeros(ncx  ,ncy-1)  # this disappears is dτ is not local
    K       = zeros(ncx  ,ncy  )
    ηc      = zeros(ncx  ,ncy  )
    ηv      = zeros(ncx+1,ncy+1)
    ηc_sharp = zeros(ncx  ,ncy  )
    ηv_sharp = zeros(ncx+1,ncy+1)
    # Initialisation
    xce, yce = LinRange(-Lx/2-Δx/2, Lx/2+Δx/2, ncx+2), LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, ncy+2)
    xc, yc   = LinRange(-Lx/2+Δx/2, Lx/2-Δx/2, ncx), LinRange(-Ly/2+Δy/2, Ly/2-Δy/2, ncy)
    xv, yv   = LinRange(-Lx/2, Lx/2, ncx+1), LinRange(-Ly/2, Ly/2, ncy+1)
    if geometry == :layer
        # Sharp layer: interfaces lie on cell faces, so the 1D solution is reproduced without smearing
        ηc_sharp .= η0; ηc_sharp[:, abs.(yc) .< h/2] .= ηi
        ηc       .= ηc_sharp
        ηv[2:end-1,2:end-1] .= av4_harm(ηc_sharp)
    elseif geometry == :circle
        ηv_sharp .= η0; ηv_sharp[xv.^2 .+ (yv').^2 .< radi^2] .= ηi
        ηc_sharp .= η0; ηc_sharp[xc.^2 .+ (yc').^2 .< radi^2] .= ηi
        # Harmonic averaging mimicking PIC interpolation
        ηc    .= av4_harm(ηv_sharp)
        ηv[2:end-1,2:end-1] .= av4_harm(ηc_sharp)
    end
    # Bulk modulus
    K    .= K0

    # Select γ
    γi   = γfact*mean(ηc).*ones(size(ηc))

    # (Pseudo-)compressibility
    γ_eff = zeros(size(K)) 
    if comp
        γ_num = γi.*ones(size(K)) * 1.0
        γ_phy = K.*Δt                         # physical bulk "viscosity" of the backward-Euler step
        γ_eff = (γ_phy.^-1 .+ γ_num.^-1).^-1
    else
        γ_eff .= γi
    end 
    ##########################################
    if one_iter
        # Converges in on PH iter
        γ_eff = K.*Δt
        rel_drop = 1e-13
    end
    ##########################################

    # Optimal pseudo-time steps - can be replaced by AD
    Dx, Dy, λmaxVx, λmaxVy = Gershgorin_Stokes2D_SchurComplement(ηc, ηv, γ_eff, Δx, Δy, ncx ,ncy)
    # Select dτ
    𝐶 = 0.99
    if dτ_local
        dτVx =  2.0./sqrt.(λmaxVx)*𝐶
        dτVy =  2.0./sqrt.(λmaxVy)*𝐶
    else
        dτVx =  2.0./sqrt.(maximum(λmaxVx))*𝐶 
        dτVy =  2.0./sqrt.(maximum(λmaxVy))*𝐶
    end
    βVx .= 2 .* dτVx ./ (2 .+ cVx.*dτVx)
    βVy .= 2 .* dτVy ./ (2 .+ cVy.*dτVy)
    αVx .= (2 .- cVx.*dτVx) ./ (2 .+ cVx.*dτVx)
    αVy .= (2 .- cVy.*dτVy) ./ (2 .+ cVy.*dτVy)
    # Initial condition
    Vx     .=   εbg.*xv .+    0*yce'
    Vy     .=     0*xce .- εbg.*yv'
    Vx[2:end-1,:] .= 0 # ensure non zero initial pressure residual
    Vy[:,2:end-1] .= 0 # ensure non zero initial pressure residual
    Pt            .= 0.0 
    ρ             .= ρref .* exp.(Pt./K)
    # Probes: layer/inclusion centre and matrix (close to the bottom boundary)
    iP, jL, jM = ncx÷2, ncy÷2, 3
    time = 0.0; t_evo = Float64[]; ΔP_evo = Float64[]; Δτ_evo = Float64[]; iter_tot = 0

    # Time loop
    @time for it=1:nt
        time += Δt
        Pt0  .= Pt
        ρ0   .= ρ
        # Iteration loop
        errVx0 = 1.0;  errVy0 = 1.0;  errPt0 = 1.0 
        errVx00= 1.0;  errVy00= 1.0; 
        iter=1; err_PT=2*ϵ; err_glob = 1.0
        for itPH = 1:50
            if (err_glob<ϵ) break end
            err_PT = 1.0
            # Set tolerance of velocity solve proportional to residual
            ϵ_vel = err_glob*rel_drop
            itPT = 0.
            while (err_PT>ϵ_vel && itPT<=iterMax)
                itPT   += 1
                # Pseudo-old dudes 
                Rx0   .= Rx
                Ry0   .= Ry
                # Boundaries
                Vx[:,1] .= Vx[:,2]; Vx[:,end] .= Vx[:,end-1]
                Vy[1,:] .= Vy[2,:]; Vy[end,:] .= Vy[end-1,:]
                # Divergence 
                ∇V    .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .+ (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy
                # density
                ρ     .= ρref .* exp.(Pt./K)
                ρx[2:end-1,:] .= av_xa(ρ)
                ρy[:,2:end-1] .= av_ya(ρ)
                ρx[[1,end],:] .= ρ[[1,end],:]
                ρy[:,[1,end]] .= ρ[:,[1,end]]
                # mass flux
                ∇ρV .=  ( ρx[2:end,:].*Vx[2:end,2:end-1] .-ρx[1:end-1,:].*Vx[1:end-1,2:end-1])./Δx .+ (ρy[:,2:end].*Vy[2:end-1,2:end] .- ρy[:,1:end-1].*Vy[2:end-1,1:end-1])./Δy
                # Deviatoric strain rate
                Exx   .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .- 1.0/3.0.*∇V
                Eyy   .= (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy .- 1.0/3.0.*∇V
                Exy   .= 0.5.*((Vx[:,2:end] .- Vx[:,1:end-1])./Δy .+ (Vy[2:end,:] .- Vy[1:end-1,:])./Δx)
                # Continuity residual
                if form == 1
                    advection!(V∇X, gx, gy, Pt, Vx, Vy, Δx, Δy)
                    Rp    .= .-∇V .- comp*((Pt .- Pt0)./Δt .+ V∇X)./K
                elseif form == 2
                    advection!(V∇X, gx, gy, log.(ρ), Vx, Vy, Δx, Δy)
                    Rp    .= .-∇V .- comp*((log.(ρ) .- log.(ρ0))./Δt .+ V∇X)
                elseif form == 3
                    Rp    .= (.-∇ρV .- comp*(ρ .- ρ0)./Δt) ./ ρ
                end
                # "Deviatoric" stress: includes Powell-Hestenes numerical bulk viscosity
                Txx   .= 2.0.*ηc.*Exx .- γ_eff .* Rp
                Tyy   .= 2.0.*ηc.*Eyy .- γ_eff .* Rp
                Txy   .= 2.0.*ηv.*Exy 
                # Residuals (PH residuals = true residuals once Pt .+= γ_eff.*Rp is applied)
                Rx    .= (.-(Pt[2:end,:] .- Pt[1:end-1,:])./Δx .+ (Txx[2:end,:] .- Txx[1:end-1,:])./Δx .+ (Txy[2:end-1,2:end] .- Txy[2:end-1,1:end-1])./Δy)
                Ry    .= (.-(Pt[:,2:end] .- Pt[:,1:end-1])./Δy .+ (Tyy[:,2:end] .- Tyy[:,1:end-1])./Δy .+ (Txy[2:end,2:end-1] .- Txy[1:end-1,2:end-1])./Δx)
                # Damping-pong + preconditioning
                dVxdτ .= αVx.*dVxdτ .+ Rx ./ Dx
                dVydτ .= αVy.*dVydτ .+ Ry ./ Dy
                # PT updates
                Vx[2:end-1,2:end-1] .+= dVxdτ.*βVx.*dτVx 
                Vy[2:end-1,2:end-1] .+= dVydτ.*βVy.*dτVy 
                # Residual check
                if mod(iter, nout)==0 || iter==1
                    # Error norms
                    errVx = norm(Rx); errVy = norm(Ry); errPt = norm(Rp)
                    # For relative error metrics
                    if (itPH==1 && iter==1) 
                        # Reference: true initial residual, i.e. without the PH term -∇(γ_eff*Rp) (evaluated only once per step)
                        errVx0 = norm(Rx .+ (γ_eff[2:end,:].*Rp[2:end,:] .- γ_eff[1:end-1,:].*Rp[1:end-1,:])./Δx)
                        errVy0 = norm(Ry .+ (γ_eff[:,2:end].*Rp[:,2:end] .- γ_eff[:,1:end-1].*Rp[:,1:end-1])./Δy)
                        errPt0 = errPt
                    end
                    if iter==1 errVx00 = errVx; errVy00 = errVy; end
                    # PT error
                    # (absolute norm needed from the 2nd time step on: the initial guess is already good, so errVx00 is tiny)
                    err_PT = maximum([min(errVx/sqrt(length(Rx)), errVx/errVx00), min(errVy/sqrt(length(Ry)), errVy/errVy00)])
                    # Global error
                    # Momentum: the PH residual equals the true residual once Pt .+= γ_eff.*Rp is applied
                    eVx = min(errVx/sqrt(length(Rx)), errVx/errVx0)
                    eVy = min(errVy/sqrt(length(Ry)), errVy/errVy0)
                    ePt = min(errPt/sqrt(length(Rp)), errPt/errPt0)
                    err_glob = maximum([eVx, eVy, ePt])
                    verbose && itPT && @printf("---> itPH = %02d iter = %06d iter/nx = %03d, err PT = %1.3e, err glob = %1.3e norm[Rx=%1.3e, Ry=%1.3e, Rp=%1.3e]\n", itPH, iter, iter/ncx, err_PT, err_glob, eVx, eVy, ePt)
                    dVx .= dVxdτ.*βVx.*dτVx
                    dVy .= dVydτ.*βVy.*dτVy
                    # Compute auto-PT parameters
                    λminV  = abs(  sum(dVx.*(Rx .- Rx0)) + sum(dVy.*(Ry .- Ry0))  ) / (sum(dVx.*Dx.*dVx) .+ sum(dVy.*Dy.*dVy))
                    cVx .= 2*sqrt.(λminV)*c_fact*0.5
                    cVy .= 2*sqrt.(λminV)*c_fact*0.5
                    βVx .= 2 .* dτVx ./ (2 .+ cVx.*dτVx)
                    βVy .= 2 .* dτVy ./ (2 .+ cVy.*dτVy)
                    αVx .= (2 .- cVx.*dτVx) ./ (2 .+ cVx.*dτVx)
                    αVy .= (2 .- cVy.*dτVy) ./ (2 .+ cVy.*dτVy)
                end
                iter += 1 
            end
            Pt .+= γ_eff.*Rp
        end
        iter_tot += iter
        # Final state of the step: density and (true) deviatoric stress
        ρ     .= ρref .* exp.(Pt./K)
        Txx   .= 2.0.*ηc.*Exx
        push!(t_evo, time); push!(ΔP_evo, Pt[iP,jL] - Pt[iP,jM]); push!(Δτ_evo, Txx[iP,jL] - Txx[iP,jM])
        verbose && @printf("it = %03d, t/tM = %1.2e, iter/nx = %d, ΔP = %1.4e, Δτxx = %1.4e\n", it, time/tM, iter/ncx, ΔP_evo[end], Δτ_evo[end])
    end
    @printf("form %d (%s): mean iter/nx per step = %.1f\n", form, geometry, iter_tot/nt/ncx)

    return (; t_evo, ΔP_evo, Δτ_evo, tM, Pt=copy(Pt), ∇V=copy(∇V), xc, yc, η0, ηi, εbg, K0, Ly, h = geometry==:layer ? h : NaN)
end

# Run the 3 formulations of the continuity equation and compare
geometry = :layer    # :layer (Duretz et al., 2018, Fig. A1b) or :circle
n        = 2
res      = [Stokes2D(n; form=f, geometry=geometry) for f in 1:3]

# Layer/matrix far-field total stress difference
r   = res[1]
Δσ  = -2*(r.ηi - r.η0)*r.εbg
for f in 2:3
    @printf("form 1 vs %d: max|ΔP diff|/Δσ = %1.2e, max|Δτ diff|/Δσ = %1.2e, rel. diff final P = %1.2e\n", f,
        maximum(abs.(res[f].ΔP_evo .- r.ΔP_evo))/Δσ, maximum(abs.(res[f].Δτ_evo .- r.Δτ_evo))/Δσ, norm(res[f].Pt .- r.Pt)/norm(r.Pt))
end

# Plotting
mk  = (:circle, :rect, :utriangle)
p1  = pt.plot(title="$(geometry)", ylabel="Δτxx/Δσff")
p2  = pt.plot(xlabel="t/tMaxwell", ylabel="ΔP/Δσff")
if geometry == :layer
    t   = LinRange(0, r.t_evo[end], 200)
    # Infinite matrix (Duretz et al., 2018, eq. A8-A9)
    e∞  = exp.(-3/4 .* t./r.tM)
    # Finite matrix with imposed top/bottom velocities: layer compaction is compensated by matrix dilation
    a   = r.h/(r.Ly - r.h)
    eBC = exp.(-3/4 .* r.K0*(1+a)/(r.ηi + a*r.η0) .* t)
    p1  = pt.plot!(p1, t./r.tM, -(1 .- 0.5.*e∞),  c=:black, ls=:dash, label="analytical (infinite matrix)")
    p1  = pt.plot!(p1, t./r.tM, -(1 .- 0.5.*eBC), c=:black,           label="analytical (kinematic BC)")
    p2  = pt.plot!(p2, t./r.tM,    1 .- e∞,        c=:black, ls=:dash, label="analytical (infinite matrix)")
    p2  = pt.plot!(p2, t./r.tM,    1 .- eBC,       c=:black,           label="analytical (kinematic BC)")
end
for f in 1:3
    pt.scatter!(p1, res[f].t_evo./r.tM, res[f].Δτ_evo./Δσ, m=mk[f], ms=3, mc=:white, msc=f, label="form $(f)")
    pt.scatter!(p2, res[f].t_evo./r.tM, res[f].ΔP_evo./Δσ, m=mk[f], ms=3, mc=:white, msc=f, label="form $(f)")
end
p3 = pt.heatmap(r.xc, r.yc, res[3].Pt', aspect_ratio=1, c=:inferno, title="Pt - form 3", xlims=(-0.5,0.5))
p4 = pt.heatmap(r.xc, r.yc, (res[3].Pt .- r.Pt)', aspect_ratio=1, c=:coolwarm, title="Pt form 3 - form 1", xlims=(-0.5,0.5))
display(pt.plot(p1, p3, p2, p4, layout=(2,2), size=(900,700)))
