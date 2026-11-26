# Notation follows the GMD article (https://gmd.copernicus.org/articles/19/5343/2026/), Eqs. (16)-(17):
#   ∂V∂τ ← β ∂V∂τ + P⁻¹ R      with β = (2 - cΔτ)/(2 + cΔτ)
#   V    ← V + α ∂V∂τ          with α = 2Δτ²/(2 + cΔτ)
# (same as 1D/Poisson1D_NonLinear.jl; α, β, c and Δτ are scalars, P = diag(PCx, PCy))
# Experimental version: adds the automatic tolerance of the inexact velocity solves (ph_inner = :balanced)
# and AR-DR (ardr = true), both ported from Stokes2D_DYREL_Decoupled_comp_v4.jl

# Initialisation
using Printf, Statistics, LinearAlgebra
import Plots as pt
# Macros
@views av_xa(A) =  0.5*(A[1:end-1,:].+A[2:end,:])
@views av_ya(A) =  0.5*(A[:,1:end-1].+A[:,2:end])
@views av4_harm(A) = 1.0./( 0.25.*(1.0./A[1:end-1,1:end-1].+1.0./A[2:end,1:end-1].+1.0./A[1:end-1,2:end].+1.0./A[2:end,2:end]) ) 
# ‖∇(γ Rp)‖: momentum-residual jump caused by the next PH update (ph_inner = :balanced)
@views penalty_jump_norm(γRp, Δx, Δy) = sqrt(sum(abs2, diff(γRp, dims=1))/Δx^2 + sum(abs2, diff(γRp, dims=2))/Δy^2)
ardr_coefficients(k, Δτ) = (β = (k-1)/(k+2); (Δτ^2*(1+β)/2, β))

# Advection term (can be replace by marker-in-cell or others)
@views function advection!(V∇X, gx, gy, X, Vx, Vy, Δx, Δy)
    gx[2:end-1,:] .= (X[2:end,:] .- X[1:end-1,:])./Δx .* Vx[2:end-1,2:end-1]
    gy[:,2:end-1] .= (X[:,2:end] .- X[:,1:end-1])./Δy .* Vy[2:end-1,2:end-1]
    V∇X           .= 0.5.*(gx[1:end-1,:] .+ gx[2:end,:] .+ gy[:,1:end-1] .+ gy[:,2:end])
end

# Momentum and continuity residuals
@views function Residual_Stokes2D!(Rx, Ry, Vx, Vy, A, p)
    (; Rp, ∇V, ∇ρV, V∇X, gx, gy, ρ, ρx, ρy, ρ0, Pt, Pt0, Exx, Eyy, Exy, Txx, Tyy, Txy, ηc, ηv, K, γ_eff) = A
    (; ρref, Δt, Δx, Δy, form) = p
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
        Rp    .= .-∇V .- ((Pt .- Pt0)./Δt .+ V∇X)./K
    elseif form == 2
        advection!(V∇X, gx, gy, log.(ρ), Vx, Vy, Δx, Δy)
        Rp    .= .-∇V .- ((log.(ρ) .- log.(ρ0))./Δt .+ V∇X)
    elseif form == 3
        Rp    .= (.-∇ρV .- (ρ .- ρ0)./Δt) ./ ρ
    end
    # "Deviatoric" stress: includes Powell-Hestenes numerical bulk viscosity
    Txx   .= 2.0.*ηc.*Exx .- γ_eff .* Rp
    Tyy   .= 2.0.*ηc.*Eyy .- γ_eff .* Rp
    Txy   .= 2.0.*ηv.*Exy 
    # Residuals (PH residuals = true residuals once Pt .+= γ_eff.*Rp is applied)
    Rx    .= (.-(Pt[2:end,:] .- Pt[1:end-1,:])./Δx .+ (Txx[2:end,:] .- Txx[1:end-1,:])./Δx .+ (Txy[2:end-1,2:end] .- Txy[2:end-1,1:end-1])./Δy)
    Ry    .= (.-(Pt[:,2:end] .- Pt[:,1:end-1])./Δy .+ (Tyy[:,2:end] .- Tyy[:,1:end-1])./Δy .+ (Txy[2:end,2:end-1] .- Txy[1:end-1,2:end-1])./Δx)
    return nothing
end

# Matrix free probing of diag PC and Gershgorin
@views function preconditioner_and_λmax(res!, Vx, Vy, args...; S=3)
    Vx, Vy  = copy(Vx), copy(Vy)            # work copies: the solution arrays are left untouched
    Vxi, Vyi = Vx[2:end-1,2:end-1], Vy[2:end-1,2:end-1]     # unknowns
    Rx0, Ry0 = zeros(size(Vxi)), zeros(size(Vyi))
    Rxp, Ryp = zeros(size(Vxi)), zeros(size(Vyi))
    Dx,  Dy  = zeros(size(Vxi)), zeros(size(Vyi))
    Ax,  Ay  = zeros(size(Vxi)), zeros(size(Vyi))
    Vxi .= 0.0; Vyi .= 0.0
    res!(Rx0, Ry0, Vx, Vy, args...)
    for (Vi, D) in ((Vxi, Dx), (Vyi, Dy)), cx in 1:S, cy in 1:S
        Vxi .= 0.0; Vyi .= 0.0
        Vi[cx:S:end, cy:S:end] .= 1.0
        res!(Rxp, Ryp, Vx, Vy, args...)
        Rxp .-= Rx0; Ryp .-= Ry0
        D[cx:S:end, cy:S:end] .= (Vi === Vxi ? Rxp : Ryp)[cx:S:end, cy:S:end]
        Ax  .+= abs.(Rxp)
        Ay  .+= abs.(Ryp)
    end
    PCx, PCy = .-Dx, .-Dy
    return PCx, PCy, Ax./PCx, Ay./PCy
end

# 2D Stokes routine
# ph_inner = :balanced : velocity solve stopped once ‖R‖ ≤ σ_inner ‖∇(γ_eff Rp)‖ (falls back to :rel_drop once Rp < ϵ)
#            :rel_drop : velocity solve stopped once its error ≤ rel_drop × global error
# ardr     = true      : AR-DR momentum ramp with adaptive restart (test every nrestart iterations) instead of λmin-based damping
@views function Stokes2D(n; form=1, geometry=:layer, verbose=false, ph_inner=:balanced, σ_inner=0.3, ardr=false, nrestart=10,
                         γfact=60, Δt_fact=0.1, nt=120, nout=100)

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
    if geometry == :layer   # Duretz et al. (2018), Fig. A1b / Moulas et al. (2018)
        ηi   = 1e1          # layer viscosity
        h    = 0.3*Ly       # layer thickness (layer parallel to compression direction)
    elseif geometry == :circle
        ηi   = 1e2          # inclusion viscosity
        radi = 0.2          # inclusion radius
    end
    tM       = ηi/K0        # Maxwell time of the layer/inclusion
    Δt       = Δt_fact*tM   # physical time step (keyword Δt_fact; large Δt_fact -> K Δt ≫ η: nearly incompressible)

    # Numerics
    ncx, ncy = n*30, n*30   # numerical grid resolution (multiple of 10: layer interfaces coincide with cell faces)
    ϵ        = 1e-8         # tolerance
    iterMax  = 100000         # max number of iters
    # nout   (keyword)      # check frequency
    c_fact   = 0.9          # damping factor: c = 2*sqrt(λmin)*c_fact
    𝐶        = 0.99         # safety factor on the pseudo time step: Δτ = 2/sqrt(λmax)*𝐶
    # γfact  (keyword)      # penalty: multiplier to the arithmetic mean of η
    rel_drop = 1e-3         # relative drop of velocity residual per PH iteration
    # Preprocessing
    Δx, Δy  = Lx/ncx, Ly/ncy
    
    ######################## Allocation #######################
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
    ΔVx     = zeros(ncx-1,ncy  )
    ΔVy     = zeros(ncx  ,ncy-1)
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
    ∂Vx∂τ   = zeros(ncx-1,ncy  )
    ∂Vy∂τ   = zeros(ncx  ,ncy-1)
    K       = zeros(ncx  ,ncy  )
    ηc      = zeros(ncx  ,ncy  )
    ηv      = zeros(ncx+1,ncy+1)
    ηc_sharp = zeros(ncx  ,ncy  )
    ηv_sharp = zeros(ncx+1,ncy+1)
    
    ######################## Initialisation #######################
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
        ηc    .= av4_harm(ηv_sharp)
        ηv[2:end-1,2:end-1] .= av4_harm(ηc_sharp)
    end
    # Bulk modulus
    K    .= K0
    ######################## Initial condition #######################
    Vx            .=   εbg.*xv .+    0*yce'
    Vy            .=     0*xce .- εbg.*yv'
    Vx[2:end-1,:] .= 0 # ensure non zero initial pressure residual
    Vy[:,2:end-1] .= 0 # ensure non zero initial pressure residual
    Pt            .= 0.0 
    ρ             .= ρref .* exp.(Pt./K)

    ######################## Probes #######################
    iP, jL, jM = ncx÷2, ncy÷2, 3
    time = 0.0; t_evo = Float64[]; ΔP_evo = Float64[]; Δτ_evo = Float64[]; iter_tot = 0; ph_tot = 0; ph_cap = 0

    ####################### Numerical parameters #######################
    # Powell-Hestenes penalty: harmonic mean of numerical and physical (K Δt, backward Euler) penalties
    γ_num = γfact*mean(ηc)
    γ_phy = K.*Δt
    γ_eff = 1.0./(1.0./γ_phy .+ 1.0/γ_num)
    # Fields and parameters needed by the residual function
    A = (; Rp, ∇V, ∇ρV, V∇X, gx, gy, ρ, ρx, ρy, ρ0, Pt, Pt0, Exx, Eyy, Exy, Txx, Tyy, Txy, ηc, ηv, K, γ_eff)
    p = (; ρref, Δt, Δx, Δy, form)
    # DR parameters: Δτ from λmax (probing, each PH iteration), c from λmin (Rayleigh quotient, during DR)
    Δτ, λmin, c, α, β = 0.0, 0.0, 0.0, 0.0, 0.0

    ######################## Time loop #######################
    @time for it=1:nt
        time += Δt
        Pt0  .= Pt
        ρ0   .= ρ
        # Iteration loop
        errVx0 = 1.0;  errVy0 = 1.0;  errPt0 = 1.0 
        errVx00= 1.0;  errVy00= 1.0;
        iter = 1; err_glob = 1.0; ePt = 1.0
        k_rs = 0            # AR-DR: iterations since last restart
        # Powell-Hestenes
        for itPH = 1:50
            if (err_glob<ϵ) break end
            ph_tot += 1
            (itPH == 50) && (ph_cap += 1)
            err_PT = 1.0
            # Diagonal preconditioner and maximum eigenvalue (preconditioned system), from probing.
            # Re-evaluated after each pressure update: the Jacobian depends on Pt through ρ and the advection terms
            PCx, PCy, λmaxVx, λmaxVy = preconditioner_and_λmax(Residual_Stokes2D!, Vx, Vy, A, p)
            # Pseudo time step is proportional to λmax (e.g., Oakley, 1995)
            Δτ = 2.0/sqrt(max(maximum(λmaxVx), maximum(λmaxVy)))*𝐶
            α  = 2*Δτ^2/(2 + c*Δτ)
            β  = (2 - c*Δτ)/(2 + c*Δτ)
            # Velocity tolerance: proportional to residual (:rel_drop) or balanced against the
            # next pressure update (:balanced, falls back to :rel_drop once Rp < ϵ)
            use_balanced   = ph_inner === :balanced && ePt > ϵ
            ϵ_vel          = use_balanced ? 0.0 : err_glob*rel_drop
            inner_balanced = false
            itPT = 0

            # Dynamic-relaxation
            while (err_PT>ϵ_vel && !inner_balanced && err_glob>ϵ && itPT<=iterMax)
                itPT   += 1
                # Pseudo-old dudes 
                Rx0   .= Rx
                Ry0   .= Ry
                # Residuals
                Residual_Stokes2D!(Rx, Ry, Vx, Vy, A, p)
                # AR-DR: momentum ramp with adaptive restart (restart test only every nrestart iterations)
                if ardr
                    k_rs = (iter > 1 && mod(iter, nrestart) == 0 && sum(Rx.*∂Vx∂τ) + sum(Ry.*∂Vy∂τ) < 0) ? 1 : k_rs + 1
                    α, β = ardr_coefficients(k_rs, Δτ)
                end
                # Damping-pong + preconditioning
                ∂Vx∂τ .= β.*∂Vx∂τ .+ Rx./PCx      # Achtung: preconditioned system
                ∂Vy∂τ .= β.*∂Vy∂τ .+ Ry./PCy
                # PT updates
                Vx[2:end-1,2:end-1] .+= α.*∂Vx∂τ
                Vy[2:end-1,2:end-1] .+= α.*∂Vy∂τ
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
                    # Balanced inexact solve: velocity error small compared to the jump of the next PH update
                    use_balanced && (inner_balanced = sqrt(errVx^2 + errVy^2) <= σ_inner*penalty_jump_norm(γ_eff.*Rp, Δx, Δy))
                    verbose && @printf("---> itPH = %02d iter = %06d iter/nx = %03d, err PT = %1.3e, err glob = %1.3e norm[Rx=%1.3e, Ry=%1.3e, Rp=%1.3e]\n", itPH, iter, iter/ncx, err_PT, err_glob, eVx, eVy, ePt)
                    ardr && (iter += 1; continue)   # AR-DR: no λmin needed
                    # Increment that produced R - R0 (α, β unchanged since then): ΔV = α ∂V∂τ_old = (α/β)(∂V∂τ - P⁻¹R)
                    ΔVx .= (α/β).*(∂Vx∂τ .- Rx./PCx)
                    ΔVy .= (α/β).*(∂Vy∂τ .- Ry./PCy)
                    # Compute auto-PT parameters
                    # Rayleigh quotient (e.g., Joldes et al., 2011) - skipped at iter 1 (ΔV = 0 and R - R0 polluted by the Pt update)
                    (iter > 1) && (λmin = abs(  sum(ΔVx.*(Rx .- Rx0)) + sum(ΔVy.*(Ry .- Ry0))  ) / (sum(ΔVx.*PCx.*ΔVx) + sum(ΔVy.*PCy.*ΔVy)))
                    # Dynamic evaluation of PT iteration parameters
                    c = 2*sqrt(λmin)*c_fact
                    α = 2*Δτ^2/(2 + c*Δτ)
                    β = (2 - c*Δτ)/(2 + c*Δτ)
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
    @printf("form %d (%s, %s, ardr=%s): mean iter/nx per step = %.1f, mean PH iter per step = %.1f, steps hitting the PH cap = %d\n", form, geometry, ph_inner, ardr, iter_tot/nt/ncx, ph_tot/nt, ph_cap)
    @printf("    γfact = %d, final Δτ = %.3e (λmax = %.3f), λmin = %.3e\n", γfact, Δτ, (2𝐶/Δτ)^2, λmin)

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
