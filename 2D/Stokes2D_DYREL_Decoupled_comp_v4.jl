# Achtung: α and β labels are interverted w.r.t to the manuscript  

# Initialisation
using Printf, Statistics, LinearAlgebra, JLD2
import Plots as pt
# Macros
@views    av(A) = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])
@views av_xa(A) =  0.5*(A[1:end-1,:].+A[2:end,:])
@views av_ya(A) =  0.5*(A[:,1:end-1].+A[:,2:end])
@views av4_harm(A) = 1.0./( 0.25.*(1.0./A[1:end-1,1:end-1].+1.0./A[2:end,1:end-1].+1.0./A[1:end-1,2:end].+1.0./A[2:end,2:end]) ) 
# ‖∇(γ Rp)‖: momentum-residual jump caused by the next PH update (ph_inner = :balanced)
@views penalty_jump_norm(γRp, Δx, Δy) = sqrt(sum(abs2, diff(γRp, dims=1))/Δx^2 + sum(abs2, diff(γRp, dims=2))/Δy^2)

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
@views function Stokes2D(n; ph_inner = :balanced, σ_inner = 0.3, nout = 100, ardr = false, dτ_local = true, maxit_nx = Inf, show = true)
    # Physics
    Lx, Ly   = 1.0, 1.0     # domain size
    radi     = 0.2          # inclusion radius
    η0       = 1.0          # viscous viscosity
    ηi       = 1e2          # min/max inclusion viscosity
    εbg      =-1.0          # background strain-rate
    comp     = true            

    one_iter = false
    Ωη       = 2e3     
      
    # Numerics
    ncx, ncy = n*31, n*31   # numerical grid resolution
    ϵ        = 1e-9         # tolerance
    iterMax  = 1e5          # max number of iters
    c_fact   = 0.5          # damping factor
    γfact    = 60           # penalty: multiplier to the arithmetic mean of η
    rel_drop = 1e-3         # relative drop of velocity residual per PH iteration
    # Preprocessing
    Δx, Δy  = Lx/ncx, Ly/ncy
    # Array initialisation
    Pt      = zeros(ncx  ,ncy  )
    ∇V      = zeros(ncx  ,ncy  )
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
    ηb      = zeros(ncx  ,ncy  )
    ηc      = zeros(ncx  ,ncy  )
    ηv      = zeros(ncx+1,ncy+1)
    ηc_sharp = zeros(ncx  ,ncy  )
    ηv_sharp = zeros(ncx+1,ncy+1)
    # Initialisation
    xce, yce = LinRange(-Lx/2-Δx/2, Lx/2+Δx/2, ncx+2), LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, ncy+2)
    xc, yc   = LinRange(-Lx/2+Δx/2, Lx/2-Δx/2, ncx), LinRange(-Ly/2+Δy/2, Ly/2-Δy/2, ncy)
    xv, yv   = LinRange(-Lx/2, Lx/2, ncx+1), LinRange(-Ly/2, Ly/2, ncy+1)
    # Multiple circles with various viscosities
    ηi    = (w=1/ηi, s=ηi) 
    x_inc = [0.0   0.2  -0.3 -0.4  0.0 -0.3 0.4  0.3  0.35 -0.1 ] 
    y_inc = [0.0   0.4   0.4 -0.3 -0.2  0.2 -0.2 -0.4 0.2  -0.4 ]
    r_inc = [radi  0.09  0.05 0.08 0.08  0.1 0.07 0.08 0.07 0.07] 
    η_inc = [ηi.s  ηi.w  ηi.w ηi.s ηi.w ηi.s ηi.w ηi.s ηi.s ηi.w]
    ηv_sharp   .= η0
    for inc in 1:1#eachindex(η_inc)
        ηv_sharp[(xv.-x_inc[inc]).^2 .+ (yv'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= η_inc[inc]
    end
    ηc_sharp   .= η0
    for inc in 1:1#eachindex(η_inc)
        ηc_sharp[(xc.-x_inc[inc]).^2 .+ (yc'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= η_inc[inc]
    end  
    # Harmonic averaging mimicking PIC interpolation
    ηc    .= av4_harm(ηv_sharp)
    ηv[2:end-1,2:end-1] .= av4_harm(ηc_sharp)
    # Bulk viscosity
    ηb    .= ηc.*Ωη

    # Select γ
    γi   = γfact*mean(ηc).*ones(size(ηc))

    # dη = maximum(ηc)/minimum(ηc)
    # @show  γi = 10^0.575*dη^0.3 * 50

    # (Pseudo-)compressibility
    γ_eff = zeros(size(ηb)) 
    if comp
        γ_num = γi.*ones(size(ηb)) * 1.0
        γ_phy = ηb
        # γ_eff = ((γ_phy.*γ_num)./(γ_phy.+γ_num))
        γ_eff = (γ_phy.^-1 .+ γ_num.^-1).^-1
    else
        γ_eff .= γi
        γ_eff .= γ_eff
    end 

    ##########################################
    if one_iter
        # Converges in on PH iter
        γ_phy = ηb
        γ_eff = ηb
        γ_num = ηb
        rel_drop = 1e-13
        ph_inner = :rel_drop
    end
    ##########################################

    # Optimal pseudo-time steps - can be replaced by AD
    Dx, Dy, λmaxVx, λmaxVy = Gershgorin_Stokes2D_SchurComplement(ηc, ηv, γ_eff, Δx, Δy, ncx ,ncy)
    # Select dτ
    if dτ_local
        dτVx =  2.0./sqrt.(λmaxVx)*0.99
        dτVy =  2.0./sqrt.(λmaxVy)*0.99
    else
        dτVx =  2.0./sqrt.(maximum(λmaxVx))*0.99 
        dτVy =  2.0./sqrt.(maximum(λmaxVy))*0.99
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
    # Iteration loop
    errVx0 = 1.0;  errVy0 = 1.0;  errPt0 = 1.0 
    errVx00= 1.0;  errVy00= 1.0; 
    iter=1; err=2*ϵ; err_evo_V=[]; err_evo_P=[]; err_evo_it=[]
    ph_evo_it = Int[]; ph_evo_err = Float64[]   # residual at each PH iteration
    k_rs = 0                                    # AR-DR: iterations since last restart
    @time for itPH = 1:50
        # Boundaries
        Vx[:,1] .= Vx[:,2]; Vx[:,end] .= Vx[:,end-1]
        Vy[1,:] .= Vy[2,:]; Vy[end,:] .= Vy[end-1,:]
        # Divergence
        ∇V    .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .+ (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy
        # Deviatoric strain rate
        Exx   .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .- 1.0/3.0.*∇V
        Eyy   .= (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy .- 1.0/3.0.*∇V
        Exy   .= 0.5.*((Vx[:,2:end] .- Vx[:,1:end-1])./Δy .+ (Vy[2:end,:] .- Vy[1:end-1,:])./Δx)
        # Deviatoric stress
        Txx   .= 2.0.*ηc.*Exx
        Tyy   .= 2.0.*ηc.*Eyy
        Txy   .= 2.0.*ηv.*Exy 
        # Residuals
        Rx    .= (.-(Pt[2:end,:] .- Pt[1:end-1,:])./Δx .+ (Txx[2:end,:] .- Txx[1:end-1,:])./Δx .+ (Txy[2:end-1,2:end] .- Txy[2:end-1,1:end-1])./Δy)
        Ry    .= (.-(Pt[:,2:end] .- Pt[:,1:end-1])./Δy .+ (Tyy[:,2:end] .- Tyy[:,1:end-1])./Δy .+ (Txy[2:end,2:end-1] .- Txy[1:end-1,2:end-1])./Δx)
        Rp    .= .-∇V .- comp*Pt./ηb 
        # Residual check
        errVx = norm(Rx); errVy = norm(Ry); errPt = norm(Rp)
        if itPH==1 errVx0=errVx; errVy0=errVy; errPt0=errPt; end
        err = maximum([errVx/errVx0, errVy/errVy0, errPt/errPt0])
        push!(ph_evo_it, iter); push!(ph_evo_err, err)
        @printf("itPH = %02d iter = %06d iter/nx = %03d, err = %1.3e norm[Rx=%1.3e, Ry=%1.3e, Rp=%1.3e] \n", itPH, iter, iter/ncx, err, errVx/errVx0, errVy/errVy0, errPt/errPt0)
        if (err<ϵ) break end
        (err > 1e3 || !isfinite(err)) && (@warn "diverged at itPH = $itPH"; break)
        iter > maxit_nx*ncx && (@warn "iteration cap reached at itPH = $itPH"; break)
        # Velocity tolerance: proportional to residual (:rel_drop) or balanced against the
        # next pressure update (:balanced, falls back to :rel_drop once Rp < ϵ)
        use_balanced   = ph_inner === :balanced && errPt/errPt0 > ϵ
        ϵ_vel = use_balanced ? 0.0 : err*rel_drop
        inner_balanced = false
        itPT = 0.
        while (err>ϵ_vel && !inner_balanced && itPT<=iterMax && iter <= maxit_nx*ncx)
            itPT   += 1
            itg    = iter
            # Pseudo-old dudes 
            Rx0   .= Rx
            Ry0   .= Ry
            # Boundaries
            Vx[:,1] .= Vx[:,2]; Vx[:,end] .= Vx[:,end-1]
            Vy[1,:] .= Vy[2,:]; Vy[end,:] .= Vy[end-1,:]
            # Divergence 
            ∇V    .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .+ (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy
            # Deviatoric strain rate
            Exx   .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .- 1.0/3.0.*∇V
            Eyy   .= (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy .- 1.0/3.0.*∇V
            Exy   .= 0.5.*((Vx[:,2:end] .- Vx[:,1:end-1])./Δy .+ (Vy[2:end,:] .- Vy[1:end-1,:])./Δx)
            # "Deviatoric" stress
            Rp    .= .-∇V .- comp*Pt./ηb 
            Txx   .= 2.0.*ηc.*Exx .- γ_eff .* Rp  
            Tyy   .= 2.0.*ηc.*Eyy .- γ_eff .* Rp  
            Txy   .= 2.0.*ηv.*Exy 
            # Residuals
            Rx    .= (1.0./Dx).*(.-(Pt[2:end,:] .- Pt[1:end-1,:])./Δx .+ (Txx[2:end,:] .- Txx[1:end-1,:])./Δx .+ (Txy[2:end-1,2:end] .- Txy[2:end-1,1:end-1])./Δy)
            Ry    .= (1.0./Dy).*(.-(Pt[:,2:end] .- Pt[:,1:end-1])./Δy .+ (Tyy[:,2:end] .- Tyy[:,1:end-1])./Δy .+ (Txy[2:end,2:end-1] .- Txy[1:end-1,2:end-1])./Δx)
            # AR-DR: β ramp + adaptive restart (α = momentum, β·dτ = step here)
            if ardr
                # Restart test before the momentum update, otherwise R⋅R/D > 0 biases it
                if mod(iter, nout)==0 && sum((Dx.*Rx).*dVxdτ) + sum((Dy.*Ry).*dVydτ) < 0
                    dVxdτ .= 0.0; dVydτ .= 0.0; k_rs = 0
                end
                k_rs += 1; mom = (k_rs - 1)/(k_rs + 2)
                αVx .= mom; αVy .= mom
                βVx .= dτVx.*(1 + mom)./2; βVy .= dτVy.*(1 + mom)./2
            end
            # Damping-pong
            dVxdτ .= αVx.*dVxdτ .+ Rx
            dVydτ .= αVy.*dVydτ .+ Ry
            # PT updates
            Vx[2:end-1,2:end-1] .+= dVxdτ.*βVx.*dτVx 
            Vy[2:end-1,2:end-1] .+= dVydτ.*βVy.*dτVy 
            # Residual check
            if mod(iter, nout)==0
                errVx = norm(Dx.*Rx); errVy = norm(Dy.*Ry)
                if iter==nout errVx00=errVx; errVy00=errVy; end
                err = maximum([min(errVx, errVx./errVx00), min(errVy, errVy./errVy00)])
                use_balanced && (inner_balanced = sqrt(errVx^2 + errVy^2) <= σ_inner*penalty_jump_norm(γ_eff.*Rp, Δx, Δy))
                push!(err_evo_V, errVx/errVx00); push!(err_evo_P, errPt/errPt0); push!(err_evo_it, itg)
                if ardr
                    iter += 1
                    continue
                end
                dVx .= dVxdτ.*βVx.*dτVx
                dVy .= dVydτ.*βVy.*dτVy
                # @printf("iter = %d, err = %1.3e norm[Rx=%1.3e, Ry=%1.3e] \n", iter, err, errVx, errVy)
                # λminV  = abs.((sum(dVx.*(Rx .- Rx0))) + abs.((sum(dVy.*(Ry .- Ry0))) )/ ( sum(dVx.*dVx)) + sum(dVy.*dVy) ) 
                λminV  = abs(  sum(dVx.*Dx.*(Rx .- Rx0)) + sum(dVy.*Dy.*(Ry .- Ry0))  ) / (sum(dVx.*Dx.*dVx) .+ sum(dVy.*Dy.*dVy))
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

    # Plotting
    p1 = pt.heatmap(xc, yc, ∇V' , aspect_ratio=1, c=:inferno, title="∇V", xlims=(-Lx/2,Lx/2), xlabel="x", ylabel="y")
    p2 = pt.heatmap(xc, yc, Pt' , aspect_ratio=1, c=:inferno, title="Pt", xlims=(-Lx/2,Lx/2), xlabel="x", ylabel="y")
    p3 = pt.plot(title="Convergence", xlabel="DR iterations / nx",ylabel="errors") 
    p3 = pt.plot!(err_evo_it/ncx, log10.(err_evo_V), label="V")
    p3 = pt.plot!(err_evo_it/ncx, log10.(err_evo_P), label="P")
    p4 = pt.heatmap(xc, yc, log10.(ηc)' , aspect_ratio=1, c=:inferno, title="ηc", xlims=(-Lx/2,Lx/2), xlabel="x", ylabel="y")
    show && display(pt.plot(p1, p2, p3, p4))

    @show iter/ncx
    n   = length(ηc)
    @show η_h = 1.0 / sum(1.0/n ./ηc)
    @show η_g = exp( sum( 1.0/n*log.(ηc)))
    @show η_a = mean(ηc)

    return
end

n = 8
Stokes2D(n)
