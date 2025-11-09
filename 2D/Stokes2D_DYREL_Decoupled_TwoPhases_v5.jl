# Initialisation
using Printf, Statistics, LinearAlgebra, JLD2
import Plots as pt

# Macros
@views    av(A) = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])
@views av_xa(A) =  0.5*(A[1:end-1,:].+A[2:end,:])
@views av_ya(A) =  0.5*(A[:,1:end-1].+A[:,2:end])
@views av4_harm(A) = 1.0./( 0.25.*(1.0./A[1:end-1,1:end-1].+1.0./A[2:end,1:end-1].+1.0./A[1:end-1,2:end].+1.0./A[2:end,2:end]) ) 

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
    
    DVx  = ones(ncx-1, ncy)
    @. DVx .= -(-ηN ./ Δy - ηS ./ Δy) ./ Δy + (ebE ./ Δx + ebW ./ Δx) ./ Δx + ((4 // 3) * ηE ./ Δx + (4 // 3) * ηW ./ Δx) ./ Δx

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

    DVy  = ones(ncx, ncy-1)
    @. DVy .= (ebN ./ Δy + ebS ./ Δy) ./ Δy + ((4 // 3) * ηN ./ Δy + (4 // 3) * ηS ./ Δy) ./ Δy - (-ηE ./ Δx - ηW ./ Δx) ./ Δx

    λmaxVx = 1.0./DVx .* (Cxx .+ Cxy)
    λmaxVy = 1.0./DVy .* (Cyx .+ Cyy)

    return DVx, DVy, λmaxVx, λmaxVy
end

# 2D Stokes routine
@views function Stokes2D(n)
    # Physics
    Lx, Ly   = 1.0, 1.0     # domain size
    radi     = 0.2          # inclusion radius
    η0       = 1.0          # viscous viscosity
    ηi       = 1e2          # min/max inclusion viscosity
    εbg      = 1.0          # background strain-rate
    Ωη       = 2.0     
    ϕi       = 1e-3   
    k_ηf0    = 1e-3  
    # Numerics
    ncx, ncy = n*31, n*31   # numerical grid resolution
    ϵ        = 1e-6         # tolerance
    iterMax  = 1e4          # max number of iters
    nout     = 100          # check frequency
    c_factV  = 0.99           # damping factor
    c_factPf = 0.99
    dτ_local = false        # helps a little bit sometimes, sometimes not! 
    γfact    = 5           # penalty: multiplier to the arithmetic mean of η
    rel_drop = 1e-3        # relative drop of velocity residual per PH iteration
    # Preprocessing
    Δx, Δy   = Lx/ncx, Ly/ncy
    # Array initialisation
    ϕ        = ϕi.*ones(ncy  ,ncy  )
    ∇qD      = zeros(ncy  ,ncy  )
    qDx      = zeros(ncy+1,ncy  )
    qDy      = zeros(ncy  ,ncy+1)
    Pt       = zeros(ncx  ,ncy  )
    Pf       = zeros(ncy+2,ncy+2)
    ∇V       = zeros(ncx  ,ncy  )
    Vx       = zeros(ncx+1,ncy+2) 
    Vy       = zeros(ncx+2,ncy+1)
    Exx      = zeros(ncx  ,ncy  )
    Eyy      = zeros(ncx  ,ncy  )
    Exy      = zeros(ncx+1,ncy+1)
    Txx      = zeros(ncx  ,ncy  )
    Tyy      = zeros(ncx  ,ncy  )
    Txy      = zeros(ncx+1,ncy+1)
    RVx      = zeros(ncx-1,ncy  )
    RVy      = zeros(ncx  ,ncy-1)
    RPt      = zeros(ncx  ,ncy  )
    RPf      = zeros(ncx  ,ncy  )
    RVx0     = zeros(ncx-1,ncy  )
    RVy0     = zeros(ncx  ,ncy-1)
    RPf0     = zeros(ncx  ,ncy  )
    dVxdτ    = zeros(ncx-1,ncy  )
    dVydτ    = zeros(ncx  ,ncy-1)
    dPfdτ    = zeros(ncx  ,ncy  )
    βVx      = zeros(ncx-1,ncy  )  # this disappears is dτ is not local
    βVy      = zeros(ncx  ,ncy-1)  # this disappears is dτ is not local
    βPf      = zeros(ncx  ,ncy  )  # this disappears is dτ is not local
    cVx      = zeros(ncx-1,ncy  )  # this disappears is dτ is not local
    cVy      = zeros(ncx  ,ncy-1)  # this disappears is dτ is not local
    cPf      = zeros(ncx  ,ncy  )  # this disappears is dτ is not local
    αVx      = zeros(ncx-1,ncy  )  # this disappears is dτ is not local
    αVy      = zeros(ncx  ,ncy-1)  # this disappears is dτ is not local
    αPf      = zeros(ncx  ,ncy  )  # this disappears is dτ is not local
    dVx      = zeros(ncx-1,ncy  )  # Could be computed on the fly
    dVy      = zeros(ncx  ,ncy-1)  # Could be computed on the fly
    dPf      = zeros(ncx  ,ncy  )  # Could be computed on the fly
    ηb       = zeros(ncx  ,ncy  )
    ηc       = zeros(ncx  ,ncy  )
    ηv       = zeros(ncx+1,ncy+1)
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
    # (Pseudo-)compressibility
    γ_eff = zeros(size(ηb)) 
    γ_num = γi.*ones(size(ηb)) * 1
    γ_phy = ηb.*(1 .- ϕ)
    γ_eff = ((γ_phy.*γ_num)./(γ_phy.+γ_num))
    # Optimal pseudo-time steps - can be replaced by AD
    DVx, DVy, λmaxVx, λmaxVy = Gershgorin_Stokes2D_SchurComplement(ηc, ηv, γ_eff, Δx, Δy, ncx ,ncy)
    DPf    =  1 ./ηb./(1 .- ϕ) .+ 2*k_ηf0*(1/Δx^2 + 1/Δy^2) 
    λmaxPf = (1 ./ηb./(1 .- ϕ) .+ 4*k_ηf0*(1/Δx^2 + 1/Δy^2)) ./ DPf
    # Select dτ
    if dτ_local
        dτVx = 2.0./sqrt.(λmaxVx)*0.99
        dτVy = 2.0./sqrt.(λmaxVy)*0.99
        dτPf = 2.0./sqrt.(λmaxPf)*0.99 
    else
        dτVx = 2.0./sqrt.(maximum(λmaxVx))*0.99 
        dτVy = 2.0./sqrt.(maximum(λmaxVy))*0.99
        dτPf = 2.0./sqrt.(maximum(λmaxPf))*0.99 
    end
    βVx   .= 2 .* dτVx ./ (2 .+ cVx.*dτVx)
    βVy   .= 2 .* dτVy ./ (2 .+ cVy.*dτVy)
    βPf   .= 2 .* dτPf ./ (2 .+ cPf.*dτPf)
    αVx   .= (2 .- cVx.*dτVx) ./ (2 .+ cVx.*dτVx)
    αVy   .= (2 .- cVy.*dτVy) ./ (2 .+ cVy.*dτVy)
    αPf   .= (2 .- cPf.*dτPf) ./ (2 .+ cPf.*dτPf)
    # Initial condition
    Vx     .=   εbg.*xv .+    0*yce'
    Vy     .=     0*xce .- εbg.*yv'
    Vx[2:end-1,:]       .= 0   # ensure non zero initial pressure residual
    Vy[:,2:end-1]       .= 0   # ensure non zero initial pressure residual
    Pf[2:end-1,2:end-1] .= 1e-3
    # Iteration loop
    errVx0 = 1.0;  errVy0 = 1.0;  errPf0 = 1.0;   errPt0 = 1.0 
    errVx00= 1.0;  errVy00= 1.0;  errPf00= 1.0 
    iter=1; err=2*ϵ; err_evo_V=[]; err_evo_Pt=[]; err_evo_Pf=[]; err_evo_it=[]
    @time for itPH = 1:50
        # Boundaries
        Vx[:,1] .= Vx[:,2]; Vx[:,end] .= Vx[:,end-1]
        Vy[1,:] .= Vy[2,:]; Vy[end,:] .= Vy[end-1,:]
        Pf[1,:] .= Pf[2,:]; Pf[end,:] .= Pf[end-1,:] 
        Pf[:,1] .= Pf[:,2]; Pf[:,end] .= Pf[:,end-1]
        # Darcy flux divergence
        qDx    .= -k_ηf0 .* diff(Pf[:,2:end-1], dims=1)/Δx
        qDy    .= -k_ηf0 .* diff(Pf[2:end-1,:], dims=2)/Δy
        ∇qD    .= diff(qDx, dims=1)/Δx .+ diff(qDy, dims=2)/Δy
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
        RVx    .= (.-(Pt[2:end,:] .- Pt[1:end-1,:])./Δx .+ (Txx[2:end,:] .- Txx[1:end-1,:])./Δx .+ (Txy[2:end-1,2:end] .- Txy[2:end-1,1:end-1])./Δy)
        RVy    .= (.-(Pt[:,2:end] .- Pt[:,1:end-1])./Δy .+ (Tyy[:,2:end] .- Tyy[:,1:end-1])./Δy .+ (Txy[2:end,2:end-1] .- Txy[1:end-1,2:end-1])./Δx)
        RPt    .= (.-∇V  .- (Pt.-Pf[2:end-1,2:end-1])./ηb./(1.0.-ϕ))  
        RPf    .= (.-∇qD .+ (Pt.-Pf[2:end-1,2:end-1])./ηb./(1.0.-ϕ))
        # Residual check
        errVx = norm(RVx); errVy = norm(RVy); errPt = norm(RPt); errPf = norm(RPf)
        if itPH==1 errVx0=errVx; errVy0=errVy; errPt0=errPt; errPf0=errPf; end

        eVx = min(errVx/sqrt(length(Vx)), errVx/errVx0)
        eVy = min(errVy/sqrt(length(Vy)), errVy/errVy0)
        ePt = min(errPt/sqrt(length(Pt)), errPt/errPt0)
        ePf = min(errPf/sqrt(length(Pf)), errPf/errPf0)
        err = maximum([eVx, eVy, ePt, ePf]) 

        err = maximum([errVx/errVx0, errVy/errVy0, errPt/errPt0, errPf/errPf0]) #, errVx/sqrt(length(errVx)), errVy/sqrt(length(errVy)), errPt/sqrt(length(errPt)), errPf/sqrt(length(errPt))
        # @printf("itPH = %02d iter = %06d iter/nx = %03d, err = %1.3e norm[RVx=%1.3e, RVy=%1.3e, RPt=%1.3e, RPf=%1.3e] \n", itPH, iter, iter/ncx, err, eVx, eVy, ePt, ePf)
        @printf("itPH = %02d iter = %06d iter/nx = %03d, err = %1.3e norm[RVx=%1.3e, RVy=%1.3e, RPt=%1.3e, RPf=%1.3e] \n", itPH, iter, iter/ncx, err, errVx/errVx0, errVy/errVy0, errPt/errPt0, errPf/errPf0)

        if (err<ϵ) break end
        # Set tolerance of velocity solve proportional to residual
        ϵ_vel = err*rel_drop
        # ϵ_vel = 1e-3
        itPT = 0.
        while (err>ϵ_vel && itPT<=iterMax)
            itPT     += 1
            itg      = iter
            # Pseudo-old dudes 
            RVx0    .= RVx
            RVy0    .= RVy
            RPf0    .= RPf
            # Boundaries
            Vx[:,1] .= Vx[:,2]; Vx[:,end] .= Vx[:,end-1]
            Vy[1,:] .= Vy[2,:]; Vy[end,:] .= Vy[end-1,:]
            Pf[1,:] .= Pf[2,:]; Pf[end,:] .= Pf[end-1,:] 
            Pf[:,1] .= Pf[:,2]; Pf[:,end] .= Pf[:,end-1]
            # Darcy flux divergence
            qDx     .= -k_ηf0 .* diff(Pf[:,2:end-1], dims=1)/Δx
            qDy     .= -k_ηf0 .* diff(Pf[2:end-1,:], dims=2)/Δy
            ∇qD     .= diff(qDx, dims=1)/Δx .+ diff(qDy, dims=2)/Δy
            # Divergence 
            ∇V      .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .+ (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy
            # Deviatoric strain rate
            Exx     .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .- 1.0/3.0.*∇V
            Eyy     .= (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy .- 1.0/3.0.*∇V
            Exy     .= 0.5.*((Vx[:,2:end] .- Vx[:,1:end-1])./Δy .+ (Vy[2:end,:] .- Vy[1:end-1,:])./Δx)
            # "Deviatoric" stress
            RPt     .= .-∇V  .- (Pt.-Pf[2:end-1,2:end-1])./ηb./(1.0.-ϕ)
            Txx     .= 2.0.*ηc.*Exx .- γ_eff .* RPt  
            Tyy     .= 2.0.*ηc.*Eyy .- γ_eff .* RPt  
            Txy     .= 2.0.*ηv.*Exy 
            # Residuals
            RVx     .= (1.0./DVx).*(.-(Pt[2:end,:] .- Pt[1:end-1,:])./Δx .+ (Txx[2:end,:] .- Txx[1:end-1,:])./Δx .+ (Txy[2:end-1,2:end] .- Txy[2:end-1,1:end-1])./Δy)
            RVy     .= (1.0./DVy).*(.-(Pt[:,2:end] .- Pt[:,1:end-1])./Δy .+ (Tyy[:,2:end] .- Tyy[:,1:end-1])./Δy .+ (Txy[2:end,2:end-1] .- Txy[1:end-1,2:end-1])./Δx)
            RPf     .= (1.0./DPf).*(.-∇qD .+ (Pt.-Pf[2:end-1,2:end-1])./ηb./(1.0.-ϕ))
            # Damping-pongs
            dVxdτ   .= αVx.*dVxdτ .+ RVx
            dVydτ   .= αVy.*dVydτ .+ RVy
            dPfdτ   .= αPf.*dPfdτ .+ RPf
            # PT updates
            Vx[2:end-1,2:end-1] .+= dVxdτ.*βVx.*dτVx 
            Vy[2:end-1,2:end-1] .+= dVydτ.*βVy.*dτVy 
            Pf[2:end-1,2:end-1] .+= dPfdτ.*βPf.*dτPf 
            # Residual check
            if mod(iter, nout)==0 || iter==1
                errVx = norm(DVx.*RVx); errVy = norm(DVy.*RVy); errPf = norm(DPf.*RPf)
                if iter==1 errVx00=errVx; errVy00=errVy; errPf00=errPf; end
                err = maximum([errVx./errVx00, errVy./errVy00, errPf./errPf00])
                @show err
                # err = maximum([errVx, errVy, errPf])

                # errVx = norm(DVx.*RVx); errVy = norm(DVy.*RVy); errPf = norm(DPf.*RPf)
                # if itPT==1 errVx00=errVx; errVy00=errVy; errPf00=errPf; end
                # eVx = min(errVx/sqrt(length(Vx)), errVx/errVx00)
                # eVy = min(errVy/sqrt(length(Vy)), errVy/errVy00)
                # ePf = min(errPf/sqrt(length(Pf)), errPf/errPf00)
                # err = maximum([eVx, eVy, ePt, ePf])

                push!(err_evo_V, errVx/errVx00); push!(err_evo_Pt, errPt/errPt0); push!(err_evo_Pf, errPf/errPf0); push!(err_evo_it, itg)
                dVx .= dVxdτ.*βVx.*dτVx
                dVy .= dVydτ.*βVy.*dτVy
                dPf .= dPfdτ.*βPf.*dτPf 
                # @printf("it = %d, iter = %d, err = %1.3e norm[RVx=%1.3e, RVy=%1.3e] \n", it, iter, err, norm_Rx, norm_Ry)
                # λminV  = abs.((sum(dVx.*(RVx .- RVx0))) + abs.((sum(dVy.*(RVy .- RVy0))) )/ ( sum(dVx.*dVx)) + sum(dVy.*dVy) ) 
                λminV  = abs( sum(dVx.*(RVx .- RVx0)) + sum(dVy.*(RVy .- RVy0))  ) / (sum(dVx.*dVx) .+ sum(dVy.*dVy))
                λminPf = abs( sum(dPf.*(RPf .- RPf0))) / sum(dPf.*dPf)
                cVx .= 2*sqrt.(λminV )*c_factV
                cVy .= 2*sqrt.(λminV )*c_factV
                cPf .= 2*sqrt.(λminPf)*c_factPf
                βVx .= 2 .* dτVx ./ (2 .+ cVx.*dτVx)
                βVy .= 2 .* dτVy ./ (2 .+ cVy.*dτVy)
                βPf .= 2 .* dτPf ./ (2 .+ cPf.*dτPf)
                αVx .= (2 .- cVx.*dτVx) ./ (2 .+ cVx.*dτVx)
                αVy .= (2 .- cVy.*dτVy) ./ (2 .+ cVy.*dτVy)
                αPf .= (2 .- cPf.*dτPf) ./ (2 .+ cPf.*dτPf)
            end
            iter += 1 
        end
        Pt .+= γ_eff.*RPt
    end

    # Plotting
    p1 = pt.heatmap(xc, yc, Pf[2:end-1,2:end-1]' , aspect_ratio=1, c=:inferno, title="Pf", xlims=(-Lx/2,Lx/2), xlabel="x",ylabel="y")
    p2 = pt.heatmap(xc, yc, Pt' , aspect_ratio=1, c=:inferno, title="Pt", xlims=(-Lx/2,Lx/2), xlabel="x",ylabel="y")
    p3 = pt.plot(title="Convergence", xlabel="DR iterations / nx",ylabel="errors") 
    p3 = pt.plot!(err_evo_it/ncx, log10.(err_evo_V ), label="V" )
    p3 = pt.plot!(err_evo_it/ncx, log10.(err_evo_Pt), label="Pt")
    p3 = pt.plot!(err_evo_it/ncx, log10.(err_evo_Pf), label="Pf")
    p3 = pt.plot!(err_evo_it/ncx, log10.(ϵ.*ones(size(err_evo_it))), label="tol")  
    p4 = pt.heatmap(xc, yc, log10.(ηc)' , aspect_ratio=1, c=:inferno, title="ηc", xlims=(-Lx/2,Lx/2), xlabel="x",ylabel="y")
    display(pt.plot(p1, p2, p3, p4))

    @show iter/ncx
    n   = length(ηc)
    @show η_h = 1.0 / sum(1.0/n ./ηc)
    @show η_g = exp( sum( 1.0/n*log.(ηc)))
    @show η_a = mean(ηc)

    return
end

n = 8
Stokes2D(n)
