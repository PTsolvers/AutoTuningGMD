# Initialisation
using Printf, Statistics, LinearAlgebra, JLD2
using ParallelStencil, ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(Threads, Float64, 2)
import Plots as pt

# Macros
@views    av(A) = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])
@views av_xa(A) =  0.5*(A[1:end-1,:].+A[2:end,:])
@views av_ya(A) =  0.5*(A[:,1:end-1].+A[:,2:end])
@views av4_harm(A) = 1.0./( 0.25.*(1.0./A[1:end-1,1:end-1].+1.0./A[2:end,1:end-1].+1.0./A[1:end-1,2:end].+1.0./A[2:end,2:end]) ) 
@views function maxloc!(A2, A)
    A2[2:end-1,2:end-1] .= max.(max.(max.(A[1:end-2,2:end-1], A[3:end,2:end-1]), A[2:end-1,2:end-1]), max.(A[2:end-1,1:end-2], A[2:end-1,3:end]))
    A2[[1,end],:] .= A2[[2,end-1],:]; A2[:,[1,end]] .= A2[:,[2,end-1]]
end

# smoothing from Raess et al 2022
@parallel function smooth!(A2::Data.Array, A::Data.Array, fact::Data.Number)
    @inn(A2) = @inn(A) + 1.0/4.1/fact*(@d2_xi(A) + @d2_yi(A))
    return
end

function Gershgorin_Stokes2D_SC(ηc, ηv, γ, Δx, Δy, ncx  ,ncy)
    ebE, ebW, ebN, ebS = γ, γ, γ, γ
    Δx, Δy = Δx, Δy
    # Gershgorin x
    ηN    = ones(ncx-1, ncy)
    ηS    = ones(ncx-1, ncy)
    ηN[:,1:end-1] .= ηv[2:end-1,2:end-1]
    ηS[:,2:end-0] .= ηv[2:end-1,2:end-1]
    ηW    = ηc[1:end-1,:]
    ηE    = ηc[2:end-0,:]
    Cxx   = ones(ncx-1, ncy)
    Cxy   = ones(ncx-1, ncy)
    @. Cxx = abs.(ηN ./ Δy .^ 2) + abs.(ηS ./ Δy .^ 2) + abs.(ebE ./ Δx .^ 2 + (4 // 3) * ηE ./ Δx .^ 2) + abs.(ebW ./ Δx .^ 2 + (4 // 3) * ηW ./ Δx .^ 2) + abs.(-(-ηN ./ Δy - ηS ./ Δy) ./ Δy + (ebE ./ Δx + ebW ./ Δx) ./ Δx + ((4 // 3) * ηE ./ Δx + (4 // 3) * ηW ./ Δx) ./ Δx)
    @. Cxy = abs.(ebE ./ (Δx .* Δy) - 2 // 3 * ηE ./ (Δx .* Δy) + ηN ./ (Δx .* Δy)) + abs.(ebE ./ (Δx .* Δy) - 2 // 3 * ηE ./ (Δx .* Δy) + ηS ./ (Δx .* Δy)) + abs.(ebW ./ (Δx .* Δy) + ηN ./ (Δx .* Δy) - 2 // 3 * ηW ./ (Δx .* Δy)) + abs.(ebW ./ (Δx .* Δy) + ηS ./ (Δx .* Δy) - 2 // 3 * ηW ./ (Δx .* Δy))
    # Diagonal x
    ηN[:,1:end-1] .= ηv[2:end-1,2:end-1]
    ηS[:,2:end-0] .= ηv[2:end-1,2:end-1]
    ηW    = ηc[1:end-1,:]
    ηE    = ηc[2:end-0,:]
    cVxC  = ones(ncx-1, ncy)
    @. cVxC .= -(-ηN ./ Δy - ηS ./ Δy) ./ Δy + (ebE ./ Δx + ebW ./ Δx) ./ Δx + ((4 // 3) * ηE ./ Δx + (4 // 3) * ηW ./ Δx) ./ Δx
    # Gershgorin y
    ηE    = ones(ncx, ncy-1)
    ηW    = ones(ncx, ncy-1)
    ηE[1:end-1,:] .= ηv[2:end-1,2:end-1]
    ηW[2:end-0,:] .= ηv[2:end-1,2:end-1]
    ηS    = ηc[:,1:end-1]
    ηN    = ηc[:,2:end-0]
    Cyy  = ones(ncx, ncy-1)
    Cyx  = ones(ncx, ncy-1)
    @. Cyy = abs.(ηE ./ Δx .^ 2) + abs.(ηW ./ Δx .^ 2) + abs.(ebN ./ Δy .^ 2 + (4 // 3) * ηN ./ Δy .^ 2) + abs.(ebS ./ Δy .^ 2 + (4 // 3) * ηS ./ Δy .^ 2) + abs.((ebN ./ Δy + ebS ./ Δy) ./ Δy + ((4 // 3) * ηN ./ Δy + (4 // 3) * ηS ./ Δy) ./ Δy - (-ηE ./ Δx - ηW ./ Δx) ./ Δx)
    @. Cyx = abs.(ebN ./ (Δx .* Δy) + ηE ./ (Δx .* Δy) - 2 // 3 * ηN ./ (Δx .* Δy)) + abs.(ebN ./ (Δx .* Δy) - 2 // 3 * ηN ./ (Δx .* Δy) + ηW ./ (Δx .* Δy)) + abs.(ebS ./ (Δx .* Δy) + ηE ./ (Δx .* Δy) - 2 // 3 * ηS ./ (Δx .* Δy)) + abs.(ebS ./ (Δx .* Δy) - 2 // 3 * ηS ./ (Δx .* Δy) + ηW ./ (Δx .* Δy))
    # Diagonal y
    ηE[1:end-1,:] .= ηv[2:end-1,2:end-1]
    ηW[2:end-0,:] .= ηv[2:end-1,2:end-1]
    ηS    = ηc[:,1:end-1]
    ηN    = ηc[:,2:end-0]
    cVyC  = ones(ncx, ncy-1)
    @. cVyC .= (ebN ./ Δy + ebS ./ Δy) ./ Δy + ((4 // 3) * ηN ./ Δy + (4 // 3) * ηS ./ Δy) ./ Δy - (-ηE ./ Δx - ηW ./ Δx) ./ Δx
    λmaxVx = 1.0./cVxC .* (Cxx .+ Cxy)
    λmaxVy = 1.0./cVyC .* (Cyx .+ Cyy)  
    return cVxC, cVyC, λmaxVx, λmaxVy
end

# 2D Stokes routine
@views function Stokes2D(n, solver, noisy)
    # Physics
    Lx, Ly  = 10., 10.       # domain size
    r       = 1.0            # inclusion radius
    ηi      = 1e-3           # inclusion viscosity
    εbg     =-1.0            # background strain-rate
    smooth  = 10             # like in Raess et al., 2022
    # Numerics
    ncx, ncy = 255, 255  # numerical grid resolution
    nt       = 1         # number of time steps
    ϵ        = 1e-6      # tolerance
    iterMax  = 2e5       # max number of iters
    nout     = 100       # check frequency
    c_fact   = 0.5
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
    Txx0    = zeros(ncx  ,ncy  )
    Tyy0    = zeros(ncx  ,ncy  )
    Txy0    = zeros(ncx+1,ncy+1)
    Rx      = zeros(ncx-1,ncy  )
    Ry      = zeros(ncx  ,ncy-1)
    Rp      = zeros(ncx  ,ncy  )
    Rx0     = zeros(ncx-1,ncy  )
    Ry0     = zeros(ncx  ,ncy-1)
    dVxdt   = zeros(ncx-1,ncy  )
    dVydt   = zeros(ncx  ,ncy-1)
    βVx     = zeros(ncx-1,ncy  )
    βVy     = zeros(ncx  ,ncy-1)
    cVx     = zeros(ncx-1,ncy  )
    cVy     = zeros(ncx  ,ncy-1)
    αVx     = zeros(ncx-1,ncy  )
    αVy     = zeros(ncx  ,ncy-1)
    ηc      = zeros(ncx  ,ncy  )
    ηv      = zeros(ncx+1,ncy+1)
    ηc_sharp = zeros(ncx  ,ncy  )
    ηv_sharp = zeros(ncx+1,ncy+1)
    # Initialisation
    xce, yce = LinRange(-Lx/2-Δx/2, Lx/2+Δx/2, ncx+2), LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, ncy+2)
    xc, yc   = LinRange(-Lx/2+Δx/2, Lx/2-Δx/2, ncx), LinRange(-Ly/2+Δy/2, Ly/2-Δy/2, ncy)
    xv, yv   = LinRange(-Lx/2, Lx/2, ncx+1), LinRange(-Ly/2, Ly/2, ncy+1)
    (Xvx,Yvx) = ([x for x=xv,y=yce], [y for x=xv,y=yce])
    (Xvy,Yvy) = ([x for x=xce,y=yv], [y for x=xce,y=yv])
    # Multiple circles with various viscosities
    x_inc = [0.0 ] 
    y_inc = [0.0 ]
    r_inc = [r   ] 
    η_inc = [ηi]
    ηc_sharp   .= 1.0
    for inc in eachindex(η_inc)
        ηc_sharp[(xc.-x_inc[inc]).^2 .+ (yc'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= η_inc[inc]
    end
    ηc     .= ηc_sharp
    for ism=1:smooth
        @parallel smooth!(ηc, ηc_sharp, 1.0)
        ηc_sharp, ηc = ηc, ηc_sharp
    end
    ηv[2:end-1,2:end-1] = av(ηc)
    # Select penalty γ
    n = length(ηc)
    γ = 2.5*mean(ηc)
    # Optimal pseudo-time steps
    cVxC, cVyC, λmaxVx, λmaxVy = Gershgorin_Stokes2D_SC(ηc, ηv, γ, Δx, Δy, ncx ,ncy)
    dtVx =  2.0./sqrt.(λmaxVx)*0.99 
    dtVy =  2.0./sqrt.(λmaxVy)*0.99
    βVx .= 2 .* dtVx ./ (2 .+ cVx.*dtVx)
    βVy .= 2 .* dtVy ./ (2 .+ cVy.*dtVy)
    αVx .= (2 .- cVx.*dtVx) ./ (2 .+ cVx.*dtVx)
    αVy .= (2 .- cVy.*dtVy) ./ (2 .+ cVy.*dtVy)
    @show minimum(dtVx), maximum(dtVx)
    @show minimum(dtVy), maximum(dtVy)
    # Initial condition
    Vx     .=   εbg.*Xvx
    Vy     .= .-εbg.*Yvy
    Vx[2:end-1,:] .= 0 # ensure non zero initial pressure residual
    Vy[:,2:end-1] .= 0 # ensure non zero initial pressure residual
    # Time loop
    t=0.0; evo_t=[]; evo_Txx=[]
    @views for it = 1:nt
        errVx0 = 1.0;  errVy0 = 1.0;  errPt0 = 1.0 
        errVx00= 1.0;  errVy00= 1.0; 
        iter=0; err=2*ϵ; err_evo_V=[]; err_evo_P=[]; err_evo_it=[]
        Txx0.=Txx; Tyy0.=Tyy; Txy0.=Txy
        @time for itPH = 1:100
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
            Rp    .= .-∇V
            # Residual check
            errVx = norm(Rx); errVy = norm(Ry); errPt = norm(Rp)
            if itPH==1 errVx0=errVx; errVy0=errVy; errPt0=errPt; end
            err = maximum([errVx/errVx0, errVy/errVy0, errPt/errPt0])
            @printf("it = %04d, itPH = %02d iter = %06d iter/nx = %03d, err = %1.3e norm[Rx=%1.3e, Ry=%1.3e, ∇V=%1.3e] \n", it, itPH, iter, iter/ncx, err, errVx/errVx0, errVy/errVy0, errPt/errPt0)
            if (err<ϵ) break end
            # Set tolerance of velocity solve proportional to residual
            # ϵ_vel = ϵ/10
            ϵ_vel = err*7e-3
            # ϵ_vel = errPt/errPt0*1e-4
            # ϵ_vel = errVy/errVy0*1e-4
            while (err>ϵ_vel && iter<=iterMax)
                iter  += 1 
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
                # Deviatoric stress
                Txx   .= 2.0.*ηc.*Exx .+ γ.*∇V
                Tyy   .= 2.0.*ηc.*Eyy .+ γ.*∇V
                Txy   .= 2.0.*ηv.*Exy 
                # Residuals
                Rx    .= (.-(Pt[2:end,:] .- Pt[1:end-1,:])./Δx .+ (Txx[2:end,:] .- Txx[1:end-1,:])./Δx .+ (Txy[2:end-1,2:end] .- Txy[2:end-1,1:end-1])./Δy)
                Ry    .= (.-(Pt[:,2:end] .- Pt[:,1:end-1])./Δy .+ (Tyy[:,2:end] .- Tyy[:,1:end-1])./Δy .+ (Txy[2:end,2:end-1] .- Txy[1:end-1,2:end-1])./Δx)
                # Damping-pong
                dVxdt .= αVx.*dVxdt .+ βVx.*(1.0./cVxC).*Rx
                dVydt .= αVy.*dVydt .+ βVy.*(1.0./cVyC).*Ry
                # PT updates
                Vx[2:end-1,2:end-1] .+= dVxdt.*dtVx
                Vy[2:end-1,2:end-1] .+= dVydt.*dtVy
                # Residual check
                if mod(iter, nout)==0
                    errVx = norm(Rx); errVy = norm(Ry)
                    if iter==nout errVx00=errVx; errVy00=errVy; end
                    err = maximum([errVx./errVx00, errVy./errVy00])
                    push!(err_evo_V, errVx/errVx00); push!(err_evo_P, errPt/errPt0); push!(err_evo_it, itg)
                    dVx .= dVxdt.*dtVx
                    dVy .= dVydt.*dtVy
                    noisy ? @printf("it = %d, iter = %d, err = %1.3e norm[Rx=%1.3e, Ry=%1.3e] \n", it, iter, err, norm_Rx, norm_Ry) : nothing
                    # Update damping
                    λminV  = (abs.((sum(dVx.*(Rx .- Rx0)./cVxC)) + abs.((sum(dVy.*(Ry .- Ry0)./cVyC)) )/ ( (sum(dVx.*dVx))) + (sum(dVy.*dVy))) )
                    cVx .= 2*sqrt.(λminV)*c_fact
                    cVy .= 2*sqrt.(λminV)*c_fact
                    βVx .= 2 .* dtVx ./ (2 .+ cVx.*dtVx)
                    βVy .= 2 .* dtVy ./ (2 .+ cVy.*dtVy)
                    αVx .= (2 .- cVx.*dtVx) ./ (2 .+ cVx.*dtVx)
                    αVy .= (2 .- cVy.*dtVy) ./ (2 .+ cVy.*dtVy)
                end
            end
            Pt .+= γ.*(.-∇V)
        end
        push!(evo_t, t); push!(evo_Txx, maximum(Txx))

        # Plotting
        p1 = pt.heatmap(xc, yc, Pt' , aspect_ratio=1, c=:inferno, title="Pt", xlims=(-Lx/2,Lx/2))
        p2 = pt.heatmap(xc, yv, Vy[2:end-1,:]' , aspect_ratio=1, c=:inferno, title="Vy", xlims=(-Lx/2,Lx/2))
        p4 = pt.plot(title="Convergence") 
        p4 = pt.plot!(err_evo_it, log10.(err_evo_V), label="V")
        p4 = pt.plot!(err_evo_it, log10.(err_evo_P), label="P")
        p4 = pt.plot!(err_evo_it, log10.(ϵ.*ones(size(err_evo_it))), label="tol")  
        p3 = pt.heatmap(xc, yc, log10.(ηc)' , aspect_ratio=1, c=:inferno, title="ηc", xlims=(-Lx/2,Lx/2))
        display(pt.plot(p1, p2, p3, p4))

        @show iter/ncx
        n   = length(ηc)
        @show η_h = 1.0 / sum(1.0/n ./ηc)
        @show η_g = exp( sum( 1.0/n*log.(ηc)))
        @show η_a = mean(ηc)
        
    end
    return
end

n = 8
Stokes2D(n, :DYREL, false)
