import JustPIC: getcell, setcell!, @cell
import Plots as pt
# assume rectangular elements
using Stokes3D, Statistics, LinearAlgebra, Plots, WriteVTK, Printf, StaticArrays, CellArrays
const year     = 365*3600*24
const USE_GPU  = true #* false
const GPU_ID   = 0
const USE_MPI  = false
const Visu     = false

using ParallelStencil
using ParallelStencil.FiniteDifferences3D

using TimerOutputs

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
    CUDA.device!(GPU_ID) # select GPU
else
    @init_parallel_stencil(Threads, Float64, 3)
end

include("./kernels/Stokes3D_FCFV_kernels.jl")

####################################################################
####################################################################
####################################################################

@views function Stokes2D_FCFV_PHDR(;n=1) 

    @info "Starting Stokes3D FCFV"

    to = TimerOutput()
    @timeit to "all" begin

    Save     = false
    out_path = "./"
    it        = 0

    L  = ( x = 1.0,  y = 1.0,  z = 1.0 )    
    Nc = ( x = n*8, y = n*8, z = n*8)
    x, y, z  = (min = -L.x/2, max = L.x/2), (min = -L.y/2, max = L.y/2), (min = -L.z/2, max = L.z/2)
    Δ  = ( x = L.x/Nc.x, y = L.y/Nc.y, z = L.z/Nc.z)     
    τr  = 4#(Nc.x + Nc.y)

    numerics  = Numerics(
        ϵ       = 1e-5,
        ϵrel    = 1e-2,
        γ       = 4e2,
        niterPH = 100,
        niterPT = 1e4,
        nout    = 100,
        cfact   = 1.0/2,
        CFL     = .45,
        PC      = true,
        ϵ_PowIt = 1e-3,
        noisy   = false,  
    )

    ndim = 3

    ##############################################
    Ω = @ones(Nc.x, Nc.y, Nc.z)
    Ω .= Δ.x*Δ.y*Δ.z

    Γ  = @zeros(Nc.x+1, Nc.y+0, Nc.z+0, celldims=(6))
    N  = @zeros(Nc.x+1, Nc.y+0, Nc.z+0, celldims=(6, ndim))

    @parallel Set_Γ_N!( Γ, N, Δ )

    #############################################
    
    cents = (
        x      = LinRange(x.min+Δ.x/2, x.max-Δ.x/2, Nc.x),
        y      = LinRange(y.min+Δ.y/2, y.max-Δ.y/2, Nc.y),
        z      = LinRange(z.min+Δ.z/2, z.max-Δ.z/2, Nc.z),
    )
    verts = (
        x      = LinRange(x.min, x.max, Nc.x+1),
        y      = LinRange(y.min, y.max, Nc.y+1),
        z      = LinRange(z.min, z.max, Nc.z+1),
    )

    V = (
        e      = @zeros(Nc.x+0, Nc.y+0, Nc.z+0, celldims=(ndim)),
        x      = @zeros(Nc.x+1, Nc.y+0, Nc.z+0, celldims=(ndim)),
        y      = @zeros(Nc.x+0, Nc.y+1, Nc.z+0, celldims=(ndim)),
        z      = @zeros(Nc.x+0, Nc.y+0, Nc.z+1, celldims=(ndim)),
    )

    σ = (
        e      = @zeros(Nc.x+0, Nc.y+0, Nc.z+0, celldims=(ndim,ndim)),
        x      = @zeros(Nc.x+1, Nc.y+0, Nc.z+0, celldims=(ndim,ndim)),
        y      = @zeros(Nc.x+0, Nc.y+1, Nc.z+0, celldims=(ndim,ndim)),
        z      = @zeros(Nc.x+0, Nc.y+0, Nc.z+1, celldims=(ndim,ndim)),
    )

    b     = @zeros(Nc.x, Nc.y, Nc.z, celldims=(ndim))
    η     =  @ones(Nc.x, Nc.y, Nc.z)
    ηv    =  @ones(Nc.x+1, Nc.y+1, Nc.z+1)
    P     = @zeros(Nc.x, Nc.y, Nc.z)
    τ     = @zeros(Nc.x, Nc.y, Nc.z, celldims=(ndim,ndim))
    ε̇     = @zeros(Nc.x, Nc.y, Nc.z, celldims=(ndim,ndim))
    ∇V    = @zeros(Nc.x, Nc.y, Nc.z)

    bP    = @zeros(Nc.x, Nc.y, Nc.z)

    R = (
        x      = @zeros(Nc.x+1, Nc.y+0, Nc.z+0, celldims=(ndim)),
        y      = @zeros(Nc.x+0, Nc.y+1, Nc.z+0, celldims=(ndim)),
        z      = @zeros(Nc.x+0, Nc.y+0, Nc.z+1, celldims=(ndim)),
    )

    R_it = ( 
        x      = @zeros(Nc.x+1, Nc.y+0, Nc.z+0, celldims=(ndim)),
        y      = @zeros(Nc.x+0, Nc.y+1, Nc.z+0, celldims=(ndim)),
        z      = @zeros(Nc.x+0, Nc.y+0, Nc.z+1, celldims=(ndim)),
    )

    D = (
        x      = @ones(Nc.x+1, Nc.y+0, Nc.z+0, celldims=(ndim)),
        y      = @ones(Nc.x+0, Nc.y+1, Nc.z+0, celldims=(ndim)),
        z      = @ones(Nc.x+0, Nc.y+0, Nc.z+1, celldims=(ndim)),
    )

    G = (
        x      = @ones(Nc.x+1, Nc.y+0, Nc.z+0, celldims=(ndim)),
        y      = @ones(Nc.x+0, Nc.y+1, Nc.z+0, celldims=(ndim)),
        z      = @ones(Nc.x+0, Nc.y+0, Nc.z+1, celldims=(ndim)),
    )

    ∂V∂τe = ( 
        x      = @zeros(Nc.x+1, Nc.y+0, Nc.z+0, celldims=(ndim)),
        y      = @zeros(Nc.x+0, Nc.y+1, Nc.z+0, celldims=(ndim)),
        z      = @zeros(Nc.x+0, Nc.y+0, Nc.z+1, celldims=(ndim)),
    )

    ph = (
        x      = @zeros(Nc.x+1, Nc.y+2, Nc.z+2),
        y      = @zeros(Nc.x+2, Nc.y+1, Nc.z+2),
        z      = @zeros(Nc.x+2, Nc.y+2, Nc.z+1),
    )

    𝐶 = (
        α  =     @zeros(Nc.x+0, Nc.y+0, Nc.z+0),
        β  =     @zeros(Nc.x+0, Nc.y+0, Nc.z+0, celldims=(ndim)),
        τe = τr * @ones(Nc.x+0, Nc.y+0, Nc.z+0),
    )

    # Set Dirichlet nodes
    ph.x[[1, end],:,:] .= 3
    ph.y[:,[1, end],:] .= 3
    ph.z[:,:,[1, end]] .= 3 # set Neumann

    errVx0, errVy0, errVz0, errPt0 = 0., 0., 0., 0.
    errVxPT0, errVyPT0, errVzPT0 = 0., 0., 0.
    tot_iter_PT = 0; tot_iter_PH = 0; iterPH = 0
    logErrVx = zeros(numerics.niterPH)
    logErrVy = zeros(numerics.niterPH)
    logErrVz = zeros(numerics.niterPH)
    logErrP  = zeros(numerics.niterPH)

    @timeit to "Dirichlets" @parallel Set_Dirichlets!( V, σ, η, ηv, ph, verts, cents )
    @parallel interp8!(η, ηv)

    Δx = verts.x[2] - verts.x[1]
    Δy = verts.y[2] - verts.y[1]
    Δz = verts.z[2] - verts.z[1]
    𝐶.τe .=  60. 
    𝐶.τe .= η/Δx
    @timeit to "FCFV coeffs" @parallel FCFV_coeffs!( 𝐶, b, V, ph, Γ, Ω, N )
    @parallel V_τ_elem!( V, ε̇, ∇V, η, τ, b, 𝐶, ph, Γ, Ω, N, false, numerics.γ )

    # Iteration parameters
    λmax = 1.0
    λmin = 1.0
    # Set whatever non-zero value inside
    field(V.x,1)[2:end-1,:,:] .= 1.
    field(V.x,2)[2:end-1,:,:] .= 1.
    field(V.y,1)[:,2:end-1,:] .= 1.
    field(V.y,2)[:,2:end-1,:] .= 1.
    field(V.z,1)[:,:,2:end-1] .= 1.
    field(V.z,2)[:,:,2:end-1] .= 1.

    @parallel V_τ_elem!( V, ε̇, ∇V, η, τ, b, 𝐶, ph, Γ, Ω, N, true, numerics.γ )
    @parallel ResidualStokes!( R, V, σ, P, τ, ph, 𝐶, Γ, Ω, N )

    @parallel FCFV_iter_params!( D, G, R_it, b, η, V, P, τ, σ, ph, 𝐶, Γ, Ω, N, true, numerics.γ, numerics.PC)
    @show extrema(field(D.x,1))
    @show λmax = maximum(max( maximum(field(G.x,1)./field(D.x,1)), maximum(field(G.x,2)./field(D.x,2)), maximum(field(G.y,1)./field(D.y,1)), maximum(field(G.y,2)./field(D.y,2)), maximum(field(G.z,1)./field(D.z,1)), maximum(field(G.z,2)./field(D.z,2)) ))

    fill!(V.e.data, 0e0)
    fill!(V.x.data, 0e0)
    fill!(V.y.data, 0e0)
    @timeit to "Dirichlets" @parallel Set_Dirichlets!( V, σ,  η, ηv, ph, verts, cents )
    @parallel interp8!(η, ηv)

    @parallel V_τ_elem!( V, ε̇, ∇V, η, τ, b, 𝐶, ph, Γ, Ω, N, false, numerics.γ )
    
    h = (
        x = 2.0./sqrt.(λmax)*numerics.CFL,
        y = 2.0./sqrt.(λmax)*numerics.CFL,
        z = 2.0./sqrt.(λmax)*numerics.CFL,
    ) 
    c     = 2.0.*sqrt(λmin)
    a1    = (
        x = (2 .- c.*h.x) ./(2 .+ c.*h.x),
        y = (2 .- c.*h.y) ./(2 .+ c.*h.y),
        z = (2 .- c.*h.z) ./(2 .+ c.*h.z),
    )
    a2    = (
        x = 2*h.x ./(2 .+ c*h.x),
        y = 2*h.y ./(2 .+ c*h.y),
        z = 2*h.z ./(2 .+ c*h.z),
    )

    @time for iterPH=1:numerics.niterPH
        tot_iter_PH += 1

        @parallel V_τ_elem!( V, ε̇, ∇V, η, τ, b, 𝐶, ph, Γ, Ω, N, false, numerics.γ )
        @parallel ResidualStokes!( R, V, σ, P, τ, ph, 𝐶, Γ, Ω, N )

        # Check residual 
        errVx = norm(R.x.data)/(length(R.x.data)); if errVx == 0 errVx += 1e-13 end
        errVy = norm(R.y.data)/(length(R.y.data)); if errVy == 0 errVy += 1e-13 end
        errVz = norm(R.z.data)/(length(R.z.data)); if errVz == 0 errVz += 1e-13 end
        errPt = norm(∇V )/(length(∇V )); if errPt == 0 errPt += 1e-13 end    
        if iterPH==1 errVx0, errVy0, errVz0, errPt0 =  errVx, errVy, errVz, errPt end
        @printf(">>>>>>>>>> PH iter %05d \n", iterPH)
        @printf("Rx = %2.4e --- Ry = %2.4e --- Rz = %2.4e --- Rp = %2.4e\n", errVx, errVy, errVz, errPt)
        ( max(errVx, errVy, errVz) < numerics.ϵ && errPt < numerics.ϵ ) && break
        logErrVx[iterPH] = errVx; logErrVy[iterPH] = errVy; logErrVz[iterPH] = errVz; logErrP[iterPH] = errPt
        isnan(errVx) && error("blam, NaNs!")
        iterPT = 0
        errVx0, errVy0, errVz0 = 0., 0., 0.

        iterPT = 0
        @timeit to "PT iterations" for iter=1:numerics.niterPT
            iterPT += 1

            @timeit to "Copy R" @parallel (1: max(length(R.x.data), length(R.y.data))) copy_R2!(R_it, R)
            @timeit to "V_τ" @parallel V_τ_elem!( V, ε̇, ∇V, η, τ, b, 𝐶, ph, Γ, Ω, N, true, numerics.γ )
            @timeit to "Residual" @parallel ResidualStokes!( R, V, σ, P, τ, ph, 𝐶, Γ, Ω, N )
            # @timeit to "Update fields" @parallel (1:max(length(V.x.data), length(V.y.data))) UpdateRatesFields3!(V, ∂V∂τe, R, a1, a2, h)
            @timeit to "Update fields" @parallel (1:max(length(V.x.data), length(V.y.data))) UpdateRatesFields4!(V, D, ∂V∂τe, R, a1, a2, h)

            @timeit to "Check Convergence" if iterPT<=2 || mod(iterPT, numerics.nout)==0
                errVxPT = norm(R.x.data)/(length(R.x.data)); if errVxPT == 0 errVxPT += 1e-13 end
                errVyPT = norm(R.y.data)/(length(R.y.data)); if errVyPT == 0 errVyPT += 1e-13 end
                errVzPT = norm(R.z.data)/(length(R.z.data)); if errVzPT == 0 errVzPT += 1e-13 end
                if iterPT==1 errVxPT0, errVyPT0, errVzPT0 =  errVxPT, errVyPT, errVzPT end
                if numerics.noisy
                    @printf(">>>>> PT iter %05d \n", iterPT )
                    @printf("R x abs = %2.4e --- Ry abs = %2.4e --- Rz abs = %2.4e\n", errVxPT, errVyPT, errVzPT)
                end
                isnan(errVxPT) && error("blam, NaNs!")
                ( min(errVxPT/errVxPT0, errVyPT/errVyPT0) < numerics.ϵrel ) && break
                # top = sum(.-(h.x*field(∂V∂τe.x,1)).*(field(R.x,1).-field(R_it.x,1))) + sum(.-(h.x*field(∂V∂τe.x,2)).*(field(R.x,2).-field(R_it.x,2))) + sum(.-(h.x*field(∂V∂τe.y,1)).*(field(R.y,1).-field(R_it.y,1))) + sum(.-(h.x*field(∂V∂τe.y,2)).*(field(R.y,2).-field(R_it.y,2))) + sum(.-(h.x*field(∂V∂τe.z,1)).*(field(R.z,1).-field(R_it.z,1))) + sum(.-(h.x*field(∂V∂τe.z,2)).*(field(R.z,2).-field(R_it.z,2))) 
                # bot = sum(  (h.x*field(∂V∂τe.x,1)).*(h.x*field(∂V∂τe.x,1))         ) + sum(  (h.x*field(∂V∂τe.x,2)).*(h.x*field(∂V∂τe.x,2))         ) + sum(  (h.x*field(∂V∂τe.y,1)).*(h.x*field(∂V∂τe.y,1))         ) + sum(  (h.x*field(∂V∂τe.y,2)).*(h.x*field(∂V∂τe.y,2))         ) + sum(  (h.x*field(∂V∂τe.z,1)).*(h.x*field(∂V∂τe.z,1))         ) + sum(  (h.x*field(∂V∂τe.z,2)).*(h.x*field(∂V∂τe.z,2))         )
                # λmin = abs(top/bot)*numerics.cfact
                # c     = 2.0.*sqrt(λmin)*numerics.cfact
                # a1    = (2 .- c.*h.x) ./(2 .+ c.*h.x)
                # a2    = 2*h.x ./(2 .+ c*h.x)
                 top = 
                    h.x * sum(.-(field(∂V∂τe.x,1)).*(field(R.x,1).-field(R_it.x,1))./field(D.x,1)) + 
                    h.x * sum(.-(field(∂V∂τe.x,2)).*(field(R.x,2).-field(R_it.x,2))./field(D.x,2)) +
                    h.y * sum(.-(field(∂V∂τe.y,1)).*(field(R.y,1).-field(R_it.y,1))./field(D.y,1)) + 
                    h.y * sum(.-(field(∂V∂τe.y,2)).*(field(R.y,2).-field(R_it.y,2))./field(D.y,2)) + 
                    h.z * sum(.-(field(∂V∂τe.z,1)).*(field(R.z,1).-field(R_it.z,1))./field(D.z,1)) + 
                    h.z * sum(.-(field(∂V∂τe.z,2)).*(field(R.z,2).-field(R_it.z,2))./field(D.z,2)) 
                bot = 
                   (
                    h.x^2 * sum( field(∂V∂τe.x,1).^2 + field(∂V∂τe.x,2).^2 ) +
                    h.y^2 * sum( field(∂V∂τe.y,1).^2 + field(∂V∂τe.y,2).^2 ) +
                    h.z^2 * sum( field(∂V∂τe.z,1).^2 + field(∂V∂τe.z,2).^2 )
                    )
                # @show top
                # @show bot
                λmin  = abs(top/bot)*numerics.cfact
                c     = 2 * sqrt(λmin) *numerics.cfact
                a1    = (
                    x = (2 .- c.*h.x) ./(2 .+ c.*h.x),
                    y = (2 .- c.*h.y) ./(2 .+ c.*h.y),
                    z = (2 .- c.*h.z) ./(2 .+ c.*h.z),
                )
                a2    = (
                    x = 2*h.x ./(2 .+ c*h.x),
                    y = 2*h.y ./(2 .+ c*h.y),
                    z = 2*h.z ./(2 .+ c*h.z),
                )
            end
        end
        tot_iter_PT += iterPT
        @timeit to "Schur" @parallel UpdatePressure_SchurComplement!( P, bP, ∇V, ε̇, numerics.γ )
        @show minimum(P), maximum(P)
    end

        @info "Total number of PH iterations: $(tot_iter_PH)"
        @info "Total number of PT iterations: $(tot_iter_PT)"
    end
    @show to

    if Visu
    Vx   = field(V.e, 1)
    Vy   = field(V.e, 2)

    Vx   = 0.5*(field(V.x, 1)[1:end-1,:,:] .+ field(V.x, 1)[2:end-0,:,:] )
    Vy   = 0.5*(field(V.x, 2)[1:end-1,:,:] .+ field(V.x, 2)[2:end-0,:,:] )

    # Vmag   = sqrt.(Vx.^2 + Vy.^2)
    τxx  = field(ε̇, 1, 1)
    τxy  = field(ε̇, 2, 1)
    τyy  = field(ε̇, 2, 2)
    τII  = sqrt.(τxx.^2 .+ τxy.^2)
   
    imid = Int(ceil(Nc.z/2))

    p1=pt.heatmap(cents.x, cents.y,  Array(Vx[:,:,imid]'), aspect_ratio=1, xlims=(x.min,x.max))
    p2=pt.heatmap(cents.x, cents.y,  Array(Vy[:,:,imid]'), aspect_ratio=1, xlims=(x.min,x.max))
    p3=pt.heatmap(cents.x, cents.y, Array(log10.(η[:,:,imid])'), aspect_ratio=1, xlims=(x.min,x.max))
    p4=pt.heatmap(cents.x, cents.y,   Array(P[:,:,imid]'), aspect_ratio=1, xlims=(x.min,x.max), clims=(-3,3)) 

    display(pt.plot(p1, p2, p3, p4, layout=(2,2)))
    end

    # ##############################################
    # Vertices coordinates 
    X = zeros(Nc.x, Nc.y, Nc.z)
    Y = zeros(Nc.x, Nc.y, Nc.z)
    Z = zeros(Nc.x, Nc.y, Nc.z)

    for k in 1:Nc.z, j in 1:Nc.y, i in 1:Nc.x
        X[i,j,k] = cents.x[i]
        Y[i,j,k] = cents.y[j]
        Z[i,j,k] = cents.z[k]
    end

    Stress = @zeros(3, 3, Nc.x, Nc.y, Nc.z, eltype=Float32)
    @parallel (1:Nc.x, 1:Nc.y, 1:Nc.z) fill_tensor_array!(Stress, τ)

    SRate = @zeros(3, 3, Nc.x, Nc.y, Nc.z, eltype=Float32)
    @parallel (1:Nc.x, 1:Nc.y, 1:Nc.z) fill_tensor_array!(SRate, ε̇)

    # remove mean
    P .= P .- mean(P)
    Ve_cpu = Array(V.e)
    filename = @sprintf( "./MultiInclusionsFCFV" )
    vtkfile               = vtk_grid(filename, X, Y, Z)
    vtkfile["P"]          = Float32.(Array(P))
    vtkfile["eta"]        = Float32.(Array(η)) 
    vtkfile["V"]          = [getcell(Ve_cpu, i,j,k) for i in axes(Ve_cpu,1), j in axes(Ve_cpu,2), k in axes(Ve_cpu,3)]
    vtkfile["Stress"]     = Array(Stress)
    vtkfile["Strain rate"]= Array(SRate)
    outfiles              = vtk_save(vtkfile)

    return nothing
end

n=8

Stokes2D_FCFV_PHDR(;n=8)
