# try JustPIC on CPU
using Statistics, LinearAlgebra, WriteVTK, JLD2, Printf, TimerOutputs, CellArrays
using JustPIC, JustPIC._3D
import JustPIC._3D: cellaxes, @cell, @index
const y   = 365*3600*24
const cmy = y*100
const My  = 1e6*y  

const USE_GPU   = false
const GPU_ID    = 6
const USE_MPI   = false

const interp_p2g = 1       # 0: arith --- 1 harm   
const interp_g2g = 0       # 0: arith --- 1 harm   
const p2g        = :vertex # center of vertex
const advect     = :linear # linear or MQS

const Visu      = true
const ThreeD    = false
const SaveGrid  = true 
const SavePart  = false 
const SaveStep  = 50
const SaveCheck = true
const SaveCheckStep = 50
const to = TimerOutput()

Visu ? import Plots  as pt : nothing
SaveGrid || SavePart ? using WriteVTK : nothing
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    using CUDA
    @init_parallel_stencil(CUDA, Float64, 3)
    CUDA.device!(GPU_ID) # select GPU
else
    @init_parallel_stencil(Threads, Float64, 3)
end

# JustPIC on CPU
const backend = JustPIC.CPUBackend 

include( "./kernels/Stokes3D_kernels_v1.jl")
include( "./kernels/Stokes3D_SchurComplement_kernels_v1.jl")

@parallel_indices (i,j,k) function SumRatios!( SumRatios, phase_rat_vert,  Nphases )
    if i<=size(SumRatios, 1) && j<=size(SumRatios, 2) && k<=size(SumRatios, 3)
        SumRatios[i,j,k] = 0.
        for ph=1:Nphases
            SumRatios[i,j,k] += @index phase_rat_vert[ph, i,j, k ]
        end
    end
    return nothing
end

function InitialFieldsParticles!( phases, px, py, pz, index, scales )


    xc = (0.333, 0.153, 0.079, -0.217, 0.328, -0.337, -0.152, -0.039, -0.463, 0.280, -0.209, 0.302, -0.159, -0.356, -0.047, -0.289, -0.482, -0.145, 0.440, -0.471, 0.091, -0.402, -0.198, -0.114, -0.332, 0.337, 0.326, 0.148, -0.144, -0.394, 0.293, -0.479, 0.478, -0.495, -0.064, -0.474, -0.227, 0.323, -0.293, -0.364, 0.298, 0.462, -0.084, -0.381, -0.123, 0.477, -0.430, -0.134, 0.353, 0.218,)
    yc = (0.123, -0.427, -0.367, -0.020, -0.128, 0.244, -0.347, 0.359, 0.289, 0.260, 0.470, -0.241, -0.379, -0.048, -0.326, -0.273, 0.327, -0.190, 0.226, 0.265, -0.407, 0.069, -0.122, 0.403, -0.051, -0.037, 0.296, 0.138, -0.041, -0.197, -0.456, -0.123, -0.103, 0.189, 0.386, 0.403, 0.025, -0.293, -0.331, 0.014, -0.073, -0.020, 0.479, 0.448, 0.103, 0.380, 0.403, -0.081, 0.131, 0.146,)
    zc = (0.436, 0.431, 0.318, 0.319, 0.302, 0.123, -0.493, 0.077, -0.317, -0.373, 0.245, 0.099, 0.434, 0.309, -0.434, 0.458, 0.168, 0.389, -0.178, 0.311, 0.242, -0.425, -0.371, 0.414, 0.374, 0.339, -0.209, 0.047, 0.090, 0.060, -0.114, -0.069, -0.098, -0.309, 0.034, -0.387, 0.456, 0.446, 0.225, 0.270, 0.390, -0.047, -0.000, 0.218, -0.014, 0.386, -0.313, -0.222, -0.090, -0.143,)
    rc = (0.134, 0.149, 0.053, 0.030, 0.103, 0.114, 0.045, 0.112, 0.110, 0.131, 0.110, 0.080, 0.009, 0.129, 0.113, 0.083, 0.033, 0.111, 0.054, 0.056, 0.013, 0.137, 0.035, 0.091, 0.014, 0.133, 0.104, 0.086, 0.094, 0.109, 0.026, 0.033, 0.046, 0.086, 0.098, 0.030, 0.048, 0.008, 0.072, 0.096, 0.079, 0.036, 0.133, 0.008, 0.039, 0.082, 0.027, 0.018, 0.095, 0.002,)
    ph = (2.0, 3.0, 3.0, 3.0, 2.0, 2.0, 3.0, 2.0, 2.0, 3.0,
          3.0, 2.0, 3.0, 3.0, 3.0, 2.0, 2.0, 3.0, 2.0, 2.0,
          3.0, 2.0, 3.0, 3.0, 3.0, 2.0, 2.0, 3.0, 2.0, 2.0,
          3.0, 2.0, 3.0, 3.0, 3.0, 2.0, 2.0, 3.0, 2.0, 2.0,
          2.0, 3.0, 3.0, 3.0, 2.0, 2.0, 3.0, 2.0, 2.0, 3.0)

    for i=1:size(phases,1), j=1:size(phases,2), k=1:size(phases,3)
        I = (i,j,k)
        @inbounds for ip in cellaxes(phases)
        # quick escape
        @index(index[ip, I...]) == 0 && continue

        x = @index px[ip, I...]
        y = @index py[ip, I...]
        z = @index pz[ip, I...]

        # background
        @index phases[ip, I...] = 1.0

        for ii in eachindex(xc)
            if ( (x-xc[ii])^2 + (y-yc[ii])^2 + (z-zc[ii])^2 ) < (rc[ii])^2
                @index phases[ip, I...] = ph[ii]
            end
        end
    end
    end
    return nothing
end

############################################## KERNELS ##############################################

@parallel_indices (i,j,k) function ViscosityVertex!( rheo, params, ε̇, ηrel, phase_rat,  Nphases, interp)

    if i<=size(rheo.ηve_v, 1) && j<=size(rheo.ηve_v, 2) && k<=size(rheo.ηve_v, 3)
        
        # ε̇xx2  = 0.125*(ε̇.xx[i,j,k  ]^2 + ε̇.xx[i,j+1,k  ]^2 + ε̇.xx[i+1,j,k  ]^2 + ε̇.xx[i+1,j+1,k  ]^2)
        # ε̇xx2 += 0.125*(ε̇.xx[i,j,k+1]^2 + ε̇.xx[i,j+1,k+1]^2 + ε̇.xx[i+1,j,k+1]^2 + ε̇.xx[i+1,j+1,k+1]^2)
        # ε̇yy2  = 0.125*(ε̇.yy[i,j,k  ]^2 + ε̇.yy[i,j+1,k  ]^2 + ε̇.yy[i+1,j,k  ]^2 + ε̇.yy[i+1,j+1,k  ]^2)
        # ε̇yy2 += 0.125*(ε̇.yy[i,j,k+1]^2 + ε̇.yy[i,j+1,k+1]^2 + ε̇.yy[i+1,j,k+1]^2 + ε̇.yy[i+1,j+1,k+1]^2)
        # ε̇xy2  = 0.5*(ε̇.xy[i,j,k]^2 + ε̇.xy[i,j,k+1]^2)
        # ε̇xz2  = 0.5*(ε̇.xz[i,j,k]^2 + ε̇.xz[i,j+1,k]^2)
        # ε̇yz2  = 0.5*(ε̇.yz[i,j,k]^2 + ε̇.yz[i+1,j,k]^2)
        # ε̇II  = sqrt( 0.5*(ε̇xx2 + ε̇yy2) + ε̇xy2 + ε̇xz2 + ε̇yz2 )
        # ε̇.II[i,j,k] = ε̇II
        
        η_eff = 0.
        ρ_eff = 0.

        for ph=1:Nphases
            ratio = @index phase_rat[ph, i, j, k ]
            if interp == 1 η_eff += ratio / params.η0[ph] end
            if interp == 0 η_eff += ratio * params.η0[ph] end
            ρ_eff += ratio * params.ρ0[ph]
        end
        if interp == 1 rheo.ηve_true[i,j,k] = ηve_true = inv(η_eff) end
        if interp == 0 rheo.ηve_true[i,j,k] = ηve_true = η_eff      end

        rheo.ρv[i,j,k]       = ρ_eff

        rheo.ηve_v[i,j,k]    = ηve_true #exp(ηrel*log(ηve_true) + (1-ηrel)*log(rheo.ηve_v[i,j,k]))    
    end
    return nothing
end

@parallel_indices (i,j,k) function ViscosityCenter!( rheo, params, ε̇, ηrel, phase_rat,  Nphases, interp)

    if 1<i<size(rheo.ηve_c, 1) && 1<j<size(rheo.ηve_c, 2) && 1<k<size(rheo.ηve_c, 3)
        
        η_eff = 0.
        ρ_eff = 0.

        for ph=1:Nphases
            ratio = @index phase_rat[ph, i-1, j-1, k-1 ]
            if interp == 1 η_eff += ratio / params.η0[ph] end
            if interp == 0 η_eff += ratio * params.η0[ph] end
            ρ_eff += ratio * params.ρ0[ph]
        end
        if interp == 1 rheo.ηve_true[i,j,k] = ηve_true = inv(η_eff) end
        if interp == 0 rheo.ηve_true[i,j,k] = ηve_true = η_eff      end

        rheo.ρc[i,j,k]       = ρ_eff

        rheo.ηve_c[i,j,k]    = ηve_true #exp(ηrel*log(ηve_true) + (1-ηrel)*log(rheo.ηve_v[i,j,k]))    
    end
    return nothing
end

@parallel_indices (i,j,k) function RogerVertex!(b, rheo, params)
    if 1<i<size(b.y, 1) && j<=size(b.y, 2) && 1<k<size(b.y, 3)
        ρ          = 0.25*(rheo.ρv[i-1,j,k-1] + rheo.ρv[i-1,j,k] + rheo.ρv[i,j,k-1] + rheo.ρv[i,j,k])
        b.y[i,j,k] = ρ*params.gy
    end
    return nothing
end

@parallel_indices (i,j,k) function RogerCenter!(b, rheo, params)
    if 1<i<size(b.y, 1) && j<=size(b.y, 2) && 1<k<size(b.y, 3)
        ρ          = 0.5*(rheo.ρc[i,j,k] + rheo.ρc[i,j+1,k])
        b.y[i,j,k] = ρ*params.gy
    end
    return nothing
end

@parallel_indices (i,j,k) function InitialFields!(V, verts, params)
    if i<=size(V.x, 1) && j<=size(V.x, 2) && k<=size(V.x, 3)
        V.x[i,j,k] = verts.x[i] * params.ε̇
    end
    if i<=size(V.y, 1) && j<=size(V.y, 2) && k<=size(V.y, 3)
        V.y[i,j,k] = -verts.y[j] * params.ε̇
    end
    return nothing
end

############################################## MAIN CODE ##############################################

@views function Stokes3D_PHDR(; n=1,  ALE=false, restart=false, restart_step=0, end_step=0)

    # out_path = "./_RUN04/"
    # isdir(out_path) ? nothing : mkdir(out_path)

    BuoyancyDriven = false

    scales  = (τ=1e0, L=1e0, t=1e0) # kg = scale->S * scale->L * pow(scale->t,2.0);
    derived = (ρ = (scales.τ * scales.L * scales.t^2)/scales.L^3, η=scales.τ*scales.t, V=scales.L/scales.t, a=scales.L/scales.t^2)
    scales  = merge(scales,derived)

    also_z = ThreeD ? 1.0  : 0.0
    @show Nc = ( x = n*32, y = n*32, z = n*32 )

    Nphases = 3

     # Load checkpoint data
     if restart
        file          = @sprintf("./Checkpoint%05d.jld2", restart_step)
        @info "Starting from $(file)"
        data          = load(file)
        particles     = TA(backend)(Float64, data["particles"])
        phases        = TA(backend)(Float64, data["phases"])
        phase_ratios  = TA(backend)(Float64, data["phase_ratios"])
        particle_args = TA(backend).(Float64, data["particle_args"])
        if USE_GPU
        V             = (
            x = CuArray(Float64.(data["Vx"])),
            y = CuArray(Float64.(data["Vy"])),
            z = CuArray(Float64.(data["Vz"]))
        )
        P             = CuArray(Float64.(data["P"]))
        else
            V             = (
                x = TA(backend)(Float64.(data["Vx"])),
                y = TA(backend)(Float64.(data["Vy"])),
                z = TA(backend)(Float64.(data["Vz"]))
            )
            P             = TA(backend)(Float64.(data["P"]))
        end
        xlims         = data["xlims"]
        ylims         = data["ylims"]
        zlims         = data["zlims"]
        t             = data["t"]
        L             = ( x =(xlims[2]-xlims[1]), y =(ylims[2]-ylims[1]), z =(zlims[2]-zlims[1]) )
        it0           = restart_step + 1
        ε̇bg           = 0.
    else
        @info "Starting Stokes3D!"
        xlims = [-0.5, 0.5]./scales.L
        ylims = [-0.5, 0.5]./scales.L
        zlims = [-0.5, 0.5]./scales.L
        L  = ( x = diff(xlims)[1], y = diff(ylims)[1], z = diff(zlims)[1] )  
        t     = 0.
        it0   = 1
        ε̇bg   = 0.
        V = (
            x      = @zeros(Nc.x+1, Nc.y+2, Nc.z+2),
            y      = @zeros(Nc.x+2, Nc.y+1, Nc.z+2),
            z      = @zeros(Nc.x+2, Nc.y+2, Nc.z+1),
        )
    end

    Δ  = ( x = L.x/Nc.x, y = L.y/Nc.y, z = L.z/Nc.z )
    if BuoyancyDriven
        physics = (
            ε̇       = 1.0e-10,
            ηref    = 1e0,
            η0      = (  1e0/scales.η,   1e-2/scales.η,   1e2/scales.η),
            ρ0      = (  1.0/scales.ρ,   1.1/scales.ρ,    0.9/scales.ρ),
            r       = 0.2,
            gy      = -1.0/scales.a,
            Vx      = 0e0/scales.V,
            Vy      = 0e0/scales.V,
        )
    else    
        physics = (
            ε̇       = 1.0,
            ηref    = 1e0,
            η0      = (  1e0/scales.η,   1e-2/scales.η,   1e2/scales.η),
            ρ0      = (  1.0/scales.ρ,   1.0/scales.ρ,    1.0/scales.ρ),
            r       = 0.2,
            gy      = -0.0/scales.a,
            Vx      = 0e0/scales.V,
            Vy      = 0e0/scales.V,
        )
    end
   
    numerics  = (
        ϵ       = 1e-6,
        ϵrel    = 1e-3,
        ϵauto   = false,
        ϵfact   = 0.01,
        γ       = 50.0,
        γauto   = false,
        γfact   = 15.0,
        niterPH = 100,
        niterPT = 5e4,
        niterCG = 5e4,
        nout    = 100,
        cfact   = 0.5,
        CFL     = 0.99,
        PC      = true,
        ϵ_PowIt = 1e-4,
        noisy   = false,
        solver  = :DYREL,
        nt      = end_step,
        Δt      = 5e-2,
        ηrel    = 1.0,
        maxloc  = false,
        Δτloc   = false,
        𝐶       = 0.25,
        λdim    = false,
    )    
    # Allocate arrays
    Sz = (x=size(V.x), y=size(V.y), z=size(V.z))
    In = (x=(2:Nc.x+1-1, 2:Nc.y+2-1, 2:Nc.z+2-1), y=(2:Nc.x+2-1, 2:Nc.y+1-1, 2:Nc.z+2-1), z=(2:Nc.x+2-1, 2:Nc.y+2-1, 2:Nc.z+1-1))
    cents = (
            x      = LinRange(xlims[1]-Δ.x/2, xlims[2]+Δ.x/2, Nc.x+2),
            y      = LinRange(ylims[1]-Δ.y/2, ylims[2]+Δ.y/2, Nc.y+2),
            z      = LinRange(zlims[1]-Δ.z/2, zlims[2]+Δ.z/2, Nc.z+2),
    )
    cents_in = (
        x      = LinRange(xlims[1]+Δ.x/2, xlims[2]-Δ.x/2, Nc.x+0),
        y      = LinRange(ylims[1]+Δ.y/2, ylims[2]-Δ.y/2, Nc.y+0),
        z      = LinRange(zlims[1]+Δ.z/2, zlims[2]-Δ.z/2, Nc.z+0),
    )
    verts = (
        x      = LinRange(xlims[1], xlims[2], Nc.x+1),
        y      = LinRange(ylims[1], ylims[2], Nc.y+1),
        z      = LinRange(zlims[1], zlims[2], Nc.z+1),
    )
    ε̇ = (
        xx      = @zeros(Nc.x+2, Nc.y+2, Nc.z+2),
        yy      = @zeros(Nc.x+2, Nc.y+2, Nc.z+2),
        zz      = @zeros(Nc.x+2, Nc.y+2, Nc.z+2),
        xy      = @zeros(Nc.x+1, Nc.y+1, Nc.z+2),
        xz      = @zeros(Nc.x+1, Nc.y+2, Nc.z+1),
        yz      = @zeros(Nc.x+2, Nc.y+1, Nc.z+1),
        II      = @zeros(Nc.x+1, Nc.y+1, Nc.z+1),
    )
    τ = (
        xx      = @zeros(Nc.x+2, Nc.y+2, Nc.z+2),
        yy      = @zeros(Nc.x+2, Nc.y+2, Nc.z+2),
        zz      = @zeros(Nc.x+2, Nc.y+2, Nc.z+2),
        xy      = @zeros(Nc.x+1, Nc.y+1, Nc.z+2),
        xz      = @zeros(Nc.x+1, Nc.y+2, Nc.z+1),
        yz      = @zeros(Nc.x+2, Nc.y+1, Nc.z+1),
    )
    P    = @zeros(Nc.x+2, Nc.y+2, Nc.z+2)
    ∇V   = @zeros(Nc.x+2, Nc.y+2, Nc.z+2) 
    RP   = @zeros(Nc.x+0, Nc.y+0, Nc.z+0) 
    bP   = @zeros(Nc.x+2, Nc.y+2, Nc.z+2)


    V_CPU = (
        x      = zeros(Sz.x),
        y      = zeros(Sz.y),
        z      = zeros(Sz.z),
    )
    phv_GPU = @zeros(Nc.x+1, Nc.y+1, Nc.z+1, celldims=(Nphases))
    phc_GPU = @zeros(Nc.x+0, Nc.y+0, Nc.z+0, celldims=(Nphases))

    R = (
        x      = @zeros(Sz.x),
        y      = @zeros(Sz.y),
        z      = @zeros(Sz.z),
    )
    R_it = (
        x      = @zeros(Sz.x),
        y      = @zeros(Sz.y),
        z      = @zeros(Sz.z),
    )
    ∂V∂τ = (
        x      = @zeros(Sz.x),
        y      = @zeros(Sz.y),
        z      = @zeros(Sz.z),
    )
    b = (  # RHS for Stokes
        x      = @zeros(Sz.x),
        y      = @zeros(Sz.y),
        z      = @zeros(Sz.z),
    )
    D = (
        x      = @ones(Sz.x),
        y      = @ones(Sz.y),
        z      = @ones(Sz.z),
    )
    D_SC = (
        x      = @ones(Sz.x),
        y      = @ones(Sz.y),
        z      = @ones(Sz.z),
    )
    G = (
        x      = @ones(Sz.x),
        y      = @ones(Sz.y),
        z      = @ones(Sz.z),
    )
    h = (
        x      = @ones(Sz.x),
        y      = @ones(Sz.y),
        z      = @ones(Sz.z),
    )
    rheo = (
        ηve_true = @zeros(Nc.x+1, Nc.y+1, Nc.z+1),
        ηve_v    = @zeros(Nc.x+1, Nc.y+1, Nc.z+1), # compute_maxloc!
        ηve_ml   = @zeros(Nc.x+1, Nc.y+1, Nc.z+1),
        ηve_c    = @zeros(Nc.x+2, Nc.y+2, Nc.z+2),
        ηve_xy   = @zeros(Nc.x+1, Nc.y+1, Nc.z+0),
        ηve_xz   = @zeros(Nc.x+1, Nc.y+0, Nc.z+1),
        ηve_yz   = @zeros(Nc.x+0, Nc.y+1, Nc.z+1),
        phase_v  = @zeros(Nc.x+1, Nc.y+1, Nc.z+1),
        ρv       = @zeros(Nc.x+1, Nc.y+1, Nc.z+1),
        ρc       = @zeros(Nc.x+2, Nc.y+2, Nc.z+2),
    )

    surf  = (
        σyyBC = @zeros(Nc.x+2, Nc.z+2),
        h     = @zeros(Nc.x+2, Nc.z+2),
        h0    = @zeros(Nc.x+2, Nc.z+2),
        ρ     = @zeros(Nc.x+2, Nc.z+2),
        V̄x    = @zeros(Nc.x+2, Nc.z+2),
        dhdx  = @zeros(Nc.x+1, Nc.z+2),
        dh̄dx  = @zeros(Nc.x+2, Nc.z+2),
    )

    # Initialize particles -------------------------------
    grid_vx = (verts.x, cents.y, cents.z)
    grid_vy = (cents.x, verts.y, cents.z)
    grid_vz = (cents.x, cents.y, verts.z)
    
    if !restart 
        nxcell, max_xcell, min_xcell = 40, 60, 20 #25, 45, 10
        particles = init_particles(
            backend, 
            nxcell, 
            max_xcell,
            min_xcell, 
            values(verts),
            values(Δ),
            values(Nc)
        ) # random position by default

        # Initialise phase field
        particle_args = phases, = init_cell_arrays(particles, Val(1))  # cool

        # Draw geometry
        InitialFieldsParticles!(phases, particles.coords..., particles.index, scales)
       
        phase_ratios = PhaseRatios(backend, Nphases, values(Nc)) 
    end
 
    # Compute phase fraction on cell vertices
    phase_ratios_vertex!(phase_ratios, particles, values(verts), phases) 
    phase_ratios_center!(phase_ratios, particles, values(cents_in), phases)
    begin 
        if USE_GPU
            phv_GPU.data .= CuArray(phase_ratios.vertex).data
            phc_GPU.data .= CuArray(phase_ratios.center).data

        else
            phv_GPU.data .=        (phase_ratios.vertex).data
            phc_GPU.data .=        (phase_ratios.center).data
        end
    end

    SumRatios = @zeros(Nc.x+1, Nc.y+1, Nc.z+1)
    @parallel SumRatios!( SumRatios, phv_GPU,  Nphases )
    @show minimum(SumRatios), maximum(SumRatios) 

    #######################################
    tot_iter_DYREL = 0; tot_iter_PCG = 0; tot_iter_GCR = 0; tot_iter_PH = 0; iterPH = 0
    logErrVx = zeros(numerics.niterPH)
    logErrVy = zeros(numerics.niterPH)
    logErrVz = zeros(numerics.niterPH)
    logErrP  = zeros(numerics.niterPH)
    t = 0.; tsolve = 0.

    @parallel InitialFields!(V, verts, physics)
    @parallel ComputeStrainRates!( ∇V, ε̇, V, Δ )
    ApplyBCs_ε̇!(ε̇)
    
    if p2g==:center 
        @parallel ViscosityCenter!( rheo, physics, ε̇, 1., phc_GPU,  Nphases, interp_p2g)
        @parallel (1:size(rheo.ηve_c,2), 1:size(rheo.ηve_c,3)) bc_x!(rheo.ηve_c)
        @parallel (1:size(rheo.ηve_c,1), 1:size(rheo.ηve_c,3)) bc_y!(rheo.ηve_c)
        @parallel (1:size(rheo.ηve_c,1), 1:size(rheo.ηve_c,2)) bc_z!(rheo.ηve_c)
        @parallel InterpViscosityCenter!(rheo, rheo.ηve_c, interp_g2g)
        @parallel RogerCenter!(b, rheo, physics)
    elseif p2g==:vertex
        @parallel ViscosityVertex!( rheo, physics, ε̇, 1., phv_GPU,  Nphases, interp_p2g)
        @parallel InterpViscosityVertex!(rheo, rheo.ηve_v, interp_g2g)
        @parallel RogerVertex!(b, rheo, physics)
    end

    @show minimum(rheo.ηve_c)*scales.η, maximum(rheo.ηve_c)*scales.η 
    @show minimum(rheo.ηve_v)*scales.η, maximum(rheo.ηve_v)*scales.η 
    @show minimum(rheo.ηve_xy)*scales.η, maximum(rheo.ηve_xy)*scales.η 
    @show minimum(rheo.ηve_xz)*scales.η, maximum(rheo.ηve_xz)*scales.η 
    @show minimum(rheo.ηve_yz)*scales.η, maximum(rheo.ηve_yz)*scales.η 

    @show minimum(rheo.ρc)*scales.ρ, maximum(rheo.ρc)*scales.ρ
    @show minimum(b.y), maximum(b.y) 
    @show L 

    # Initial pressure is set lithostic gradient
    P[1:end-0,2:end-1,1:end-0] .= .-reverse(cumsum(reverse((b.y[1:end-0,1:end-1,1:end-0]).* Δ.y, dims=2), dims=2), dims=2)
    @show minimum(P[1:end-0,2:end-1,1:end-0])*scales.τ maximum(P[1:end-0,2:end-1,1:end-0])*scales.τ

    #######################################

    for it=it0:numerics.nt

        t += numerics.Δt
        
        @printf(">>>>>>>>>> Time step %05d --- t = %2.2e <<<<<<<<<<<\n", it, t)

        λmin     = 1.
        @parallel InitialFields!(V, verts, physics)
        @parallel ComputeStrainRates!( ∇V, ε̇, V, Δ )
        if p2g==:center 
            @parallel ViscosityCenter!( rheo, physics, ε̇, 1., phc_GPU,  Nphases, interp_p2g)
            @parallel (1:size(rheo.ηve_c,2), 1:size(rheo.ηve_c,3)) bc_x!(rheo.ηve_c)
            @parallel (1:size(rheo.ηve_c,1), 1:size(rheo.ηve_c,3)) bc_y!(rheo.ηve_c)
            @parallel (1:size(rheo.ηve_c,1), 1:size(rheo.ηve_c,2)) bc_z!(rheo.ηve_c)
            @parallel InterpViscosityCenter!(rheo, rheo.ηve_c, interp_g2g)
            @parallel RogerCenter!(b, rheo, physics)
        elseif p2g==:vertex
            @parallel ViscosityVertex!( rheo, physics, ε̇, 1., phv_GPU,  Nphases, interp_p2g)
            @parallel InterpViscosityVertex!(rheo, rheo.ηve_v, interp_g2g)
            @parallel RogerVertex!(b, rheo, physics)
        end

        errVx0, errVy0, errVz0, errPt0, errη0 = 0., 0., 0., 0., 0.
        errVxPT0, errVyPT0, errVzPT0 = 0., 0., 0., 0., 0.
        tot_iter_DYREL = 0; tot_iter_PCG = 0; tot_iter_GCR = 0; tot_iter_PH = 0; iterPH = 0

        tsolve = @elapsed @timeit to "Powell-Hestenes solver" for iterPH=1:numerics.niterPH
            tot_iter_PH += 1

            # Stokes residual
            ApplyBCs!(V)
            @parallel ComputeStrainRates!( ∇V, ε̇, V, Δ )
            @parallel ComputeStress!( P, τ, ε̇, rheo, physics )
            @parallel ComputeResidualsσyyBC!( R, RP, τ, P, ∇V, b, D, surf.σyyBC, Δ )

            # Check residual 
            errVx = norm(R.x)/sqrt(length(R.x)); if errVx == 0 errVx += 1e-13 end
            errVy = norm(R.y)/sqrt(length(R.y)); if errVy == 0 errVy += 1e-13 end
            errVz = norm(R.z)/sqrt(length(R.z)); if errVz == 0 errVz += 1e-13 end
            errPt = norm(RP )/sqrt(length(RP )); if errPt == 0 errPt += 1e-13 end   
            errη  = norm(rheo.ηve_v -  rheo.ηve_true )/sqrt(length(rheo.ηve_v)); if errη == 0 errη += 1e-13 end
            err = max(errVx, errVy, errVz, errPt, errη)
            if iterPH==1 errVx0, errVy0, errVz0, errPt0, errη0 =  err, err, err, err, err end
            # if iterPH==1 errVx0, errVy0, errVz0, errPt0 =  errVx, errVy, errVz, errVx end

            @printf(">>>>>>>>>> PH iter %05d - %s - iter/nx = %03d - iter = %03d\n", iterPH, string(numerics.solver), (tot_iter_DYREL+tot_iter_PCG+tot_iter_GCR)/Nc.x, (tot_iter_DYREL+tot_iter_PCG+tot_iter_GCR))
            @printf("Rx = %2.4e --- Ry = %2.4e --- Rz = %2.4e --- Rp = %2.4e --- Rη = %2.4e\n", errVx,        errVy,        errVz,        errPt,        errη)
            @printf("Rx = %2.4e --- Ry = %2.4e --- Rz = %2.4e --- Rp = %2.4e --- Rη = %2.4e\n", errVx/errVx0, errVy/errVy0, errVz/errVz0, errPt/errPt0, errη/errη0)
            ( max( min(errVx/errVx0, errVx), min(errVy/errVy0, errVy),  also_z*min(errVz/errVz0, errVz)) < numerics.ϵ && min(errPt/errPt0, errPt) < numerics.ϵ ) && break
            # ( max( errVx/errVx0, errVy/errVy0,  0*also_z*errVz/errVz0) < numerics.ϵ && errPt/errPt0 < numerics.ϵ ) && break
            logErrVx[iterPH] = errVx; logErrVy[iterPH] = errVy; logErrVz[iterPH] = errVz; logErrP[iterPH] = errPt

            # Set penalty automatically
            numerics.γauto ? numerics.γ =  mean(rheo.ηve_c)*numerics.γfact : nothing
            
            @parallel DiagMechanics3DσyyBC!( D, D_SC, rheo, Δ, numerics.γ, numerics.PC )
            @parallel GershgorinMechanics3D!( G, D_SC, rheo, Δ, numerics.γ )
            SetPseudoTimeStep!(h, G, numerics)
            c     = (x=2.0*sqrt(λmin), y=2.0*sqrt(λmin), z=2.0*sqrt(λmin))
            @show minimum(rheo.ηve_v), maximum(rheo.ηve_v) 

            # Schur complement residual
            ApplyBCs!(V)
            @parallel ComputeStrainRates!( ∇V, ε̇, V, Δ )
            @parallel ComputeStress_SchurComplement!( τ, ε̇, ∇V, rheo, numerics.γ )
            @parallel ComputeResidualsσyyBC!( R, RP, τ, P, ∇V, b, D, surf.σyyBC, Δ )

            numerics.ϵauto ? numerics.ϵrel = max(errVx/errVx0, errVy/errVy0, errVz/errVz0)*numerics.ϵfact : nothing 
                    
            @timeit to "DYREL" for iterPT=1:numerics.niterPT
                
                tot_iter_DYREL += 1
                @parallel SaveOldResidual_v1!( R_it, R )

                # Schur complement residual
                ApplyBCs!(V)
                @parallel ComputeStrainRates!( ∇V, ε̇, V, Δ )
                @parallel ComputeStress_SchurComplement!( τ, ε̇, ∇V, rheo, numerics.γ )
                @parallel ComputeResidualsσyyBC!( R, RP, τ, P, ∇V, b, D, surf.σyyBC, Δ )

                # Updates
                @parallel UpdateRates_v2!( ∂V∂τ, R, D_SC, h, c )
                @parallel UpdateV_v1!( V, ∂V∂τ, h )

                if iterPT<=2 || mod(iterPT, numerics.nout)==0
                    errVxPT = norm(R.x)/sqrt(length(R.x)); if errVxPT == 0 errVxPT += 1e-13 end
                    errVyPT = norm(R.y)/sqrt(length(R.y)); if errVyPT == 0 errVyPT += 1e-13 end
                    errVzPT = norm(R.z)/sqrt(length(R.z)); if errVzPT == 0 errVzPT += 1e-13 end
                    errPT = max(errVxPT, errVyPT, errVzPT)
                    if iterPT==1 errVxPT0, errVyPT0, errVzPT0 =  errPT, errPT, errPT end
                    if numerics.noisy
                        @printf(">>>>> DYREL iter %05d \n", iterPT )
                        @printf("Rx abs = %2.4e --- Ry abs = %2.4e --- Rz abs = %2.4e\n", errVxPT, errVyPT, errVzPT)
                        @printf("Rx rel = %2.4e --- Ry rel = %2.4e --- Rz rel = %2.4e\n", errVxPT/errVxPT0, errVyPT/errVyPT0, errVzPT/errVzPT0)
                    end
                    ( max( errVxPT/errVxPT0, errVyPT/errVyPT0, also_z*errVzPT/errVzPT0) < numerics.ϵrel ) && break
                    isnan(errVxPT) ? error("NaNs") : nothing
                    # Update dt
                    @parallel DiagMechanics3DσyyBC!( D, D_SC, rheo, Δ, numerics.γ, numerics.PC )
                    @parallel GershgorinMechanics3D!( G, D_SC, rheo, Δ, numerics.γ )
                    SetPseudoTimeStep!(h, G, numerics)
                    c = SetDamping(h, ∂V∂τ, R, R_it, D_SC, In, ThreeD, numerics)
                end
            end

            # Pressure update
            @parallel UpdatePressure_SchurComplement!( P, bP, ∇V, rheo, numerics.γ )
        end
        P[2:end-1,2:end-1,2:end-1] .= P[2:end-1,2:end-1,2:end-1] .- mean(P[2:end-1,2:end-1,2:end-1])

        if ( Visu )
            p1 = pt.plot() 
            p1 = pt.plot!(1:tot_iter_PH, log10.(logErrVx[1:tot_iter_PH]), label="x")
            p1 = pt.plot!(1:tot_iter_PH, log10.(logErrVy[1:tot_iter_PH]), label="y")
            p1 = pt.plot!(1:tot_iter_PH, log10.(logErrVz[1:tot_iter_PH]), label="z")
            p1 = pt.plot!(1:tot_iter_PH, log10.(logErrP[1:tot_iter_PH]), label="p")
            p1 = pt.heatmap(verts.x.*scales.L, cents_in.y.*scales.L,  V.x[1:end-0,2:end-1,Int64(ceil(size(τ.xx,3)/2))]'.*scales.V, aspect_ratio=1, xlims=(verts.x[1].*scales.L, verts.x[end].*scales.L), ylims=(verts.y[1].*scales.L, verts.y[end].*scales.L))
            p2 = pt.heatmap(cents_in.x.*scales.L, verts.y.*scales.L,  V.y[2:end-1,1:end-0,Int64(ceil(size(τ.xx,3)/2))]'.*scales.V, aspect_ratio=1, xlims=(verts.x[1].*scales.L, verts.x[end].*scales.L), ylims=(verts.y[1].*scales.L, verts.y[end].*scales.L))
            p3 = pt.heatmap(cents.x.*scales.L,   cents.y.*scales.L, log10.(rheo.ηve_c[:,:,Int64(ceil(size(τ.xx,3)/2))]'.*scales.η),     aspect_ratio=1, xlims=(verts.x[1].*scales.L, verts.x[end].*scales.L), ylims=(verts.y[1].*scales.L, verts.y[end].*scales.L))
            p4 = pt.heatmap(cents.x.*scales.L,   cents.y.*scales.L, P[:,:,Int64(ceil(size(τ.xx,3)/2))]'.*scales.τ,     aspect_ratio=1, xlims=(verts.x[1].*scales.L, verts.x[end].*scales.L), ylims=(verts.y[1].*scales.L, verts.y[end].*scales.L), clims=(-3,3))
            display(pt.plot(p1, p2, p3, p4, layout=(2,2)))
        end

        if (mod(it, SaveStep)==0 || it==1)
            if SaveGrid
                # Save grid data
                Vxc = 0.5*(Array(V.x[1:end-1,2:end-1,2:end-1] .+ V.x[2:end-0,2:end-1,2:end-1]))
                Vyc = 0.5*(Array(V.y[2:end-1,1:end-1,2:end-1] .+ V.y[2:end-1,2:end-0,2:end-1]))
                Vzc = 0.5*(Array(V.z[2:end-1,2:end-1,1:end-1] .+ V.z[2:end-1,2:end-1,2:end-0]))
                ε̇II = 1.0/8.0*(Array(ε̇.II[1:end-1,1:end-1,1:end-1]) .+ Array(ε̇.II[2:end-0,1:end-1,1:end-1]) .+ Array(ε̇.II[1:end-1,2:end-0,1:end-1]) .+ Array(ε̇.II[2:end-0,2:end-0,1:end-1]) +
                               Array(ε̇.II[1:end-1,1:end-1,2:end-0]) .+ Array(ε̇.II[2:end-0,1:end-1,2:end-0]) .+ Array(ε̇.II[1:end-1,2:end-0,2:end-0]) .+ Array(ε̇.II[2:end-0,2:end-0,2:end-0]) )
                ρ   = 1.0/8.0*(Array(rheo.ρv[1:end-1,1:end-1,1:end-1]) .+ Array(rheo.ρv[2:end-0,1:end-1,1:end-1]) .+ Array(rheo.ρv[1:end-1,2:end-0,1:end-1]) .+ Array(rheo.ρv[2:end-0,2:end-0,1:end-1]) +
                                Array(rheo.ρv[1:end-1,1:end-1,2:end-0]) .+ Array(rheo.ρv[2:end-0,1:end-1,2:end-0]) .+ Array(rheo.ρv[1:end-1,2:end-0,2:end-0]) .+ Array(rheo.ρv[2:end-0,2:end-0,2:end-0]) )
                filename = @sprintf( "./Stokes3D%05d", it )
                @info "writing "*filename*" to disk"
                vtkfile                    = vtk_grid(filename, Float32.(Array(cents_in.x.*scales.L)), Float32.(Array(cents_in.y.*scales.L)), Float32.(Array(cents_in.z.*scales.L)))
                vtkfile["Pressure"]        = Float32.(Array(P[2:end-1,2:end-1,2:end-1].*scales.τ))
                vtkfile["log10 Viscosity"] = Float32.(Array(log10.(rheo.ηve_c[2:end-1,2:end-1,2:end-1].*scales.η)))
                vtkfile["log10 StrainRate"]= Float32.(Array(log10.(ε̇II./scales.t)))
                vtkfile["Velocity"]        = (Float32.(Array(Vxc.*scales.V)), Float32.(Array(Vyc.*scales.V)), Float32.(Array(Vzc.*scales.V)))
                vtkfile["Density"]         = Float32.(Array(ρ.*scales.ρ))
                vtkfile["TimeValue"]       = t*scales.t/My
                outfiles                   = vtk_save(vtkfile)
            end
            if SavePart
                # Save particles
                Npart = sum(particles.index.data)
                p = particles.coords
                ppx, ppy, ppz = p
                index = particles.index.data
                pxv = Float32.(Array(ppx.data[index][:]))
                pyv = Float32.(Array(ppy.data[index][:]))
                pzv = Float32.(Array(ppz.data[index][:]))
                phase_scatter = Float32.(Array(phases.data[index][:]))
                cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i, )) for i = 1:Npart]
                filename = @sprintf( "./ParticlesStokes3D%05d", it )
                @info "writing "*filename*" to disk"
                vtk_grid(filename, pxv, pyv, pzv, cells) do vtk
                    vtk["phase", VTKPointData()] = phase_scatter
                end
            end
        end
        if SaveCheck && (mod(it, SaveCheckStep)==0)
            # Create new one
            filename = @sprintf( "./Checkpoint%05d.jld2", it)
            @info "writing "*filename*" to disk"
            jldsave(filename; 
            particles     = Array( Float32, particles), 
            phases        = Array( Float32, phases), 
            phase_ratios  = Array( Float32, phase_ratios), 
            particle_args = Array.( Float32, particle_args),
            Vx            = Float32.(Array(V.x)), 
            Vy            = Float32.(Array(V.y)),
            Vz            = Float32.(Array(V.z)),
            P             = Float32.(Array(P)),
            xlims, ylims, zlims, t)
            # Remove previous one
            filename = @sprintf( "./Checkpoint%05d.jld2", it-SaveCheckStep)
            @show isfile(filename)
            if isfile(filename)
                rm(filename) 
            end
        end
        ndof = prod(size(V.x)) + prod(size(V.y)) + prod(size(V.z)) + prod(size(P))
        @info "ndof       = $(ndof)"
        @info "Iter/nx    = $((tot_iter_DYREL)/Nc.x)"
        @info "Mean(eta)  = $(mean(rheo.ηve_c))"
        @info "γ          = $(numerics.γ)"
        @info "Tot. iter. = $(tot_iter_DYREL)"
        show(to); @printf("\n\n")
    end
    return tot_iter_PH, tot_iter_DYREL, tot_iter_DYREL/Nc.x, tsolve
end

function main_call()
    N = [2]
    itPH    = zeros(length(N))
    itDYREL = zeros(length(N))
    it_nx   = zeros(length(N))
    wtime   = zeros(length(N)) 
    for i in eachindex(N)
        itPH[i], itDYREL[i], it_nx[i], wtime[i] = Stokes3D_PHDR(; n=N[i], ALE=false, restart=false, restart_step=500, end_step=1)
    end
    jldsave("Scaling_Balls_1m6_FDM.jld2"; itPH, itDYREL, it_nx, wtime)
    @show itPH
    @show wtime
end

main_call()
