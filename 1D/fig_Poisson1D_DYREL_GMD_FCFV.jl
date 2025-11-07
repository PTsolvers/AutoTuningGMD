using CairoMakie, LinearAlgebra, MathTeXEngine
Makie.update_theme!( fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

# Poisson equation
# q = -k*dTdx
# 0 = dqdx - b 

"""
FCFV Residual of Poisson problem in 1D (Dirichlet BC's)
"""
function Residual_FCFV_1D!(f,H,Hel,kel,qxel,αe,Ωe,Γ,τ,nx1,nx2)
    f     .= 0.0
    # per elemem
    Hel   .= 0.
    qxel  .= 0.
 
    # W
    Hel  .+= Γ*τ*H[1:end-1]/αe
    qxel .-= kel.*Γ.*nx1.*H[1:end-1]/Ωe
    # E
    Hel  .+= Γ*τ*H[2:end-0]/αe
    qxel .-= kel.*Γ.*nx2.*H[2:end-0]/Ωe
    # Residual at each face
    f   .-= (nx1.*qxel[1:end-1] .+ τ*(Hel[1:end-1] .- H[2:end-1])) # from west element
    f   .-= (nx2.*qxel[2:end-0] .+ τ*(Hel[2:end-0] .- H[2:end-1])) # from east element
    return nothing
end

function main_FCFV(;ncx, variable_coefficient, preconditioning, tol, c_fact, nout)

    # Definition of computation domain
    L    = 1.0
    Δx   = L/ncx
    xc   = LinRange(-L/2+Δx/2, L/2-Δx/2, ncx)
    xv   = LinRange(-L/2, L/2, ncx+1)

    # Boundaries conditions (Dirichlet)
    uBC  = (W=1.0, E=2.0)
    
    # Allocations
    b    = zeros(ncx)
    k    =  ones(ncx+0)

    uh   = zeros(ncx+1)
    R    = zeros(ncx-1)
    R0   = zeros(ncx-1)
    ∂u∂τ = zeros(ncx-1)

    # FCFV
    qxel      = ones(ncx+0)  
    τ         = 1.0                  # face stabilisation parameter
    Γ         = 1.0                  # face length is 1.0 in 1D !!!!!!!!!!!
    nx1       = -1.0                 # face normal to W
    nx2       = 1.0                  # face normal to E
    qxel      = zeros(ncx)           # flux on elements
    uel       = zeros(ncx)           # solution on elements
    Ωe        = Δx                   # element volume = spacing in 1D
    αe        = 2*Γ*τ          # in 1D, 2 faces will contribute    
    uh[1]      = uBC.W         # set BC values (won't be updated)
    uh[end]    = uBC.E         # set BC values (won't be updated)

    # Coefficient
    if variable_coefficient
        k    = @.  1 + 100 * ( (1 + exp(-200 * (xc + 0.1)))^(-1)  - (1 + exp(-200 * (xc - 0.1)))^(-1) )    
    end

    # Initiate
    ∂u∂τ .= 0.0

    # Diagonal preconditional 
    alpha_e, Omega_e, tau, Gamma = αe, Ωe, τ,  Γ
    kW, kE = k[1:end-1],  k[2:end-0]

    if preconditioning
        # PC    = (k[1:end-1] +  k[2:end-0])/Δx^2
        PC   = @. (-Gamma .* alpha_e .* nx1 .* nx2 .* (kE + kW) - 2 * Omega_e .* tau .* (Gamma .* tau - alpha_e)) ./ (Omega_e .* alpha_e)
    else
        PC    = ones(size(R))
    end

    # Maximum eigenvalue: Gershgorin circle theorem
    # Achtung: preconditioned system 
    λmax = @. (abs((Gamma .* alpha_e .* nx1 .* nx2 .* (kE + kW) + 2 * Omega_e .* tau .* (Gamma .* tau - alpha_e)) ./ (Omega_e .* alpha_e)) + abs(Gamma .* (Omega_e .* tau .^ 2 + alpha_e .* kE .* nx1 .^ 2) ./ (Omega_e .* alpha_e)) + abs(Gamma .* (Omega_e .* tau .^ 2 + alpha_e .* kW .* nx2 .^ 2) ./ (Omega_e .* alpha_e)) ) ./ PC

    # Pseudo time step is proportional to λmax (e.g., Oakley, 1995)
    Δτ    = 2 / sqrt(maximum(λmax)) * 0.99

    # Damping coefficient is proportional to λmin (e.g., Oakley, 1995)
    λmin  = 0.          # wild guess  
    c     = 2*sqrt.(λmin)*0.5
    β     = 2 .* Δτ^2 ./ (2 .+ c.*Δτ)
    α     = (2 .- c.*Δτ) ./ (2 .+ c.*Δτ)  

    # Iteration loop 
    iter, err, err0 = 0,  1.0, 1.0
    err_vec, iter_vec = [], []

    while err > tol #&& iter<2

        # Increment iteration count
        iter  += 1

        # Previous residual
        R0    .= R  

        Residual_FCFV_1D!(R, uh, uel, k, qxel, αe, Ωe, Γ, τ, nx1, nx2)
       
        # Pseudo-rate update
        ∂u∂τ        .= R./PC .+ β*∂u∂τ      # Achtung: preconditioned system

        # Solution update
        uh[2:end-1] .+= α*∂u∂τ

        if mod(iter, nout)==0 || iter==1
            # Residual check 
            (iter == 1) && (err0 = norm(R))
            err   = norm(R)/err0
            push!(err_vec, err)
            push!(iter_vec, iter)
            isnan(err) && error("nan $(iter)")
            # Rayleigh quotient (e.g., Joldes et al., 2011)
            λmin  = abs.((sum(Δτ*∂u∂τ.*(R .- R0)./ PC))) / sum( (Δτ*∂u∂τ) .* (Δτ*∂u∂τ) )
            # Dynamic evaluation of PT iteration parameters
            c = 2*sqrt.(λmin)*c_fact
            α = 2 .* Δτ^2 ./ (2 .+ c.*Δτ)
            β = (2 .- c.*Δτ) ./ (2 .+ c.*Δτ)
        end
    end

    return (xc=xc, xv=xv, ue=uel, uh=uh, k=k, b=b), err_vec, iter_vec

end 

# let
#     ncx    = 400
#     c_fact = 0.7

#     sol, err_vec, iter_vec = main_FCFV(; ncx=ncx, variable_coefficient=false, preconditioning=true, tol=1e-10, c_fact=c_fact, nout=100)
#     display(maximum(iter_vec))

# end

function main(;ncx, variable_coefficient, preconditioning, tol, c_fact, nout)

    # Definition of computation domain
    L    = 1.0
    Δx   = L/ncx
    xc   = LinRange(-L/2+Δx/2, L/2-Δx/2, ncx)
    xv   = LinRange(-L/2, L/2, ncx+1)

    # Boundaries conditions (Dirichlet)
    uBC  = (W=1.0, E=2.0)
    
    # Allocations
    b    = zeros(ncx)
    u    = zeros(ncx+2)
    q    = zeros(ncx+1)
    k    =  ones(ncx+1)
    R    = zeros(ncx+0)
    R0   = zeros(ncx+0)
    ∂u∂τ = zeros(ncx+0)

    # Coefficient
    if variable_coefficient
        # k    = @.  1 + 100 / (1 + exp(-200 * (xv + 0.1)))  - 100 / (1 + exp(-200 * (xv - 0.1)))    
        k    = @.  1 + 100 * ( (1 + exp(-200 * (xv + 0.1)))^(-1)  - (1 + exp(-200 * (xv - 0.1)))^(-1) )    

    end

    # Initiate
    u    .= 0.0
    ∂u∂τ .= 0.0

    # Diagonal preconditional 
    if preconditioning
        PC    = (k[1:end-1] +  k[2:end-0])/Δx^2
    else
        PC    = ones(size(r))
    end

    # Maximum eigenvalue: Gershgorin circle theorem
    # Achtung: preconditioned system 
    λmax  = ( (k[1:end-1] +  k[2:end-0])/Δx^2 + k[1:end-1]/Δx^2 + k[2:end-0]/Δx^2 ) ./ PC
    
    # Pseudo time step is proportional to λmax (e.g., Oakley, 1995)
    Δτ    = 2 / sqrt(maximum(λmax)) * 0.99

    # Damping coefficient is proportional to λmin (e.g., Oakley, 1995)
    λmin  = 0.          # wild guess  
    c     = 2*sqrt.(λmin)*0.5
    β     = 2 .* Δτ^2 ./ (2 .+ c.*Δτ)
    α     = (2 .- c.*Δτ) ./ (2 .+ c.*Δτ)  

    # Iteration loop 
    iter, err, err0 = 0,  1.0, 1.0
    err_vec, iter_vec = [], []

    while err > tol

        # Increment iteration count
        iter  += 1

        # Previous residual
        R0    .= R  

        # Set boundary conditions
        u[1]   = 2*uBC.W - u[2]
        u[end] = 2*uBC.E - u[end-1]

        # Flux
        q     .= -k.*diff(u)/Δx

        # Residual
        R     .= .- diff(q)/Δx .- b

        # Pseudo-rate update
        ∂u∂τ        .= R./PC .+ β*∂u∂τ      # Achtung: preconditioned system

        # Solution update
        u[2:end-1] .+= α*∂u∂τ

        if mod(iter, nout)==0 || iter==1
            # Residual check 
            (iter == 1) && (err0 = norm(R))
            err   = norm(R)/err0
            push!(err_vec, err)
            push!(iter_vec, iter)
            # Rayleigh quotient (e.g., Joldes et al., 2011)
            λmin  = abs.((sum(Δτ*∂u∂τ.*(R .- R0)./ PC))) / sum( (Δτ*∂u∂τ) .* (Δτ*∂u∂τ) )
            # Dynamic evaluation of PT iteration parameters
            c = 2*sqrt.(λmin)*c_fact
            α = 2 .* Δτ^2 ./ (2 .+ c.*Δτ)
            β = (2 .- c.*Δτ) ./ (2 .+ c.*Δτ)
        end
    end

    return (xc=xc, xv=xv, u=u, k=k, b=b), err_vec, iter_vec

end 

#-------------------------------------------------------#

let
    ncx    = 400
    c_fact = 0.7

    # # Case 1: constant coefficient  
    # sol, err_vec, iter_vec = main(; ncx=ncx, variable_coefficient=false, preconditioning=true, tol=1e-10, c_fact=c_fact, nout=100)
    # display(maximum(iter_vec))

    # Case 2: variable coefficient  
    sol, err_vec, iter_vec = main(; ncx=ncx, variable_coefficient=true, preconditioning=true, tol=1e-10, c_fact=c_fact, nout=100)
    display(maximum(iter_vec))

    sol_FCFV, err_vec, iter_vec = main_FCFV(; ncx=ncx, variable_coefficient=true, preconditioning=true, tol=1e-10, c_fact=c_fact, nout=100)
    display(maximum(iter_vec))

    # Visualization
    fig = Figure(size = (500, 500), fontsize=18)
    ax1 = Axis(fig[1, 1], title=L"$$ (a) Solution field", xlabel=L"$x$", ylabel=L"$u$")
    scatter!(ax1, sol.xc[1:10:end], sol.u[2:10:end-1], label=L"$$FDM", color=:blue)
    lines!(ax1, sol_FCFV.xc[1:10:end], sol_FCFV.ue[1:10:end], color=:black, label=L"$$FCFV")
    # lines!(ax1, sol_FCFV.xv[1:10:end], sol_FCFV.uh[1:10:end], label=L"u", color=:orange)
    axislegend(ax1, position = :lt, framevisible=false, labelsize=14)  # right-top


    ax2 = Axis(fig[1, 2], title=L"$$ (b) Coefficient", xlabel=L"$x$", ylabel=L"$k$")
    scatter!(ax2, sol.xv, sol.k, label=L"$$FDM", color=:blue)
    lines!(ax2, sol.xc, sol_FCFV.k, label=L"$$FCFV", color=:black)
    # axislegend(ax2, position = :lt, framevisible=false, labelsize=14)  # right-top


    #-------------------------------------------------------#

    ax3 = Axis(fig[2, 1], title=L"$$ (c) Convergence history", xlabel=L"$10^3$ iterations", ylabel=L"$\log_{10}$ relative residual")

    # Variable coefficient - preconditioning 
    ncx = 200
    sol, err_vec, iter_vec = main(; ncx=ncx, variable_coefficient=true, preconditioning=true, tol=1e-10, c_fact=c_fact, nout=100)
    scatter!(ax3, iter_vec./1000, log10.(err_vec), label=L"$$ncx = 200", color=:blue)

    sol_FCFV, err_vec, iter_vec = main_FCFV(; ncx=ncx, variable_coefficient=true, preconditioning=true, tol=1e-10, c_fact=c_fact, nout=100)
    lines!(ax3, iter_vec./1000, log10.(err_vec), color=:black)

    ncx = 400
    sol, err_vec, iter_vec = main(; ncx=ncx, variable_coefficient=true, preconditioning=true, tol=1e-10, c_fact=c_fact, nout=100)
    scatter!(ax3, iter_vec./1000, log10.(err_vec), label=L"$$ncx = 400", color=:green)

    sol_FCFV, err_vec, iter_vec = main_FCFV(; ncx=ncx, variable_coefficient=true, preconditioning=true, tol=1e-10, c_fact=c_fact, nout=100)
    lines!(ax3, iter_vec./1000, log10.(err_vec), color=:black)

    ncx = 800
    sol, err_vec, iter_vec = main(; ncx=ncx, variable_coefficient=true, preconditioning=true, tol=1e-10, c_fact=c_fact, nout=100)
    scatter!(ax3, iter_vec./1000, log10.(err_vec), label=L"$$ncx = 800", color=:orange)

    sol_FCFV, err_vec, iter_vec = main_FCFV(; ncx=ncx, variable_coefficient=true, preconditioning=true, tol=1e-10, c_fact=c_fact, nout=100)
    lines!(ax3, iter_vec./1000, log10.(err_vec), color=:black)

    axislegend(ax3, position = :rt, framevisible=false, labelsize=14)  # right-top

    # #-------------------------------------------------------#

    ax4 = Axis(fig[2, 2], title=L"$$ (d) Scaling", xlabel=L"$\log_{10}$ resolution", ylabel=L"$\log_{10}$ iterations")

    ncx = 100:100:2000
    iter_count = zeros(length(ncx)) 
    iter_count_FCFV = zeros(length(ncx)) 

    for n in eachindex(ncx)
        sol, err_vec, iter_vec = main(; ncx=ncx[n], variable_coefficient=true, preconditioning=true, tol=1e-10, c_fact=c_fact, nout=100)
        iter_count[n] = maximum(iter_vec)
        sol, err_vec, iter_vec = main_FCFV(; ncx=ncx[n], variable_coefficient=true, preconditioning=true, tol=1e-10, c_fact=c_fact, nout=100)
        iter_count_FCFV[n] = maximum(iter_vec)
    end

    scatter!(ax4, log10.(ncx), log10.(iter_count), color=:blue, label=L"$$FDM")
    lines!(ax4, log10.(ncx), log10.(ncx).+1, linestyle=:dash, color=:black, label=L"$$O(1)")
    lines!(ax4, log10.(ncx), log10.(iter_count), color=:black, label=L"$$FCFV")
    # lines!(ax4, log10.(ncx), log10.(ncx.^2), label=L"$$ncx = 800", linestyle=:dashdot)
    ylims!(2.9,5.2)
    axislegend(ax4, position = :lt, framevisible=false, labelsize=14, nbanks=2)  # right-top

    display(fig)
    save("./img/BasicDyRel.png", fig, px_per_unit = 4) 
end