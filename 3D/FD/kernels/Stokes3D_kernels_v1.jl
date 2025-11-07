
@parallel function compute_maxloc!(Musτ2::Data.Array, Musτ::Data.Array)
    @inn(Musτ2) = @maxloc(Musτ)
    return
end

@parallel function compute_minloc!(Musτ2::Data.Array, Musτ::Data.Array)
    @inn(Musτ2) = @minloc(Musτ)
    return
end

@views function ApplyBCs!(V)
    @parallel (1:size(V.y,2), 1:size(V.y,3)) bc_x!(V.y)
    @parallel (1:size(V.z,2), 1:size(V.z,3)) bc_x!(V.z)
    @parallel (1:size(V.x,1), 1:size(V.x,3)) bc_y!(V.x)
    @parallel (1:size(V.z,1), 1:size(V.z,3)) bc_y!(V.z)
    @parallel (1:size(V.x,1), 1:size(V.x,2)) bc_z!(V.x)
    @parallel (1:size(V.y,1), 1:size(V.y,2)) bc_z!(V.y)
    return nothing
end

@views function ApplyBCs_Subduction!(V)
    @parallel (1:size(V.y,2), 1:size(V.y,3)) bc_x!(V.y)
    @parallel (1:size(V.z,2), 1:size(V.z,3)) bc_x!(V.z)
    @parallel (1:size(V.x,1), 1:size(V.x,3)) bc_y!(V.x)
    @parallel (1:size(V.z,1), 1:size(V.z,3)) bc_y!(V.z)
    @parallel (1:size(V.x,1), 1:size(V.x,2)) bc_z!(V.x)
    @parallel (1:size(V.y,1), 1:size(V.y,2)) bc_z!(V.y)
    return nothing
end

@views function ApplyBCs_ε̇!(ε̇)
    @parallel (1:size(ε̇.xx,2), 1:size(ε̇.xx,3)) bc_x!(ε̇.xx)
    @parallel (1:size(ε̇.xx,1), 1:size(ε̇.xx,3)) bc_y!(ε̇.xx)
    @parallel (1:size(ε̇.xx,1), 1:size(ε̇.xx,2)) bc_z!(ε̇.xx)
    @parallel (1:size(ε̇.yy,2), 1:size(ε̇.yy,3)) bc_x!(ε̇.yy)
    @parallel (1:size(ε̇.yy,1), 1:size(ε̇.yy,3)) bc_y!(ε̇.yy)
    @parallel (1:size(ε̇.yy,1), 1:size(ε̇.yy,2)) bc_z!(ε̇.yy)
    @parallel (1:size(ε̇.zz,2), 1:size(ε̇.zz,3)) bc_x!(ε̇.zz)
    @parallel (1:size(ε̇.zz,1), 1:size(ε̇.zz,3)) bc_y!(ε̇.zz)
    @parallel (1:size(ε̇.zz,1), 1:size(ε̇.zz,2)) bc_z!(ε̇.zz)
return nothing
end

@parallel_indices (j,k) function bc_x!(A::Data.Array)
    A[  1, j,  k] = A[    2,   j,   k]
    A[end, j,  k] = A[end-1,   j,   k]
    return
end

@parallel_indices (i,k) function bc_y!(A::Data.Array)
    A[ i,  1,  k] = A[   i,    2,   k]
    A[ i,end,  k] = A[   i,end-1,   k]
    return
end

@parallel_indices (i,j) function bc_z!(A::Data.Array)
    A[ i,  j,  1] = A[   i,   j,    2]
    A[ i,  j,end] = A[   i,   j,end-1]
    return
end

@parallel_indices (j,k) function bc_x_noslip!(A::Data.Array, val)
    A[  1, j,  k] = 2*val.W[j,k] - A[    2,   j,   k]
    A[end, j,  k] = 2*val.E[j,k] - A[end-1,   j,   k]
    return
end

@parallel_indices (i,k) function bc_y_noslip_true!(A::Data.Array)
    A[ i,  1,  k] =  - A[   i,    2,   k]
    A[ i,end,  k] =  - A[   i,end-1,   k]
    return
end

@parallel_indices (i,k) function bc_y_noslip!(A::Data.Array, val)
    A[ i,  1,  k] = 2*val.S[i,k] - A[   i,    2,   k]
    A[ i,end,  k] = 2*val.N[i,k] - A[   i,end-1,   k]
    return
end

@parallel_indices (i,j) function bc_z_noslip!(A::Data.Array, val)
    A[ i,  j,  1] = 2*val.B[j] - A[   i,   j,    2]
    A[ i,  j,end] = 2*val.F[j] - A[   i,   j,end-1]
    return
end

@parallel_indices (i,j) function bc_z_noslip_true!(A::Data.Array)
    A[ i,  j,  1] = - A[   i,   j,    2]
    A[ i,  j,end] = - A[   i,   j,end-1]
    return
end

@parallel_indices (i,j,k) function GershgorinDiagMechanics3D!( G, D, D_SC, rheo, Δ, γ, PC )
    if i<=size(D.x,1) && j<=size(D.x,2)-2 && k<=size(D.x,3)-2
        if i>1 && i<size(D.x,1) # avoid Dirichlets
            ηW = rheo.ηve_c[i  ,j+1,k+1]
            ηE = rheo.ηve_c[i+1,j+1,k+1]
            ηS = rheo.ηve_xy[i,j,k]
            ηN = rheo.ηve_xy[i,j+1,k]
            ηB = rheo.ηve_xz[i,j,k]
            ηF = rheo.ηve_xz[i,j,k+1]
            # Diagonal
            if PC
                D.x[i,j+1,k+1]    = abs(ηB ./ Δ.z .^ 2 + ηF ./ Δ.z .^ 2 + ηN ./ Δ.y .^ 2 + ηS ./ Δ.y .^ 2 + (4 / 3) * ηE ./ Δ.x .^ 2 + (4 / 3) * ηW ./ Δ.x .^ 2)
                D_SC.x[i,j+1,k+1] = abs(ηB ./ Δ.z .^ 2 + ηF ./ Δ.z .^ 2 + ηN ./ Δ.y .^ 2 + ηS ./ Δ.y .^ 2 + 2 * γ ./ Δ.x .^ 2 + (4 / 3) * ηE ./ Δ.x .^ 2 + (4 / 3) * ηW ./ Δ.x .^ 2)
            end
            # Gerhsgorin
            G.x[i,j+1,k+1]  = abs((3 * γ + 4 * ηE) ./ Δ.x .^ 2) / 3 + abs((3 * γ + 4 * ηW) ./ Δ.x .^ 2) / 3 + abs(ηN ./ Δ.y .^ 2) + abs(ηS ./ Δ.y .^ 2) + abs(ηB ./ Δ.z .^ 2) + abs(ηF ./ Δ.z .^ 2) + abs((3 * γ - 2 * ηE + 3 * ηN) ./ (Δ.x .* Δ.y)) / 3 + abs((3 * γ - 2 * ηE + 3 * ηS) ./ (Δ.x .* Δ.y)) / 3 + abs((3 * γ + 3 * ηN - 2 * ηW) ./ (Δ.x .* Δ.y)) / 3 + abs((3 * γ + 3 * ηS - 2 * ηW) ./ (Δ.x .* Δ.y)) / 3 + abs((3 * γ + 3 * ηB - 2 * ηE) ./ (Δ.x .* Δ.z)) / 3 + abs((3 * γ + 3 * ηB - 2 * ηW) ./ (Δ.x .* Δ.z)) / 3 + abs((3 * γ - 2 * ηE + 3 * ηF) ./ (Δ.x .* Δ.z)) / 3 + abs((3 * γ + 3 * ηF - 2 * ηW) ./ (Δ.x .* Δ.z)) / 3 + abs((3 * Δ.x .^ 2 .* Δ.y .^ 2 .* (ηB + ηF) + 3 * Δ.x .^ 2 .* Δ.z .^ 2 .* (ηN + ηS) + 2 * Δ.y .^ 2 .* Δ.z .^ 2 .* (3 * γ + 2 * ηE + 2 * ηW)) ./ (Δ.x .^ 2 .* Δ.y .^ 2 .* Δ.z .^ 2)) / 3
            # Apply PC
            G.x[i,j+1,k+1] /= D_SC.x[i,j+1,k+1]      
          end
    end
    if i<=size(D.y,1)-2 && j<=size(D.y,2) && k<=size(D.y,3)-2
        if j>1 && j<size(D.y,2) # avoid Dirichlets
            ηW = rheo.ηve_xy[i,  j,k]
            ηE = rheo.ηve_xy[i+1,j,k]
            ηS = rheo.ηve_c[i+1,j,  k+1]
            ηN = rheo.ηve_c[i+1,j+1,k+1]
            ηB = rheo.ηve_xz[i,j,k]
            ηF = rheo.ηve_xz[i,j,k+1]
            # Diagonal
            if PC
                D.y[i+1,j,k+1]    = abs(ηB ./ Δ.z .^ 2 + ηF ./ Δ.z .^ 2 + (4 / 3) * ηN ./ Δ.y .^ 2 + (4 / 3) * ηS ./ Δ.y .^ 2 + ηE ./ Δ.x .^ 2 + ηW ./ Δ.x .^ 2)
                D_SC.y[i+1,j,k+1] = abs(ηB ./ Δ.z .^ 2 + ηF ./ Δ.z .^ 2 + 2 * γ ./ Δ.y .^ 2 + (4 / 3) * ηN ./ Δ.y .^ 2 + (4 / 3) * ηS ./ Δ.y .^ 2 + ηE ./ Δ.x .^ 2 + ηW ./ Δ.x .^ 2)
            end
            # Gerhsgorin
            G.y[i+1,j,k+1]  = abs(ηE ./ Δ.x .^ 2) + abs(ηW ./ Δ.x .^ 2) + abs((3 * γ + 4 * ηN) ./ Δ.y .^ 2) / 3 + abs((3 * γ + 4 * ηS) ./ Δ.y .^ 2) / 3 + abs(ηB ./ Δ.z .^ 2) + abs(ηF ./ Δ.z .^ 2) + abs((3 * γ + 3 * ηE - 2 * ηN) ./ (Δ.x .* Δ.y)) / 3 + abs((3 * γ + 3 * ηE - 2 * ηS) ./ (Δ.x .* Δ.y)) / 3 + abs((3 * γ - 2 * ηN + 3 * ηW) ./ (Δ.x .* Δ.y)) / 3 + abs((3 * γ - 2 * ηS + 3 * ηW) ./ (Δ.x .* Δ.y)) / 3 + abs((3 * γ + 3 * ηB - 2 * ηN) ./ (Δ.y .* Δ.z)) / 3 + abs((3 * γ + 3 * ηB - 2 * ηS) ./ (Δ.y .* Δ.z)) / 3 + abs((3 * γ + 3 * ηF - 2 * ηN) ./ (Δ.y .* Δ.z)) / 3 + abs((3 * γ + 3 * ηF - 2 * ηS) ./ (Δ.y .* Δ.z)) / 3 + abs((3 * Δ.x .^ 2 .* Δ.y .^ 2 .* (ηB + ηF) + 2 * Δ.x .^ 2 .* Δ.z .^ 2 .* (3 * γ + 2 * ηN + 2 * ηS) + 3 * Δ.y .^ 2 .* Δ.z .^ 2 .* (ηE + ηW)) ./ (Δ.x .^ 2 .* Δ.y .^ 2 .* Δ.z .^ 2)) / 3
            # Apply PC
            G.y[i+1,j,k+1] /= D_SC.y[i+1,j,k+1]
        end
    end
    if i<=size(D.z,1)-2 && j<=size(D.z,2)-2 && k<=size(D.z,3)
        if k>1 && k<size(D.z,3) # avoid Dirichlets
            ηW = rheo.ηve_xz[i,  j,k]
            ηE = rheo.ηve_xz[i+1,j,k]
            ηS = rheo.ηve_xy[i,j,k]
            ηN = rheo.ηve_xy[i,j+1,k]
            ηB = rheo.ηve_c[i+1,j+1,k]
            ηF = rheo.ηve_c[i+1,j+1,k+1]
            # Diagonal
            if PC
                D.z[i+1,j+1,k]    = abs((4 / 3) * ηB ./ Δ.z .^ 2 + (4 / 3) * ηF ./ Δ.z .^ 2 + ηN ./ Δ.y .^ 2 + ηS ./ Δ.y .^ 2 + ηE ./ Δ.x .^ 2 + ηW ./ Δ.x .^ 2)
                D_SC.z[i+1,j+1,k] = abs(2 * γ ./ Δ.z .^ 2 + (4 / 3) * ηB ./ Δ.z .^ 2 + (4 / 3) * ηF ./ Δ.z .^ 2 + ηN ./ Δ.y .^ 2 + ηS ./ Δ.y .^ 2 + ηE ./ Δ.x .^ 2 + ηW ./ Δ.x .^ 2)
            end
            # Gerhsgorin
            G.z[i+1,j+1,k] = abs(ηE ./ Δ.x .^ 2) + abs(ηW ./ Δ.x .^ 2) + abs(ηN ./ Δ.y .^ 2) + abs(ηS ./ Δ.y .^ 2) + abs((3 * γ + 4 * ηB) ./ Δ.z .^ 2) / 3 + abs((3 * γ + 4 * ηF) ./ Δ.z .^ 2) / 3 + abs((3 * γ - 2 * ηB + 3 * ηE) ./ (Δ.x .* Δ.z)) / 3 + abs((3 * γ - 2 * ηB + 3 * ηW) ./ (Δ.x .* Δ.z)) / 3 + abs((3 * γ + 3 * ηE - 2 * ηF) ./ (Δ.x .* Δ.z)) / 3 + abs((3 * γ - 2 * ηF + 3 * ηW) ./ (Δ.x .* Δ.z)) / 3 + abs((3 * γ - 2 * ηB + 3 * ηN) ./ (Δ.y .* Δ.z)) / 3 + abs((3 * γ - 2 * ηB + 3 * ηS) ./ (Δ.y .* Δ.z)) / 3 + abs((3 * γ - 2 * ηF + 3 * ηN) ./ (Δ.y .* Δ.z)) / 3 + abs((3 * γ - 2 * ηF + 3 * ηS) ./ (Δ.y .* Δ.z)) / 3 + abs((2 * Δ.x .^ 2 .* Δ.y .^ 2 .* (3 * γ + 2 * ηB + 2 * ηF) + 3 * Δ.x .^ 2 .* Δ.z .^ 2 .* (ηN + ηS) + 3 * Δ.y .^ 2 .* Δ.z .^ 2 .* (ηE + ηW)) ./ (Δ.x .^ 2 .* Δ.y .^ 2 .* Δ.z .^ 2)) / 3
            # Apply PC
            G.z[i+1,j+1,k] /= D_SC.z[i+1,j+1,k]
        end
    end
return nothing
end

@parallel_indices (i,j,k) function DiagMechanics3D!( D, D_SC, rheo, Δ, γ, PC )
    if i<=size(D.x,1) && j<=size(D.x,2)-2 && k<=size(D.x,3)-2
        if i>1 && i<size(D.x,1) # avoid Dirichlets
            ηW = rheo.ηve_c[i  ,j+1,k+1]
            ηE = rheo.ηve_c[i+1,j+1,k+1]
            ηS = rheo.ηve_xy[i,j,k]
            ηN = rheo.ηve_xy[i,j+1,k]
            ηB = rheo.ηve_xz[i,j,k]
            ηF = rheo.ηve_xz[i,j,k+1]
            # Diagonal
            if PC
                D.x[i,j+1,k+1]    = abs(ηB ./ Δ.z .^ 2 + ηF ./ Δ.z .^ 2 + ηN ./ Δ.y .^ 2 + ηS ./ Δ.y .^ 2 + (4 / 3) * ηE ./ Δ.x .^ 2 + (4 / 3) * ηW ./ Δ.x .^ 2)
                D_SC.x[i,j+1,k+1] = abs(ηB ./ Δ.z .^ 2 + ηF ./ Δ.z .^ 2 + ηN ./ Δ.y .^ 2 + ηS ./ Δ.y .^ 2 + 2 * γ ./ Δ.x .^ 2 + (4 / 3) * ηE ./ Δ.x .^ 2 + (4 / 3) * ηW ./ Δ.x .^ 2)
            end     
          end
    end
    if i<=size(D.y,1)-2 && j<=size(D.y,2) && k<=size(D.y,3)-2
        if j>1 && j<size(D.y,2) # avoid Dirichlets
            ηW = rheo.ηve_xy[i,  j,k]
            ηE = rheo.ηve_xy[i+1,j,k]
            ηS = rheo.ηve_c[i+1,j,  k+1]
            ηN = rheo.ηve_c[i+1,j+1,k+1]
            ηB = rheo.ηve_xz[i,j,k]
            ηF = rheo.ηve_xz[i,j,k+1]
            # Diagonal
            if PC
                D.y[i+1,j,k+1]    = abs(ηB ./ Δ.z .^ 2 + ηF ./ Δ.z .^ 2 + (4 / 3) * ηN ./ Δ.y .^ 2 + (4 / 3) * ηS ./ Δ.y .^ 2 + ηE ./ Δ.x .^ 2 + ηW ./ Δ.x .^ 2)
                D_SC.y[i+1,j,k+1] = abs(ηB ./ Δ.z .^ 2 + ηF ./ Δ.z .^ 2 + 2 * γ ./ Δ.y .^ 2 + (4 / 3) * ηN ./ Δ.y .^ 2 + (4 / 3) * ηS ./ Δ.y .^ 2 + ηE ./ Δ.x .^ 2 + ηW ./ Δ.x .^ 2)
            end
        end
    end
    if i<=size(D.z,1)-2 && j<=size(D.z,2)-2 && k<=size(D.z,3)
        if k>1 && k<size(D.z,3) # avoid Dirichlets
            ηW = rheo.ηve_xz[i,  j,k]
            ηE = rheo.ηve_xz[i+1,j,k]
            ηS = rheo.ηve_xy[i,j,k]
            ηN = rheo.ηve_xy[i,j+1,k]
            ηB = rheo.ηve_c[i+1,j+1,k]
            ηF = rheo.ηve_c[i+1,j+1,k+1]
            # Diagonal
            if PC
                D.z[i+1,j+1,k]    = abs((4 / 3) * ηB ./ Δ.z .^ 2 + (4 / 3) * ηF ./ Δ.z .^ 2 + ηN ./ Δ.y .^ 2 + ηS ./ Δ.y .^ 2 + ηE ./ Δ.x .^ 2 + ηW ./ Δ.x .^ 2)
                D_SC.z[i+1,j+1,k] = abs(2 * γ ./ Δ.z .^ 2 + (4 / 3) * ηB ./ Δ.z .^ 2 + (4 / 3) * ηF ./ Δ.z .^ 2 + ηN ./ Δ.y .^ 2 + ηS ./ Δ.y .^ 2 + ηE ./ Δ.x .^ 2 + ηW ./ Δ.x .^ 2)
            end
        end
    end
return nothing
end

@parallel_indices (i,j,k) function GershgorinMechanics3D!( G, D_SC, rheo, Δ, γ )
    if i<=size(G.x,1) && j<=size(G.x,2)-2 && k<=size(G.x,3)-2
        if i>1 && i<size(G.x,1) # avoid Dirichlets
            ηW = rheo.ηve_c[i  ,j+1,k+1]
            ηE = rheo.ηve_c[i+1,j+1,k+1]
            ηS = rheo.ηve_xy[i,j,k]
            ηN = rheo.ηve_xy[i,j+1,k]
            ηB = rheo.ηve_xz[i,j,k]
            ηF = rheo.ηve_xz[i,j,k+1]
            # Gerhsgorin
            G.x[i,j+1,k+1]  = abs((3 * γ + 4 * ηE) ./ Δ.x .^ 2) / 3 + abs((3 * γ + 4 * ηW) ./ Δ.x .^ 2) / 3 + abs(ηN ./ Δ.y .^ 2) + abs(ηS ./ Δ.y .^ 2) + abs(ηB ./ Δ.z .^ 2) + abs(ηF ./ Δ.z .^ 2) + abs((3 * γ - 2 * ηE + 3 * ηN) ./ (Δ.x .* Δ.y)) / 3 + abs((3 * γ - 2 * ηE + 3 * ηS) ./ (Δ.x .* Δ.y)) / 3 + abs((3 * γ + 3 * ηN - 2 * ηW) ./ (Δ.x .* Δ.y)) / 3 + abs((3 * γ + 3 * ηS - 2 * ηW) ./ (Δ.x .* Δ.y)) / 3 + abs((3 * γ + 3 * ηB - 2 * ηE) ./ (Δ.x .* Δ.z)) / 3 + abs((3 * γ + 3 * ηB - 2 * ηW) ./ (Δ.x .* Δ.z)) / 3 + abs((3 * γ - 2 * ηE + 3 * ηF) ./ (Δ.x .* Δ.z)) / 3 + abs((3 * γ + 3 * ηF - 2 * ηW) ./ (Δ.x .* Δ.z)) / 3 + abs((3 * Δ.x .^ 2 .* Δ.y .^ 2 .* (ηB + ηF) + 3 * Δ.x .^ 2 .* Δ.z .^ 2 .* (ηN + ηS) + 2 * Δ.y .^ 2 .* Δ.z .^ 2 .* (3 * γ + 2 * ηE + 2 * ηW)) ./ (Δ.x .^ 2 .* Δ.y .^ 2 .* Δ.z .^ 2)) / 3
            # Apply PC
            G.x[i,j+1,k+1] /= D_SC.x[i,j+1,k+1]      
          end
    end
    if i<=size(G.y,1)-2 && j<=size(G.y,2) && k<=size(G.y,3)-2
        if j>1 && j<size(G.y,2) # avoid Dirichlets
            ηW = rheo.ηve_xy[i,  j,k]
            ηE = rheo.ηve_xy[i+1,j,k]
            ηS = rheo.ηve_c[i+1,j,  k+1]
            ηN = rheo.ηve_c[i+1,j+1,k+1]
            ηB = rheo.ηve_xz[i,j,k]
            ηF = rheo.ηve_xz[i,j,k+1]
            # Gerhsgorin
            G.y[i+1,j,k+1]  = abs(ηE ./ Δ.x .^ 2) + abs(ηW ./ Δ.x .^ 2) + abs((3 * γ + 4 * ηN) ./ Δ.y .^ 2) / 3 + abs((3 * γ + 4 * ηS) ./ Δ.y .^ 2) / 3 + abs(ηB ./ Δ.z .^ 2) + abs(ηF ./ Δ.z .^ 2) + abs((3 * γ + 3 * ηE - 2 * ηN) ./ (Δ.x .* Δ.y)) / 3 + abs((3 * γ + 3 * ηE - 2 * ηS) ./ (Δ.x .* Δ.y)) / 3 + abs((3 * γ - 2 * ηN + 3 * ηW) ./ (Δ.x .* Δ.y)) / 3 + abs((3 * γ - 2 * ηS + 3 * ηW) ./ (Δ.x .* Δ.y)) / 3 + abs((3 * γ + 3 * ηB - 2 * ηN) ./ (Δ.y .* Δ.z)) / 3 + abs((3 * γ + 3 * ηB - 2 * ηS) ./ (Δ.y .* Δ.z)) / 3 + abs((3 * γ + 3 * ηF - 2 * ηN) ./ (Δ.y .* Δ.z)) / 3 + abs((3 * γ + 3 * ηF - 2 * ηS) ./ (Δ.y .* Δ.z)) / 3 + abs((3 * Δ.x .^ 2 .* Δ.y .^ 2 .* (ηB + ηF) + 2 * Δ.x .^ 2 .* Δ.z .^ 2 .* (3 * γ + 2 * ηN + 2 * ηS) + 3 * Δ.y .^ 2 .* Δ.z .^ 2 .* (ηE + ηW)) ./ (Δ.x .^ 2 .* Δ.y .^ 2 .* Δ.z .^ 2)) / 3
            # Apply PC
            G.y[i+1,j,k+1] /= D_SC.y[i+1,j,k+1]
        end
    end
    if i<=size(G.z,1)-2 && j<=size(G.z,2)-2 && k<=size(G.z,3)
        if k>1 && k<size(G.z,3) # avoid Dirichlets
            ηW = rheo.ηve_xz[i,  j,k]
            ηE = rheo.ηve_xz[i+1,j,k]
            ηS = rheo.ηve_xy[i,j,k]
            ηN = rheo.ηve_xy[i,j+1,k]
            ηB = rheo.ηve_c[i+1,j+1,k]
            ηF = rheo.ηve_c[i+1,j+1,k+1]
            # Gerhsgorin
            G.z[i+1,j+1,k] = abs(ηE ./ Δ.x .^ 2) + abs(ηW ./ Δ.x .^ 2) + abs(ηN ./ Δ.y .^ 2) + abs(ηS ./ Δ.y .^ 2) + abs((3 * γ + 4 * ηB) ./ Δ.z .^ 2) / 3 + abs((3 * γ + 4 * ηF) ./ Δ.z .^ 2) / 3 + abs((3 * γ - 2 * ηB + 3 * ηE) ./ (Δ.x .* Δ.z)) / 3 + abs((3 * γ - 2 * ηB + 3 * ηW) ./ (Δ.x .* Δ.z)) / 3 + abs((3 * γ + 3 * ηE - 2 * ηF) ./ (Δ.x .* Δ.z)) / 3 + abs((3 * γ - 2 * ηF + 3 * ηW) ./ (Δ.x .* Δ.z)) / 3 + abs((3 * γ - 2 * ηB + 3 * ηN) ./ (Δ.y .* Δ.z)) / 3 + abs((3 * γ - 2 * ηB + 3 * ηS) ./ (Δ.y .* Δ.z)) / 3 + abs((3 * γ - 2 * ηF + 3 * ηN) ./ (Δ.y .* Δ.z)) / 3 + abs((3 * γ - 2 * ηF + 3 * ηS) ./ (Δ.y .* Δ.z)) / 3 + abs((2 * Δ.x .^ 2 .* Δ.y .^ 2 .* (3 * γ + 2 * ηB + 2 * ηF) + 3 * Δ.x .^ 2 .* Δ.z .^ 2 .* (ηN + ηS) + 3 * Δ.y .^ 2 .* Δ.z .^ 2 .* (ηE + ηW)) ./ (Δ.x .^ 2 .* Δ.y .^ 2 .* Δ.z .^ 2)) / 3
            # Apply PC
            G.z[i+1,j+1,k] /= D_SC.z[i+1,j+1,k]
        end
    end
return nothing
end

@parallel_indices (i,j,k) function SaveOldResidual!( R_it, R, D )
    if i<=size(R.x,1) && j<=size(R.x,2) && k<=size(R.x,3)
        R_it.x[i,j,k] = R.x[i,j,k]/D.x[i,j,k]
    end
    if i<=size(R.y,1) && j<=size(R.y,2) && k<=size(R.y,3)
        R_it.y[i,j,k] = R.y[i,j,k]/D.y[i,j,k]
    end
    if i<=size(R.z,1) && j<=size(R.z,2) && k<=size(R.z,3)
        R_it.z[i,j,k] = R.z[i,j,k]/D.z[i,j,k]
    end
    return nothing
end

@parallel_indices (i,j,k) function SaveOldResidual_v1!( R_it, R )
    if i<=size(R.x,1) && j<=size(R.x,2) && k<=size(R.x,3)
        R_it.x[i,j,k] = R.x[i,j,k]
    end
    if i<=size(R.y,1) && j<=size(R.y,2) && k<=size(R.y,3)
        R_it.y[i,j,k] = R.y[i,j,k]
    end
    if i<=size(R.z,1) && j<=size(R.z,2) && k<=size(R.z,3)
        R_it.z[i,j,k] = R.z[i,j,k]
    end
    return nothing
end

@parallel_indices (i,j,k) function UpdateRates!( ∂V∂τ, R, D, a1, a2 )
    if i<=size(R.x,1) && j<=size(R.x,2) && k<=size(R.x,3)
        ∂V∂τ.x[i,j,k] = a1.x*∂V∂τ.x[i,j,k] - a2.x*R.x[i,j,k]/D.x[i,j,k]
    end
    if i<=size(R.y,1) && j<=size(R.y,2) && k<=size(R.y,3)
        ∂V∂τ.y[i,j,k] = a1.y*∂V∂τ.y[i,j,k] - a2.y*R.y[i,j,k]/D.y[i,j,k]
    end
    if i<=size(R.z,1) && j<=size(R.z,2) && k<=size(R.z,3)
        ∂V∂τ.z[i,j,k] = a1.z*∂V∂τ.z[i,j,k] - a2.z*R.z[i,j,k]/D.z[i,j,k]
    end
    return nothing
end

@parallel_indices (i,j,k) function UpdateV!( V, ∂V∂τ, h )
    if i<=size(∂V∂τ.x,1) && j<=size(∂V∂τ.x,2) && k<=size(∂V∂τ.x,3)
        V.x[i,j,k] += h.x*∂V∂τ.x[i,j,k]
    end
    if i<=size(∂V∂τ.y,1) && j<=size(∂V∂τ.y,2) && k<=size(∂V∂τ.y,3)
        V.y[i,j,k] += h.y*∂V∂τ.y[i,j,k]
    end
    if i<=size(∂V∂τ.z,1) && j<=size(∂V∂τ.z,2) && k<=size(∂V∂τ.z,3)
        V.z[i,j,k] += h.z*∂V∂τ.z[i,j,k]
    end
    return nothing
end

@parallel_indices (i,j,k) function UpdateRates_v2!( ∂V∂τ, R, D, h, c )
    if i<=size(R.x,1) && j<=size(R.x,2) && k<=size(R.x,3)
        Δτ = h.x[i,j,k]
        α1 = (2.0 - c.x*Δτ) /(2.0 + c.x*Δτ) 
        α2 = 2.0*Δτ /(2.0 + c.x*Δτ)
        ∂V∂τ.x[i,j,k] = α1*∂V∂τ.x[i,j,k] - α2*R.x[i,j,k]/D.x[i,j,k]
    end
    if i<=size(R.y,1) && j<=size(R.y,2) && k<=size(R.y,3)
        Δτ = h.y[i,j,k]
        α1 = (2.0 - c.y*Δτ) /(2.0 + c.y*Δτ) 
        α2 = 2.0*Δτ /(2.0 + c.y*Δτ)
        ∂V∂τ.y[i,j,k] = α1*∂V∂τ.y[i,j,k] - α2*R.y[i,j,k]/D.y[i,j,k]
    end
    if i<=size(R.z,1) && j<=size(R.z,2) && k<=size(R.z,3)
        Δτ = h.z[i,j,k]
        α1 = (2.0 - c.z*Δτ) /(2.0 + c.z*Δτ) 
        α2 = 2.0*Δτ /(2.0 + c.z*Δτ)
        ∂V∂τ.z[i,j,k] = α1*∂V∂τ.z[i,j,k] - α2*R.z[i,j,k]/D.z[i,j,k]
    end
    return nothing
end

@parallel_indices (i,j,k) function UpdateV_v1!( V, ∂V∂τ, h )
    if i<=size(∂V∂τ.x,1) && j<=size(∂V∂τ.x,2) && k<=size(∂V∂τ.x,3)
        V.x[i,j,k] += h.x[i,j,k]*∂V∂τ.x[i,j,k]
    end
    if i<=size(∂V∂τ.y,1) && j<=size(∂V∂τ.y,2) && k<=size(∂V∂τ.y,3)
        V.y[i,j,k] += h.y[i,j,k]*∂V∂τ.y[i,j,k]
    end
    if i<=size(∂V∂τ.z,1) && j<=size(∂V∂τ.z,2) && k<=size(∂V∂τ.z,3)
        V.z[i,j,k] += h.z[i,j,k]*∂V∂τ.z[i,j,k]
    end
    return nothing
end

@parallel_indices (i,j,k) function InterpViscosity!(rheo)
    if i<=size(rheo.ηve_c, 1)-2 && j<=size(rheo.ηve_c, 2)-2 && k<=size(rheo.ηve_c, 3)-2
        rheo.ηve_c[i+1,j+1,k+1] = 0.125*(rheo.ηve_v[i,j,k]   + rheo.ηve_v[i+1,j,k]   + rheo.ηve_v[i,j+1,k]   + rheo.ηve_v[i+1,j+1,k] +
                                         rheo.ηve_v[i,j,k+1] + rheo.ηve_v[i+1,j,k+1] + rheo.ηve_v[i,j+1,k+1] + rheo.ηve_v[i+1,j+1,k+1] )
    end
    if i<=size(rheo.ηve_xy,1) && j<=size(rheo.ηve_xy,2) && k<=size(rheo.ηve_xy,3) rheo.ηve_xy[i,j,k]  = 1.0/2.0*( rheo.ηve_v[i,j,k] + rheo.ηve_v[i,j,k+1]) end
    if i<=size(rheo.ηve_xz,1) && j<=size(rheo.ηve_xz,2) && k<=size(rheo.ηve_xz,3) rheo.ηve_xz[i,j,k]  = 1.0/2.0*( rheo.ηve_v[i,j,k] + rheo.ηve_v[i,j+1,k]) end
    if i<=size(rheo.ηve_yz,1) && j<=size(rheo.ηve_yz,2) && k<=size(rheo.ηve_yz,3) rheo.ηve_yz[i,j,k]  = 1.0/2.0*( rheo.ηve_v[i,j,k] + rheo.ηve_v[i+1,j,k]) end
    return nothing
end

@parallel_indices (i,j,k) function InterpViscosityVertex!(rheo, η, itp)
    if itp==0
        if i<=size(rheo.ηve_c, 1)-2 && j<=size(rheo.ηve_c, 2)-2 && k<=size(rheo.ηve_c, 3)-2
            rheo.ηve_c[i+1,j+1,k+1] = 0.125*(η[i,j,k]   + η[i+1,j,k]   + η[i,j+1,k]   + η[i+1,j+1,k] +
                                             η[i,j,k+1] + η[i+1,j,k+1] + η[i,j+1,k+1] + η[i+1,j+1,k+1] )
        end
        if i<=size(rheo.ηve_xy,1) && j<=size(rheo.ηve_xy,2) && k<=size(rheo.ηve_xy,3) rheo.ηve_xy[i,j,k]  = 1.0/2.0*( η[i,j,k] + η[i,j,k+1]) end
        if i<=size(rheo.ηve_xz,1) && j<=size(rheo.ηve_xz,2) && k<=size(rheo.ηve_xz,3) rheo.ηve_xz[i,j,k]  = 1.0/2.0*( η[i,j,k] + η[i,j+1,k]) end
        if i<=size(rheo.ηve_yz,1) && j<=size(rheo.ηve_yz,2) && k<=size(rheo.ηve_yz,3) rheo.ηve_yz[i,j,k]  = 1.0/2.0*( η[i,j,k] + η[i+1,j,k]) end
    elseif itp==1
        if i<=size(rheo.ηve_c, 1)-2 && j<=size(rheo.ηve_c, 2)-2 && k<=size(rheo.ηve_c, 3)-2
            rheo.ηve_c[i+1,j+1,k+1] = 2.0 #1.0/( 0.125*(1.0/rheo.ηve_v[i,j,k]   + 1.0/rheo.ηve_v[i+1,j,k]   + 1.0/rheo.ηve_v[i,j+1,k]   + 1.0/rheo.ηve_v[i+1,j+1,k] +
                                                #    1.0/rheo.ηve_v[i,j,k+1] + 1.0/rheo.ηve_v[i+1,j,k+1] + 1.0/rheo.ηve_v[i,j+1,k+1] + 1.0/rheo.ηve_v[i+1,j+1,k+1]) )
        end
        if i<=size(rheo.ηve_xy,1) && j<=size(rheo.ηve_xy,2) && k<=size(rheo.ηve_xy,3) rheo.ηve_xy[i,j,k]  = 1.0/*( 0.5*(1.0/rheo.ηve_v[i,j,k] + 1.0/rheo.ηve_v[i,j,k+1])) end
        if i<=size(rheo.ηve_xz,1) && j<=size(rheo.ηve_xz,2) && k<=size(rheo.ηve_xz,3) rheo.ηve_xz[i,j,k]  = 1.0/*( 0.5*(1.0/rheo.ηve_v[i,j,k] + 1.0/rheo.ηve_v[i,j+1,k])) end
        if i<=size(rheo.ηve_yz,1) && j<=size(rheo.ηve_yz,2) && k<=size(rheo.ηve_yz,3) rheo.ηve_yz[i,j,k]  = 1.0/*( 0.5*(1.0/rheo.ηve_v[i,j,k] + 1.0/rheo.ηve_v[i+1,j,k])) end
    end
    return nothing
end

@parallel_indices (i,j,k) function InterpViscosityCenter!(rheo, η, itp)
    if itp==0
        if i<=size(rheo.ηve_v, 1) && j<=size(rheo.ηve_v, 2) && k<=size(rheo.ηve_v, 3)
            rheo.ηve_v[i,j,k] = 0.125*(η[i,j,k]   + η[i+1,j,k]   + η[i,j+1,k]   + η[i+1,j+1,k] +
                                       η[i,j,k+1] + η[i+1,j,k+1] + η[i,j+1,k+1] + η[i+1,j+1,k+1] )
        end
        if i<=size(rheo.ηve_xy,1) && j<=size(rheo.ηve_xy,2) && k<=size(rheo.ηve_xy,3) 
            rheo.ηve_xy[i,j,k]  = 1.0/4.0*( η[i,j,k] + η[i+1,j,k] + η[i,j+1,k] + η[i+1,j+1,k]) 
        end
        if i<=size(rheo.ηve_xz,1) && j<=size(rheo.ηve_xz,2) && k<=size(rheo.ηve_xz,3) 
            rheo.ηve_xz[i,j,k]  = 1.0/4.0*( η[i,j,k] + η[i+1,j,k] + η[i,j,k+1] + η[i+1,j,k+1]) 
        end
        if i<=size(rheo.ηve_yz,1) && j<=size(rheo.ηve_yz,2) && k<=size(rheo.ηve_yz,3) 
            rheo.ηve_yz[i,j,k]  = 1.0/4.0*( η[i,j,k] + η[i,j+1,k] + η[i,j,k+1] + η[i,j+1,k+1]) 
        end
    elseif itp==1
        if i<=size(rheo.ηve_v, 1) && j<=size(rheo.ηve_v, 2) && k<=size(rheo.ηve_v, 3)
            rheo.ηve_v[i,j,k] = 1/(0.125*(1/η[i,j,k]   + 1/η[i+1,j,k]   + 1/η[i,j+1,k]   + 1/η[i+1,j+1,k] +
            1/η[i,j,k+1] + 1/η[i+1,j,k+1] + 1/η[i,j+1,k+1] + 1/η[i+1,j+1,k+1] ))
        end
        if i<=size(rheo.ηve_xy,1) && j<=size(rheo.ηve_xy,2) && k<=size(rheo.ηve_xy,3) 
            rheo.ηve_xy[i,j,k]  = 1/(1.0/4.0*( 1/η[i,j,k] + 1/η[i+1,j,k] + 1/η[i,j+1,k] + 1/η[i+1,j+1,k])) 
        end
        if i<=size(rheo.ηve_xz,1) && j<=size(rheo.ηve_xz,2) && k<=size(rheo.ηve_xz,3) 
            rheo.ηve_xz[i,j,k]  = 1/(1.0/4.0*( 1/η[i,j,k] + 1/η[i+1,j,k] + 1/η[i,j,k+1] + 1/η[i+1,j,k+1]))
        end
        if i<=size(rheo.ηve_yz,1) && j<=size(rheo.ηve_yz,2) && k<=size(rheo.ηve_yz,3) 
            rheo.ηve_yz[i,j,k]  = 1/(1.0/4.0*( 1/η[i,j,k] + 1/η[i,j+1,k] + 1/η[i,j,k+1] + 1/η[i,j+1,k+1]))
        end
    end
    return nothing
end

@parallel_indices (i,j,k) function InterpViscosityHarm!(rheo)
    if i<=size(rheo.ηve_c, 1)-2 && j<=size(rheo.ηve_c, 2)-2 && k<=size(rheo.ηve_c, 3)-2
        rheo.ηve_c[i+1,j+1,k+1] = 1.0/( 0.125*(1.0/rheo.ηve_v[i,j,k]   + 1.0/rheo.ηve_v[i+1,j,k]   + 1.0/rheo.ηve_v[i,j+1,k]   + 1.0/rheo.ηve_v[i+1,j+1,k] +
                                               1.0/rheo.ηve_v[i,j,k+1] + 1.0/rheo.ηve_v[i+1,j,k+1] + 1.0/rheo.ηve_v[i,j+1,k+1] + 1.0/rheo.ηve_v[i+1,j+1,k+1]) )
    end
    if i<=size(rheo.ηve_xy,1) && j<=size(rheo.ηve_xy,2) && k<=size(rheo.ηve_xy,3) rheo.ηve_xy[i,j,k]  = 1.0/*( 0.5*(1.0/rheo.ηve_v[i,j,k] + 1.0/rheo.ηve_v[i,j,k+1])) end
    if i<=size(rheo.ηve_xz,1) && j<=size(rheo.ηve_xz,2) && k<=size(rheo.ηve_xz,3) rheo.ηve_xz[i,j,k]  = 1.0/*( 0.5*(1.0/rheo.ηve_v[i,j,k] + 1.0/rheo.ηve_v[i,j+1,k])) end
    if i<=size(rheo.ηve_yz,1) && j<=size(rheo.ηve_yz,2) && k<=size(rheo.ηve_yz,3) rheo.ηve_yz[i,j,k]  = 1.0/*( 0.5*(1.0/rheo.ηve_v[i,j,k] + 1.0/rheo.ηve_v[i+1,j,k])) end
    return nothing
end

@parallel_indices (i,j,k) function ComputeStrainRates!( ∇V, ε̇, V, Δ )
    if i<=size(ε̇.xx,1)-2 && j<=size(ε̇.xx,2)-2 && k<=size(ε̇.xx,3)-2
        dVxΔx      = (V.x[i+1,j+1,k+1] - V.x[i,j+1,k+1]) / Δ.x
        dVyΔy      = (V.y[i+1,j+1,k+1] - V.y[i+1,j,k+1]) / Δ.y
        dVzΔz      = (V.z[i+1,j+1,k+1] - V.z[i+1,j+1,k]) / Δ.z
        ∇V[i+1,j+1,k+1]   = ∇V_ijk = dVxΔx + dVyΔy + dVzΔz
        ε̇.xx[i+1,j+1,k+1] = dVxΔx - ∇V_ijk / 3
        ε̇.yy[i+1,j+1,k+1] = dVyΔy - ∇V_ijk / 3
        ε̇.zz[i+1,j+1,k+1] = dVzΔz - ∇V_ijk / 3
    end
    if i<=size(ε̇.xy,1) && j<=size(ε̇.xy,2) && k<=size(ε̇.xy,3)-2
        dVxΔy      = (V.x[i,j+1,k+1] - V.x[i,j,k+1]) / Δ.y 
        dVyΔx      = (V.y[i+1,j,k+1] - V.y[i,j,k+1]) / Δ.x 
        ε̇.xy[i,j,k+1] = 0.5*(dVxΔy + dVyΔx)
    end
    if i<=size(ε̇.xz,1) && j<=size(ε̇.xz,2)-2 && k<=size(ε̇.xz,3)
        dVxΔz      = (V.x[i  ,j+1,k+1] - V.x[i,j+1,k]) / Δ.z                     
        dVzΔx      = (V.z[i+1,j+1,k  ] - V.z[i,j+1,k]) / Δ.x 
        ε̇.xz[i,j+1,k] = 0.5*(dVxΔz + dVzΔx)
    end
    if i<=size(ε̇.yz,1)-2 && j<=size(ε̇.yz,2) && k<=size(ε̇.yz,3)
        dVyΔz      = (V.y[i+1,j,k+1] - V.y[i+1,j,k]) / Δ.z 
        dVzΔy      = (V.z[i+1,j+1,k] - V.z[i+1,j,k]) / Δ.y 
        ε̇.yz[i+1,j,k] = 0.5*(dVyΔz + dVzΔy)
    end
    return nothing
end

@parallel_indices (i,j,k) function ComputeStress!( P, τ, ε̇, rheo, params )
    if i<=size(ε̇.xx,1)-2 && j<=size(ε̇.xx,2)-2 && k<=size(ε̇.xx,3)-2
        ηve_c_x_2 = 2 * rheo.ηve_c[i+1,j+1,k+1]
        τ.xx[i+1,j+1,k+1] = ηve_c_x_2 * ε̇.xx[i+1,j+1,k+1]
        τ.yy[i+1,j+1,k+1] = ηve_c_x_2 * ε̇.yy[i+1,j+1,k+1]
        τ.zz[i+1,j+1,k+1] = ηve_c_x_2 * ε̇.zz[i+1,j+1,k+1]
    end
    if i<=size(ε̇.xy,1) && j<=size(ε̇.xy,2) && k<=size(ε̇.xy,3)-2
        τ.xy[i,j,k+1] = 2.0 * rheo.ηve_xy[i,j,k] * ε̇.xy[i,j,k+1]
    end
    if i<=size(ε̇.xz,1) && j<=size(ε̇.xz,2)-3 && k<=size(ε̇.xz,3)
        τ.xz[i,j+1,k] =  2.0 * rheo.ηve_xz[i,j,k] * ε̇.xz[i,j+1,k]
    end
    if i<=size(ε̇.yz,1)-2 && j<=size(ε̇.yz,2) && k<=size(ε̇.yz,3)
        τ.yz[i+1,j,k] = 2.0 * rheo.ηve_yz[i,j,k] *  ε̇.yz[i+1,j,k]
    end
    return nothing
end

@parallel_indices (i,j,k) function ComputeResiduals!(R, RP, τ, P, ∇V, b, D, Δ  )
    if i<=size(R.x,1) && j<=size(R.x,2)-2 && k<=size(R.x,3)-2
        if 1 < i < size(R.x,1) # avoid Dirichlets
            Rx  = b.x[i,j+1,k+1]
            Rx += (τ.xx[i+1,j+1,k+1] - τ.xx[i,j+1,k+1]) / Δ.x
            Rx -= (   P[i+1,j+1,k+1] -    P[i,j+1,k+1]) / Δ.x
            Rx += (τ.xy[i,j+1,k+1] - τ.xy[i,j,k+1]) / Δ.y
            Rx += (τ.xz[i,j+1,k+1] - τ.xz[i,j+1,k]) / Δ.z
            # R.x[i,j+1,k+1] /= D.x[i,j+1,k+1] * -1.
            R.x[i,j+1,k+1] = -Rx
        end
    end
    if i<=size(R.y,1)-2 && j<=size(R.y,2) && k<=size(R.y,3)-2
        if 1 < j < size(R.y,2) # avoid Dirichlets
            Ry  = b.y[i+1,j,k+1]
            Ry += (τ.yy[i+1,j+1,k+1] - τ.yy[i+1,j,k+1]) / Δ.y
            Ry -= (   P[i+1,j+1,k+1] -    P[i+1,j,k+1]) / Δ.y
            Ry += (τ.xy[i+1,j,k+1] - τ.xy[i,j,k+1]) / Δ.x
            Ry += (τ.yz[i+1,j,k+1] - τ.yz[i+1,j,k]) / Δ.z
            # R.y[i+1,j,k+1] /= D.y[i+1,j,k+1] * -1.
            R.y[i+1,j,k+1] =  -Ry

        end
    end
    if  i<=size(R.z,1)-2 && j<=size(R.z,2)-2 && k<=size(R.z,3)
        if 1 < k < size(R.z,3) # avoid Dirichlets
            Rz  = b.z[i+1,j,k+1]
            Rz += (τ.zz[i+1,j+1,k+1] - τ.zz[i+1,j+1,k]) / Δ.z
            Rz -= (   P[i+1,j+1,k+1] -    P[i+1,j+1,k]) / Δ.z
            Rz += (τ.xz[i+1,j+1,k] - τ.xz[i,j+1,k]) / Δ.x
            Rz += (τ.yz[i+1,j+1,k] - τ.yz[i+1,j,k]) / Δ.y
            # R.z[i+1,j+1,k] /= D.z[i+1,j+1,k] * -1.
            R.z[i+1,j+1,k] = -Rz

        end
    end
    if i<=size(RP,1) && j<=size(RP,2) && k<=size(RP,3)
        RP[i, j, k] = -∇V[i+1,j+1,k+1]
    end
    return nothing
end

@views function SetDamping(h, ∂V∂τ, R,  R_it, D_SC, In, ThreeD, numerics)

    λmin0 = abs( (sum(.-(h.x[In.x...].*∂V∂τ.x[In.x...]).*(R.x[In.x...].-R_it.x[In.x...])./D_SC.x[In.x...]) + sum(.-(h.y[In.y...].*∂V∂τ.y[In.y...]).*(R.y[In.y...].-R_it.y[In.y...])./D_SC.y[In.y...]) + sum(.-(h.z[In.z...].*∂V∂τ.z[In.z...]).*(R.z[In.z...].-R_it.z[In.z...])./D_SC.z[In.z...]) ) / (sum((h.x[In.x...].*∂V∂τ.x[In.x...]).*(h.x[In.x...].*∂V∂τ.x[In.x...])) + sum((h.y[In.y...].*∂V∂τ.y[In.y...]).*(h.y[In.y...].*∂V∂τ.y[In.y...])) + sum((h.z[In.z...].*∂V∂τ.z[In.z...]).*(h.z[In.z...].*∂V∂τ.z[In.z...]))) )
    if numerics.λdim
        λmin = ( x =        (0.2*abs( (sum(.-(h.x[In.x...].*∂V∂τ.x[In.x...]).*(R.x[In.x...].-R_it.x[In.x...])./D_SC.x[In.x...]) ) / (sum((h.x[In.x...].*∂V∂τ.x[In.x...]).*(h.x[In.x...].*∂V∂τ.x[In.x...])))) ),
                 y =        (0.2*abs( (sum(.-(h.y[In.y...].*∂V∂τ.y[In.y...]).*(R.y[In.y...].-R_it.y[In.y...])./D_SC.y[In.y...]) ) / (sum((h.y[In.y...].*∂V∂τ.y[In.y...]).*(h.y[In.y...].*∂V∂τ.y[In.y...])))) ),              
                 z =        (0.2*abs( (sum(.-(h.z[In.z...].*∂V∂τ.z[In.z...]).*(R.z[In.z...].-R_it.z[In.z...])./D_SC.z[In.z...]) ) / (sum((h.z[In.z...].*∂V∂τ.z[In.z...]).*(h.z[In.z...].*∂V∂τ.z[In.z...])))) ),
                #  z = ThreeD*(0.2*abs( (sum(.-(h.z[In.z...].*∂V∂τ.z[In.z...]).*(R.z[In.z...].-R_it.z[In.z...])./D_SC.z[In.z...]) ) / (sum((h.z[In.z...].*∂V∂τ.z[In.z...]).*(h.z[In.z...].*∂V∂τ.z[In.z...])))) ),
        )
    else
        λmin = ( x = λmin0,
                 y = λmin0,
                 z = λmin0,
    )
    end
    return ( y    = 2.0*sqrt(λmin.x)*numerics.cfact,
             x    = 2.0*sqrt(λmin.y)*numerics.cfact,
             z    = 2.0*sqrt(λmin.z)*numerics.cfact,)
end

@views function SetPseudoTimeStep!(h, G, numerics)
    if numerics.Δτloc
        h.x .= 2.0./sqrt.((G.x))*numerics.CFL
        h.y .= 2.0./sqrt.((G.y))*numerics.CFL
        h.z .= 2.0./sqrt.((G.z))*numerics.CFL
    else
        h.x .= 2.0./sqrt.(maximum(G.x))*numerics.CFL
        h.y .= 2.0./sqrt.(maximum(G.y))*numerics.CFL
        h.z .= 2.0./sqrt.(maximum(G.z))*numerics.CFL
    end
    return nothing
end

@parallel_indices (i,j,k) function ComputeResidualsσyyBC!(R, RP, τ, P, ∇V, b, D, σyyBC, Δ  )
    if i<=size(R.x,1) && j<=size(R.x,2)-2 && k<=size(R.x,3)-2
        if 1 < i < size(R.x,1) # avoid Dirichlets
            Rx  = b.x[i,j+1,k+1]
            Rx += (τ.xx[i+1,j+1,k+1] - τ.xx[i,j+1,k+1]) / Δ.x
            Rx -= (   P[i+1,j+1,k+1] -    P[i,j+1,k+1]) / Δ.x
            Rx += (τ.xy[i,j+1,k+1] - τ.xy[i,j,k+1]) / Δ.y
            Rx += (τ.xz[i,j+1,k+1] - τ.xz[i,j+1,k]) / Δ.z
            R.x[i,j+1,k+1] = -Rx
        end
    end
    if i<=size(R.y,1)-2 && j<=size(R.y,2) && k<=size(R.y,3)-2
        if 1 < j <= size(R.y,2) # avoid Dirichlets at the base only
            
            if j == size(R.y,2)
                σyyS = τ.yy[i+1,j,k+1]  -  P[i+1,j,k+1]
                σyyN = 2*σyyBC[i+1,k+1] -  σyyS
                Ry   = b.y[i+1,j,k+1]
                Ry  += (σyyN - σyyS) / Δ.y
                Ry  += (τ.xy[i+1,j,k+1] - τ.xy[i,j,k+1]) / Δ.x
                Ry  += (τ.yz[i+1,j,k+1] - τ.yz[i+1,j,k]) / Δ.z
                R.y[i+1,j,k+1] = -Ry
                # if i+1==2 && k+1==2
                #     R.y[i+1,j,k+1] = 0.0
                # end
            else
                σyyS = τ.yy[i+1,j,k+1]   -  P[i+1,j,k+1]
                σyyN = τ.yy[i+1,j+1,k+1] -  P[i+1,j+1,k+1]
                Ry   = b.y[i+1,j,k+1]
                Ry  += (σyyN - σyyS) / Δ.y
                Ry  += (τ.xy[i+1,j,k+1] - τ.xy[i,j,k+1]) / Δ.x
                Ry  += (τ.yz[i+1,j,k+1] - τ.yz[i+1,j,k]) / Δ.z
                R.y[i+1,j,k+1] = -Ry
            end
            
            # σyyS = τ.yy[i+1,j,k+1]   -  P[i+1,j,k+1]
            # if j == size(R.y,2)
            #     σyyN = 2*σyyBC[i+1,k+1] -  σyyS
            # else
            #     σyyN = τ.yy[i+1,j+1,k+1] -  P[i+1,j+1,k+1]
            # end
            # Ry  = b.y[i+1,j,k+1]
            # Ry += (σyyN - σyyS) / Δ.y
            # Ry += (τ.xy[i+1,j,k+1] - τ.xy[i,j,k+1]) / Δ.x
            # Ry += (τ.yz[i+1,j,k+1] - τ.yz[i+1,j,k]) / Δ.z
            # R.y[i+1,j,k+1] =  -Ry
            # if j == size(R.y,2) && i==2 && k==2
            #     R.y[i+1,j,k+1] =  0.
            # end  
        end
    end
    if  i<=size(R.z,1)-2 && j<=size(R.z,2)-2 && k<=size(R.z,3)
        if 1 < k < size(R.z,3) # avoid Dirichlets
            Rz  = b.z[i+1,j,k+1]
            Rz += (τ.zz[i+1,j+1,k+1] - τ.zz[i+1,j+1,k]) / Δ.z
            Rz -= (   P[i+1,j+1,k+1] -    P[i+1,j+1,k]) / Δ.z
            Rz += (τ.xz[i+1,j+1,k] - τ.xz[i,j+1,k]) / Δ.x
            Rz += (τ.yz[i+1,j+1,k] - τ.yz[i+1,j,k]) / Δ.y
            R.z[i+1,j+1,k] = -Rz
        end
    end
    if i<=size(RP,1) && j<=size(RP,2) && k<=size(RP,3)
        RP[i, j, k] = -∇V[i+1,j+1,k+1]
    end
    return nothing
end

@parallel_indices (i,j,k) function DiagMechanics3DσyyBC!( D, D_SC, rheo, Δ, γ, PC )
    if i<=size(D.x,1) && j<=size(D.x,2)-2 && k<=size(D.x,3)-2
        if i>1 && i<size(D.x,1) # avoid Dirichlets
            ηW = rheo.ηve_c[i  ,j+1,k+1]
            ηE = rheo.ηve_c[i+1,j+1,k+1]
            ηS = rheo.ηve_xy[i,j,k]
            ηN = rheo.ηve_xy[i,j+1,k]
            ηB = rheo.ηve_xz[i,j,k]
            ηF = rheo.ηve_xz[i,j,k+1]
            # Diagonal
            if PC
                D.x[i,j+1,k+1]    = abs(ηB ./ Δ.z .^ 2 + ηF ./ Δ.z .^ 2 + ηN ./ Δ.y .^ 2 + ηS ./ Δ.y .^ 2 + (4 / 3) * ηE ./ Δ.x .^ 2 + (4 / 3) * ηW ./ Δ.x .^ 2)
                D_SC.x[i,j+1,k+1] = abs(ηB ./ Δ.z .^ 2 + ηF ./ Δ.z .^ 2 + ηN ./ Δ.y .^ 2 + ηS ./ Δ.y .^ 2 + 2 * γ ./ Δ.x .^ 2 + (4 / 3) * ηE ./ Δ.x .^ 2 + (4 / 3) * ηW ./ Δ.x .^ 2)
            end     
          end
    end
    if i<=size(D.y,1)-2 && j<=size(D.y,2) && k<=size(D.y,3)-2
        if j>1 
            if j==size(D.y,2) # avoid Dirichlets
                ηN = 0.
                ηB = 0.
                ηF = 0.
            else
                ηN = rheo.ηve_c[i+1,j+1,k+1]
                ηB = rheo.ηve_xz[i,j,k]
                ηF = rheo.ηve_xz[i,j,k+1]
            end
            ηW = rheo.ηve_xy[i,  j,k]
            ηE = rheo.ηve_xy[i+1,j,k]
            ηS = rheo.ηve_c[i+1,j,  k+1]
            # ηN = rheo.ηve_c[i+1,j+1,k+1]

            # Diagonal
            if PC
                D.y[i+1,j,k+1]    = abs(ηB ./ Δ.z .^ 2 + ηF ./ Δ.z .^ 2 + (4 / 3) * ηN ./ Δ.y .^ 2 + (4 / 3) * ηS ./ Δ.y .^ 2 + ηE ./ Δ.x .^ 2 + ηW ./ Δ.x .^ 2)
                D_SC.y[i+1,j,k+1] = abs(ηB ./ Δ.z .^ 2 + ηF ./ Δ.z .^ 2 + 2 * γ ./ Δ.y .^ 2 + (4 / 3) * ηN ./ Δ.y .^ 2 + (4 / 3) * ηS ./ Δ.y .^ 2 + ηE ./ Δ.x .^ 2 + ηW ./ Δ.x .^ 2)
            end
        end
    end
    if i<=size(D.z,1)-2 && j<=size(D.z,2)-2 && k<=size(D.z,3)
        if k>1 && k<size(D.z,3) # avoid Dirichlets
            ηW = rheo.ηve_xz[i,  j,k]
            ηE = rheo.ηve_xz[i+1,j,k]
            ηS = rheo.ηve_xy[i,j,k]
            ηN = rheo.ηve_xy[i,j+1,k]
            ηB = rheo.ηve_c[i+1,j+1,k]
            ηF = rheo.ηve_c[i+1,j+1,k+1]
            # Diagonal
            if PC
                D.z[i+1,j+1,k]    = abs((4 / 3) * ηB ./ Δ.z .^ 2 + (4 / 3) * ηF ./ Δ.z .^ 2 + ηN ./ Δ.y .^ 2 + ηS ./ Δ.y .^ 2 + ηE ./ Δ.x .^ 2 + ηW ./ Δ.x .^ 2)
                D_SC.z[i+1,j+1,k] = abs(2 * γ ./ Δ.z .^ 2 + (4 / 3) * ηB ./ Δ.z .^ 2 + (4 / 3) * ηF ./ Δ.z .^ 2 + ηN ./ Δ.y .^ 2 + ηS ./ Δ.y .^ 2 + ηE ./ Δ.x .^ 2 + ηW ./ Δ.x .^ 2)
            end
        end
    end
return nothing
end


