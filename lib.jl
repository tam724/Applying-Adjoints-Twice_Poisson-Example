using Gridap, GridapGmsh
using Plots, Interpolations , SparseArrays, Distributions, Pardiso, Krylov
using LinearAlgebra
using Zygote
using FiniteDifferences, Optim
using Zygote: @adjoint
using Lux, Random, Optimisers
using LaTeXStrings
using BenchmarkTools
using IncompleteLU
using SparseArrays
using Krylov
using Plots
using StaticArrays

function assemble_bilinear(a, pars, U, V)
    u = get_trial_fe_basis(U)
    v = get_fe_basis(V)
    matcontribs = a(u, v, pars...)
    data = Gridap.FESpaces.collect_cell_matrix(U, V, matcontribs)
    return assemble_matrix(SparseMatrixAssembler(U, V), data)
end

function assemble_linear(b, pars, U, V)
    v = get_fe_basis(V)
    veccontribs = b(v, pars...)
    data = Gridap.FESpaces.collect_cell_vector(V, veccontribs)
    return assemble_vector(SparseMatrixAssembler(U, V), data)
end

function project_function(U, (dΩ, dΓ, n, α_h), f)
	op = AffineFEOperator((u, v) -> ∫(u*v)dΩ, v -> ∫(v*f)dΩ, U, U)
	return Gridap.solve(op)
end

function find_nzval_index(A, i, j)
    for nzval_idx in nzrange(A, j)
        if A.rowval[nzval_idx] == i
            return nzval_idx
        end
    end
    @error "index not found."
end

function build_projector(f, U, V)
    v = get_fe_basis(V)
    u = get_trial_fe_basis(U)

    domain_contrib = f(u, v)

    # compute sparsity structure of the sparse matrix
    IJs = Tuple{Int64, Int64}[]

    for Ω in Gridap.CellData.get_domains(domain_contrib)
        # cell_mats = domain_contrib[Ω]

        U_ids = Gridap.get_cell_dof_ids(U, Ω)
        V_ids = Gridap.get_cell_dof_ids(V, Ω)
        for cell_idx in 1:Gridap.num_cells(Ω)
            U_id = U_ids[cell_idx]
            V_id = V_ids[cell_idx]

            # cell_mat = cell_mats[cell_idx]
            for (j, U_idj) in enumerate(U_id)
                for (i, V_idi) in enumerate(V_id)
                    #if !iszero(cell_mat[i, j]) 
                        push!(IJs, (V_idi, U_idj))
                    #end
                end
            end
        end
    end
    #sort!(IJs)
    #unique!(IJs)
    A_skeleton = sparse(getindex.(IJs, 1), getindex.(IJs, 2), zeros(length(IJs)))

    Dc = num_dims(get_background_model(get_triangulation(U)))

    I_p = Int64[]
    J_p = Int64[]
    V_p = Float64[]
    for Ω in Gridap.CellData.get_domains(domain_contrib)
        cell_mats = domain_contrib[Ω]

        U_ids = Gridap.get_cell_dof_ids(U, Ω)
        V_ids = Gridap.get_cell_dof_ids(V, Ω)
        for cell_idx in 1:Gridap.num_cells(Ω)
            U_id = U_ids[cell_idx]
            V_id = V_ids[cell_idx]

            cell_mat = cell_mats[cell_idx]
            for (j, U_idj) in enumerate(U_id)
                for (i, V_idi) in enumerate(V_id)
                    if !iszero(cell_mat[i, j])
                        i_p = find_nzval_index(A_skeleton, V_idi, U_idj)
                        # j_p = cell_idx
                        j_p = Gridap.get_glue(Ω, Val(Dc)).tface_to_mface[cell_idx]
                        push!(I_p, i_p)
                        push!(J_p, j_p)
                        push!(V_p, cell_mat[i, j])
                    end
                end
            end
        end
    end

    m = length(A_skeleton.nzval)
    n = num_cells(get_background_model(get_triangulation(U)))
    return A_skeleton, sparse(I_p, J_p, V_p, m, n)
end

function backproject!(b, backprojector, u, v)
    cache, backpr = backprojector
    for (i, (A, u_id, v_id)) in enumerate(backpr)
        @inbounds mul!(cache, A, @view(u[u_id]))
        @inbounds b[i] = dot(@view(v[v_id]), cache)
    end
end         

function build_backprojector(g, U, V)
    v = get_fe_basis(V)
    u = get_trial_fe_basis(U)

    domain_contrib = g(u, v)
    Dc = num_dims(get_background_model(get_triangulation(U)))

    backprojector = Tuple{Matrix{Float64}, Vector{Int32}, Vector{Int32}}[]
    resize!(backprojector, num_cells(get_background_model(get_triangulation(U))))

    for Ω in Gridap.CellData.get_domains(domain_contrib)
        cell_mats = domain_contrib[Ω]

        U_ids = Gridap.get_cell_dof_ids(U, Ω)
        V_ids = Gridap.get_cell_dof_ids(V, Ω)
        for cell_idx in 1:Gridap.num_cells(Ω)
            U_id = U_ids[cell_idx]
            V_id = V_ids[cell_idx]

            cell_mat = cell_mats[cell_idx]
            idx = Gridap.get_glue(Ω, Val(Dc)).tface_to_mface[cell_idx]

            if isassigned(backprojector, idx)
                @assert backprojector[idx][2] == U_id
                @assert backprojector[idx][3] == V_id
                backprojector[idx][1] .+= cell_mat
            else
                backprojector[idx] = (Matrix(cell_mat), Vector(U_id), Vector(V_id))
            end
        end
    end

    for i in eachindex(backprojector)
        if !isassigned(backprojector, i)
            backprojector[i] = (zeros(0, 0), zeros(Int64, 0), zeros(Int64, 0))
        end
    end
    
    return (zeros(3), backprojector)
end


struct ModelFunction{Mspace, Uspace, Vspace, Mfunc, Omega, Backr, Pars}
    measurement_angles::Vector{Float64}
    extraction_locations::Vector{Tuple{Float64, Float64}}

    M::Mspace
    U::Uspace
    V::Vspace
    
    m::Mfunc
    
    A::SparseMatrixCSC{Float64}
    AT::SparseMatrixCSC{Float64}

    iterative::Bool

    I_M::Vector{Float64}
    
    b::Matrix{Float64}
    c::Matrix{Float64}
    
    Ω::Omega
    pars::Pars

    projr::SparseMatrixCSC{Float64}
    Atemp::SparseMatrixCSC{Float64}
    backr::Backr

    temp_storage::Matrix{Float64}
end

function ModelFunction(measurement_angles, extraction_locations, grid_path; iterative=false, order=1)
    model = GmshDiscreteModel(joinpath(@__DIR__(), grid_path))
    refel = ReferenceFE(lagrangian, Float64, order)
    V = TestFESpace(model, refel, conformity=:H1)
    U = TrialFESpace(V)
    M = TestFESpace(model, ReferenceFE(lagrangian, Float64, 0), conformity=:L2)
    
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2*order + 1)
    Γ = BoundaryTriangulation(model)
    dΓ = Measure(Γ, 2*order + 1)
    n = get_normal_vector(Γ)

    # nietsche coefficient α / cell radius
    α_h = 500.0/(sqrt(π/num_free_dofs(M)))
    #∫(FEFunction(FF.M, ones(num_free_dofs(FF.M))))*FF.pars.dΩ)[FF.Ω][:]
    pars = (dΩ=dΩ, dΓ=dΓ, n=n, α_h=[α_h])
    
    m = FEFunction(M, ones(num_free_dofs(M)))
    
    A = assemble_bilinear(a_, (m, pars), U, V)
    AT = sparse(transpose(A))

    excitations = hcat([assemble_linear(b_, (m, excitation_func(θ), pars), U, V) for θ in measurement_angles]...)
    extractions = hcat([assemble_linear(c_, (m, extraction_func(loc), pars), U, V) for loc in extraction_locations]...)
    
    I_M = 1.0 ./ Vector(diag(assemble_bilinear((u, v) -> ∫(u*v)dΩ, (), M, M)))
    # for speed 
    backr = build_backprojector((u, v) -> dot_a_temp(u, v, nothing, nothing, pars), U, V)
    skel, projr = build_projector((u, v) -> a1_(u, v, nothing, pars), U, V)
    Atemp = assemble_bilinear(a2_, (m, pars), U, V)
    return ModelFunction{typeof(M), typeof(U), typeof(V), typeof(m), typeof(Ω), typeof(backr), typeof(pars)}(
        measurement_angles, extraction_locations, M, U, V, m, A, AT, iterative, I_M, excitations, extractions, Ω, pars, projr, Atemp, backr, zeros(num_free_dofs(V), length(extraction_locations)))
end

function set_params!(f, p)
    f.m.free_values .= p
    # f.A .= assemble_bilinear(a_, (f.m, f.pars), f.U, f.V)
    f.A.nzval .= f.Atemp.nzval .+ (f.projr * p)
    f.AT .= transpose(f.A)
    # for i in 1:length(f.measurement_angles)
    # 	f.b[:, i] .= assemble_linear(b_, (m(f.pp), g[i], f.pars), f.U, f.V)
    # end
    nothing
end


# function set_alpha!(f, α)
#     # this should now also recompute the projections!
#     f.pars.α_h[1] = α
#     f.A .= assemble_bilinear(a_, (f.m, f.pars), f.U, f.V)
#     f.AT .= transpose(f.A)
#     f.b .= hcat([assemble_linear(b_, (f.m, g[i], f.pars), f.U, f.V) for i in 1:length(f.measurement_angles)]...)
# end

function plot_solution(sol, clims=extrema(sol.free_values), res=200, rev=false; cmap=cgrad(:temperaturemap, rev=rev))
	gr()
	g = x -> sol(x / norm(x)*0.98)
	scale_point(x) = (sqrt(x[1]*x[1] + x[2]*x[2]) >= 0.99) ? x / norm(x) * 0.99 : x
	# cmap=cgrad(:temperaturemap, rev=rev)
	xpoints = range(-1.0, 1.0, length=res)
	ypoints = range(-1.0, 1.0, length=res)

	points = scale_point.(Point.(xpoints', ypoints))[:]
	z = reshape(sol(points), (res, res))
	p = contourf(xpoints, ypoints, z, aspect_ratio=:equal, cmap=cmap, clims=clims, linewidth=0, levels=30, grid=:none, right_margin=5Plots.mm)
    #p = heatmap(xpoints, ypoints, z, aspect_ratio=:equal, cmap=cmap, clims=clims, grid=:none, right_margin=5Plots.mm)
    plot!(-1.01:0.005:1.01, x -> (1 - x^2) > 0 ? sqrt(1 - x^2) : 0.0, fillrange = x -> 1.01, color=:white, linewidth=0, label=nothing)
    plot!(-1.01:0.005:1.01, x -> -1.01, fillrange = x -> (1 - x^2) > 0 ? -sqrt(1 - x^2) : 0.0, color=:white, linewidth=0, label=nothing)
	#plot!(1.3*sin.(0:0.05:2π), 1.3*cos.(0:0.05:2π), color=:white, label=nothing, linewidth=94) #, linecolor=get.(Ref(cgrad(:thermal)), g.(Point.(sin.(0:0.01:2π), cos.(0:0.01:2π)))), linewidth=3)
	#scale(x, clims) = (x - clims[1])/(clims[2] - clims[1])
	#p = plot!(sin.(0:0.02:2π), cos.(0:0.02:2π), color=:black, label=nothing, linecolor=get.(Ref(cgrad(cmap)), scale.(g.(Point.(sin.(0:0.02:2π), cos.(0:0.02:2π))), Ref(clims))), linewidth=3)
	xlabel!(L"x")
	ylabel!(L"y")
	xlims!(-1.1, 1.1)
	ylims!(-1.1, 1.1)
	return p
end

function solve_forward(f, p)
    set_params!(f, p)
    solutions = Matrix{Float64}(undef, num_free_dofs(f.U), length(f.measurement_angles))
    if f.iterative
        solver = Krylov.GmresSolver(f.A, @view(f.b[:, 1]))
        Pℓ = ilu(f.A)
        for i in 1:length(f.measurement_angles)
            Krylov.gmres!(solver, f.A, @view(f.b[:, i]), M=Pℓ, ldiv=true)
            solutions[:, i] .= -solver.x
        end
    else
        ps = MKLPardisoSolver()
        Pardiso.solve!(ps, solutions, f.A, -f.b)
    end
    return [FEFunction(f.U, solutions[:, i]) for i in 1:length(f.measurement_angles)]
end

function solve_adjoint(f, p)
    set_params!(f, p)
    solutions = Matrix{Float64}(undef, num_free_dofs(f.U), length(f.extraction_locations))
    if f.iterative
        solver = Krylov.GmresSolver(f.AT, @view(f.c[:, 1]))
        Pℓ = ilu(f.AT)
        for i in 1:length(f.extraction_locations)
            Krylov.gmres!(solver, f.AT, @view(f.c[:, i]), M=Pℓ, ldiv=true)
            solutions[:, i] .= -solver.x
        end
    else
        ps = MKLPardisoSolver()
        Pardiso.solve!(ps, solutions, f.AT, -f.c)
    end
    return [FEFunction(f.U, solutions[:, i]) for i in 1:length(f.extraction_locations)]
end

function solve_tangent(f, λ_j, dot_m)
    rhs = assemble_linear((v, pars...) -> dot_a_(v, λ_j, nothing, dot_m, pars), f.pars, f.U, f.V)
    dot_λ_j =  f.AT \ -rhs
    return FEFunction(f.V, dot_λ_j)
end


# function measure_forward_fast(f::ModelFunction, p)
#     set_params!(f, p)
#     # solutions = zeros(num_free_dofs(f.U), length(f.measurement_angles))
#     measurements = Matrix{Float64}(undef, length(f.measurement_angles), length(f.extraction_locations))
#     solutions = Matrix{Float64}(undef, num_free_dofs(f.U), length(f.measurement_angles))
#     ps = MKLPardisoSolver()
#     Pardiso.solve!(ps, solutions, f.A, -f.b)
#     mul!(measurements, solutions', f.c)
#     return measurements
# end

function _measure_forward_pardiso!(measurements, f::ModelFunction)
    solutions = Matrix{Float64}(undef, num_free_dofs(f.U), length(f.measurement_angles))
    ps = MKLPardisoSolver()
    Pardiso.solve!(ps, solutions, f.A, -f.b)
    mul!(measurements, solutions', f.c)
end

function _measure_forward_get_solver_and_precond(f::ModelFunction)
    return Krylov.GmresSolver(f.A, @view(f.b[:, 1])), ilu(f.A)
end

function _measure_forward_gmres_precond!(measurements, f::ModelFunction, (solver, Pℓ))
    for i in 1:length(f.measurement_angles)
        Krylov.gmres!(solver, f.A, @view(f.b[:, i]), M=Pℓ, ldiv=true)
        # @show solver.stats
        sol = solver.x
        # solutions[:, i] .= sol
        mul!(@view(measurements[i:i, :]), sol', f.c, -1.0, false)
    end
end

function measure_forward(f::ModelFunction, p)
    set_params!(f, p)
    measurements = Matrix{Float64}(undef, length(f.measurement_angles), length(f.extraction_locations))
    if f.iterative
        solv_and_precon = _measure_forward_get_solver_and_precond(f)
        _measure_forward_gmres_precond!(measurements, f, solv_and_precon)
    else
        _measure_forward_pardiso!(measurements, f)
    end
    return measurements
end

function _measure_adjoint_pardiso!(measurements, solutions_adjoint, f)
    ps = MKLPardisoSolver()
    Pardiso.solve!(ps, solutions_adjoint, f.AT, -f.c)
    mul!(measurements, f.b', solutions_adjoint)
end

# computes the tangent variables \lambda_dot of the adjoint forward algorithm
function tangent_adjoint(f, p)
    set_params!(f, p)

    ps = MKLPardisoSolver()
    solutions_adjoint = Matrix{Float64}(undef, num_free_dofs(f.U), length(f.extraction_locations))

    Pardiso.solve!(ps, solutions_adjoint, f.AT, -f.c)
    measurements_tangent = Array{Float64}(undef, length(f.measurement_angles), length(f.extraction_locations), num_free_dofs(f.M))
    #mul!(measurements, f.b', solutions_adjoint)
    for j in 1:length(f.extraction_locations)
        λ_j = FEFunction(f.V, solutions_adjoint[:, j])
        dot_m = FEFunction(f.M, zeros(num_free_dofs(f.M)))
        for i in 1:num_free_dofs(f.M) #  |> num_free_dofs)
            dot_m.free_values[i] = 1.0
            # @show λ_j, dot_m
            rhs = assemble_linear((v, pars...) -> dot_a_(v, λ_j, nothing, dot_m, pars), f.pars, f.U, f.V)
            dot_λ_j =  f.AT \ -rhs
            dot_m.free_values[i] = 0.0
            mul!(@view(measurements_tangent[:, j, i]), f.b', dot_λ_j)
        end
    end
    return measurements_tangent
end

function _measure_adjoint_get_solver_and_precond(f::ModelFunction)
    return Krylov.GmresSolver(f.AT, @view(f.c[:, 1])), ilu(f.AT)
end

function _measure_adjoint_gmres_precond!(measurements, f, (solver, Pℓ), storage=nothing)
    for i in 1:length(f.extraction_locations)
        Krylov.gmres!(solver, f.AT, @view(f.c[:, i]), M=Pℓ, ldiv=true)
        # @show solver.stats
        sol_ = solver.x
        if !isnothing(storage)
            storage[:, i] .= -sol_
        end
        # @show log
        mul!(@view(measurements[:, i:i]), f.b', sol_, -1.0, 0.0)
    end
end


function (f::ModelFunction)(p)
    set_params!(f, p)
    measurements = Matrix{Float64}(undef, length(f.measurement_angles), length(f.extraction_locations))
    #solve
    if f.iterative
        solv_and_precon = _measure_adjoint_get_solver_and_precond(f)
        _measure_adjoint_gmres_precond!(measurements, f, solv_and_precon)
    else
        solutions_adjoint = Matrix{Float64}(undef, num_free_dofs(f.U), length(f.extraction_locations))
        _measure_adjoint_pardiso!(measurements, solutions_adjoint, f)
    end
    return measurements
end

Zygote.@adjoint function (f::ModelFunction)(p)
    set_params!(f, p)
    # allocate
    solutions_adjoint = Matrix{Float64}(undef, num_free_dofs(f.U), length(f.extraction_locations))
    measurements = Matrix{Float64}(undef, length(f.measurement_angles), length(f.extraction_locations))
    #solve
    if f.iterative
        solv_and_precon = _measure_adjoint_get_solver_and_precond(f)
        _measure_adjoint_gmres_precond!(measurements, f, solv_and_precon, solutions_adjoint)

        # solver = Krylov.GmresSolver(f.AT, @view(f.c[:, 1]))
        # if f.precondition
        #     Pℓ = ilu(f.AT)
        #     for i in 1:length(f.extraction_locations)
        #         Krylov.gmres!(solver, f.AT, @view(f.c[:, i]), M=Pℓ, ldiv=true)
        #         #@show solver.stats
        #         sol_ = solver.x
        #         solutions_adjoint[:, i] .= -sol_
        #         # @show log
        #         mul!(@view(measurements[:, i:i]), f.b', sol_, -1.0, 0.0)
        #     end
        # else

        # end
    else
        _measure_adjoint_pardiso!(measurements, solutions_adjoint, f)
    end
    function f_pullback(measurements_) # measurements_ = \bar{Sigma}
        source_adjoint = f.b * measurements_
        solutions_adjoint_gradient = Matrix{Float64}(undef, num_free_dofs(f.U), length(f.extraction_locations))
        if f.iterative
            solver, Pℓ = _measure_forward_get_solver_and_precond(f)
            for i in 1:length(f.extraction_locations)
                Krylov.gmres!(solver, f.A, @view(source_adjoint[:, i]), M=Pℓ, ldiv=true)
                # @show solver.stats
                sol = solver.x
                solutions_adjoint_gradient[:, i] .= -sol
                # solutions[:, i] .= sol
            end
        else
            ps = MKLPardisoSolver()
            Pardiso.solve!(ps, solutions_adjoint_gradient, f.A, -source_adjoint)
        end
        # Pardiso.solve!(ps, solutions_adjoint_gradient, f.A, -source_adjoint)
        f.temp_storage .= solutions_adjoint_gradient
        
        # M_ = TrialFESpace(f.M)
        grad_vals = zeros(num_free_dofs(f.M))
        dot_a = zeros(num_free_dofs(f.M))

        for i in 1:length(f.extraction_locations)
            # bar_u = FEFunction(f.U, solutions_adjoint_gradient[:, i])
            # λ = FEFunction(f.U, solutions_adjoint[:, i])

            # op = AffineFEOperator(
            #     (u_m, v_m) -> ∫(u_m*v_m)f.pars.dΩ, 
            #     v_m -> dot_a_(bar_u, λ, f.m, v_m, f.pars),
            #     #v_m -> dot_a(f.pp)(bar_u, λ, v_m, pars),
            #     M_, f.M)
            # grad = Gridap.solve(op)
            # global Atest = Gridap.get_matrix(op)
            #dot_a = assemble_linear(v_m -> dot_a_(bar_u, λ, f.m, v_m, f.pars), (), f.M, f.M)
            backproject!(dot_a, f.backr, @view(solutions_adjoint_gradient[:, i]), @view(solutions_adjoint[:, i]))
            # grad = 
            # @show maximum(abs.(dot_a .- dot_a2))
            # @show grad
            # @show grad2
            # @show maximum(abs.(grad.free_values .- grad2))
            # grad_vals += f.I_M .* dot_a
            grad_vals += dot_a
        end
        return (nothing, grad_vals, )
    end
    return measurements, f_pullback
end

function compute_angle((x, y))
    θ = atan(y, x)
    return θ < 0 ? 2π + θ : θ
end

function ids_of_boundary_basis_sorted(Ω)
    Γ = BoundaryTriangulation(Ω.model)
    id_list = Int64[]
    # collect all node-ids of the boundary triangulation
    for basis_id in Γ.trian.grid.cell_to_parent_cell
        push!(id_list, Γ.trian.grid.parent.cell_node_ids[basis_id]...)
    end
    # remove duplicates
    id_list = unique(id_list)
    # sort by angle
    θs = compute_angle.(getindex.(Ref(Ω.grid.node_coordinates), id_list))
    perm = sortperm(θs)
    return θs[perm], id_list[perm]
end

function ids_of_boundary_cells_sorted(Ω)
    id_list = Int64[]
    θs = Float64[]
    for (i, face3s) in enumerate(Ω.model.grid_topology.n_m_to_nface_to_mfaces[2, 3])
        if length(face3s) == 1
            push!(id_list, face3s[1])
            vertex_idxs = Ω.model.grid_topology.n_m_to_nface_to_mfaces[2, 1][i]
            midpoint = mean(getindex.(Ref(Ω.model.grid.node_coordinates), vertex_idxs))
            push!(θs, compute_angle(midpoint))
        end
    end
    perm = sortperm(θs)
    return θs[perm], id_list[perm]
end

function project_to_boundary(f, adj_sols)
    θs, bnd_cell_ids = ids_of_boundary_cells_sorted(f.Ω)
    a_id(u, v, (dΩ, dΓ, n, α_h)) = ∫(u*v)dΓ
    # M is a zeroth order FE space (we use it here for projection)
    g_hats = zeros(length(bnd_cell_ids), length(adj_sols))
    for (i, adj_sol) in enumerate(adj_sols)
        op = AffineFEOperator((u, v) -> a_id(u, v, f.pars), v -> b_(adj_sol, nothing, v, f.pars), f.M, f.M)
        A = Gridap.get_matrix(op)[bnd_cell_ids, bnd_cell_ids]
        b = Gridap.get_vector(op)[bnd_cell_ids]
        g_hats[:, i] .= A\b
    end
    return θs, g_hats
end

import Optim: common_trace!
# quick hack to make adam run.
function common_trace!(tr, d, state, iteration, method::Optim.FirstOrderOptimizer, options, curr_time=time())
    dt = Dict()
    dt["time"] = curr_time
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(Optim.gradient(d))
        #dt["Current step size"] = state.alpha
    end
    g_norm = maximum(abs, Optim.gradient(d))
    Optim.update!(tr,
            iteration,
            Optim.value(d),
            g_norm,
            dt,
            options.store_trace,
            options.show_trace,
            options.show_every,
            options.callback)
end