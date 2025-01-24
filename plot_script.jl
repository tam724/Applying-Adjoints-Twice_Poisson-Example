include("lib.jl")
include("test_problem.jl")

mkpath("figures")
mkpath("figures/mp4s")
mkpath("figures/gifs")
mkpath("figures/tangent_gradient")

# plot material
let
    FF_true = ModelFunction(measurement_angles(), extraction_locations(), "circle_fine.msh")
    FF = ModelFunction(measurement_angles(), extraction_locations(), "circle_middle.msh")
    true_m_pars = project_function(FF_true.M, FF_true.pars, true_m_func).free_values

    θs = 0:0.01:2π
    plot_solution(FEFunction(FF_true.M, true_m_pars), (0, 1))

    xlims!(-1.05, 1.05)
    ylims!(-1.05, 1.05)
    plot!(cos.(θs), sin.(θs), color=:gray, label=nothing)

    # plot grid
    # for line in FF.pars.dΩ.quad.trian.model.grid_topology.n_m_to_nface_to_mfaces[2, 1]
    #     x_start, y_start =  FF.pars.dΩ.quad.trian.model.grid_topology.vertex_coordinates[line[1]]
    #     x_end, y_end =  FF.pars.dΩ.quad.trian.model.grid_topology.vertex_coordinates[line[2]]
    #     plot!([x_start, x_end], [y_start, y_end], color=:black, label=nothing, linewidth=0.1)
    # end

    scatter!(cos.(measurement_angles()[1:end]), sin.(measurement_angles()[1:end]), color=:white, label=nothing, marker=:dot, ms=1.5, markerstrokecolor=:black, markerstrokewidth=0.5)

    extr_locs = extraction_locations()
    for i in eachindex(extr_locs)
        r = extraction_radius()
        xs = [extr_locs[i][1] .+ r*cos.(θs)]
        ys = [extr_locs[i][2] .+ r*sin.(θs)]
        plot!(xs, ys, color=:black, linestyle=:dash, label=nothing)
        
        annotate!(extr_locs[i][1]-r/3, extr_locs[i][2]+r/3, ("$i", 8, :center, :black))
    #    text!()
    end

    
    plot!(size=(400, 300), dpi=1000)
    savefig("figures/material.png")
end


## plot main figure
let
    FF_true = ModelFunction(measurement_angles(), extraction_locations(), "circle_fine.msh", iterative=false)
    true_m_pars = project_function(FF_true.M, FF_true.pars, true_m_func).free_values

    sols = solve_forward(FF_true, true_m_pars)

    #plot(aspect_ratio=:equal, size=(100, 200))
    θs = 0:0.01:2π
    plot_solution(sols[1], (0, 1))
    for i in reverse(1:10)
        rs = [1.0 + 0.3*excitation_func(FF_true.measurement_angles[i])([cos(θ), sin(θ)]) for θ ∈ θs]

        xs = [rs[i]*cos(θ) for (i, θ) ∈ enumerate(θs)]
        ys = [rs[i]*sin(θ) for (i, θ) ∈ enumerate(θs)]
        if i == 1
            plot!(xs, ys, color=:black, linestyle=:dash, label=nothing)
        else
            plot!(xs, ys, color=:gray, linestyle=:dash, label=nothing)
        end
    end
    xlims!(-1.05, 1.35)
    ylims!(-1.05, 1.05)
    plot!(cos.(θs), sin.(θs), color=:gray, label=nothing)


    scatter!(cos.(FF_true.measurement_angles[2:end]), sin.(FF_true.measurement_angles[2:end]), color=:white, label=nothing, marker=:dot, ms=1.5, markerstrokecolor=:black, markerstrokewidth=0.5)
    scatter!(cos.(FF_true.measurement_angles[1:1]), sin.(FF_true.measurement_angles[1:1]), color=:black, label=nothing, marker=:dot, ms=1.5, markerstrokecolor=:black, markerstrokewidth=0.5)

    for i in 1:length(FF_true.extraction_locations)
        r = extraction_radius()
        xs = [FF_true.extraction_locations[i][1] .+ r*cos.(θs)]
        ys = [FF_true.extraction_locations[i][2] .+ r*sin.(θs)]
        plot!(xs, ys, color=:white, linestyle=:dash, label=nothing)
        
        annotate!(FF_true.extraction_locations[i][1]-r/3, FF_true.extraction_locations[i][2]+r/3, ("$i", 8, :center, :white))
    #    text!()
    end
    plot!(size=(400, 300), dpi=1000)
    savefig("figures/forward_main.png")
end

# plot forward animation
let
    FF_true = ModelFunction(measurement_angles(), extraction_locations(), "circle_fine.msh", iterative=false)
    true_m_pars = project_function(FF_true.M, FF_true.pars, true_m_func).free_values

    sols = solve_forward(FF_true, true_m_pars)

    #plot(aspect_ratio=:equal, size=(100, 200))
    θs = 0:0.01:2π
    anim = @animate for i in 1:length(FF_true.measurement_angles)
        plot_solution(sols[i], (0, 1))
        for Δi_bc in reverse(0:9)
            i_bc = i - Δi_bc
            if i_bc <= 0
                i_bc = i_bc + length(FF_true.measurement_angles)
            end
            rs = [1.0 + 0.3*excitation_func(FF_true.measurement_angles[i_bc])([cos(θ), sin(θ)]) for θ ∈ θs]

            xs = [rs[i]*cos(θ) for (i, θ) ∈ enumerate(θs)]
            ys = [rs[i]*sin(θ) for (i, θ) ∈ enumerate(θs)]
            if i_bc == i
                plot!(xs, ys, color=:black, linestyle=:dash, label=nothing)
            else
                plot!(xs, ys, color=:gray, linestyle=:dash, label=nothing, alpha=range(1, 0, 10)[Δi_bc+1])
            end
        end
        xlims!(-1.35, 1.35)
        ylims!(-1.35, 1.35)
        plot!(cos.(θs), sin.(θs), color=:gray, label=nothing)

        scatter!(cos.(FF_true.measurement_angles[[1:i-1..., i+1:end...]]), sin.(FF_true.measurement_angles[[1:i-1..., i+1:end...]]), color=:white, label=nothing, marker=:dot, ms=1.5, markerstrokecolor=:black, markerstrokewidth=0.5)
        scatter!(cos.(FF_true.measurement_angles[i:i]), sin.(FF_true.measurement_angles[i:i]), color=:black, label=nothing, marker=:dot, ms=1.5, markerstrokecolor=:black, markerstrokewidth=0.5)
        for i in 1:length(FF_true.extraction_locations)
            r = extraction_radius()
            xs = [FF_true.extraction_locations[i][1] .+ r*cos.(θs)]
            ys = [FF_true.extraction_locations[i][2] .+ r*sin.(θs)]
            plot!(xs, ys, color=:white, linestyle=:dash, label=nothing)
            
            annotate!(FF_true.extraction_locations[i][1]-r/3, FF_true.extraction_locations[i][2]+r/3, ("$i", 8, :center, :white))
        #    text!()
        end
        plot!(size=(400, 300), dpi=250)
    end
    gif(anim, "figures/mp4s/forward.mp4")
    gif(anim, "figures/gifs/forward.gif")
end

# forward with trace
let
    FF_true = ModelFunction(measurement_angles(), extraction_locations(), "circle_fine.msh", iterative=false)
    true_m_pars = project_function(FF_true.M, FF_true.pars, true_m_func).free_values

    sols = solve_forward(FF_true, true_m_pars)

    #plot(aspect_ratio=:equal, size=(100, 200))
    θs = 0:0.01:2π
    for i in [1, 51, 101, 151]
        plot_solution(sols[i], (0, 1))
        for Δi_bc in reverse(0:9)
            i_bc = i - Δi_bc
            if i_bc <= 0
                i_bc = i_bc + length(FF_true.measurement_angles)
            end
            rs = [1.0 + 0.3*excitation_func(FF_true.measurement_angles[i_bc])([cos(θ), sin(θ)]) for θ ∈ θs]

            xs = [rs[i]*cos(θ) for (i, θ) ∈ enumerate(θs)]
            ys = [rs[i]*sin(θ) for (i, θ) ∈ enumerate(θs)]
            if i_bc == i
                plot!(xs, ys, color=:black, linestyle=:dash, label=nothing)
            else
                plot!(xs, ys, color=:gray, linestyle=:dash, label=nothing, alpha=range(1, 0, 10)[Δi_bc+1])
            end
        end
        xlims!(-1.35, 1.35)
        ylims!(-1.35, 1.35)
        plot!(cos.(θs), sin.(θs), color=:gray, label=nothing)

        scatter!(cos.(FF_true.measurement_angles[[1:i-1..., i+1:end...]]), sin.(FF_true.measurement_angles[[1:i-1..., i+1:end...]]), color=:white, label=nothing, marker=:dot, ms=1.5, markerstrokecolor=:black, markerstrokewidth=0.5)
        scatter!(cos.(FF_true.measurement_angles[i:i]), sin.(FF_true.measurement_angles[i:i]), color=:black, label=nothing, marker=:dot, ms=1.5, markerstrokecolor=:black, markerstrokewidth=0.5)
        for i in 1:length(FF_true.extraction_locations)
            r = extraction_radius()
            xs = [FF_true.extraction_locations[i][1] .+ r*cos.(θs)]
            ys = [FF_true.extraction_locations[i][2] .+ r*sin.(θs)]
            plot!(xs, ys, color=:white, linestyle=:dash, label=nothing)
            
            annotate!(FF_true.extraction_locations[i][1]-r/3, FF_true.extraction_locations[i][2]+r/3, ("$i", 8, :center, :white))
        #    text!()
        end
        plot!(size=(400, 300), dpi=1000)
        savefig("figures/forward_trace_$(i).png")
    end
end

# plot additional forward figures
let
    FF_true = ModelFunction(measurement_angles(), extraction_locations(), "circle_fine.msh", iterative=false)

    true_m_pars = project_function(FF_true.M, FF_true.pars, true_m_func).free_values
    sols = solve_forward(FF_true, true_m_pars)

    for n_sol ∈ [51, 101, 151]
        n_sol = 51
        plot(aspect_ratio=:equal)
        θs = 0:0.01:2π
        plot_solution(sols[n_sol], (0, 1))

        xlims!(-1.05, 1.05)
        ylims!(-1.05, 1.05)
        plot!(cos.(θs), sin.(θs), color=:gray, label=nothing)

        p1 = plot!([1.0, 0.0, cos(FF_true.measurement_angles[n_sol])], [0.0, 0.0, sin(FF_true.measurement_angles[n_sol])], color=:white, label=nothing)
        p1 = plot!(size=(480, 480))
        p1 = plot!(0.1.*cos.(0:0.01:FF_true.measurement_angles[n_sol]), 0.1.*sin.(0:0.01:FF_true.measurement_angles[n_sol]), color=:white, label=nothing)
        p1 = annotate!(0.2*cos(FF_true.measurement_angles[n_sol] / 2.0), 0.2*sin(FF_true.measurement_angles[n_sol] / 2.0), (L"\theta", 8, :white), :gray, label=nothing)
        plot!(right_margin = 5Plots.mm)

        scatter!(cos.(FF_true.measurement_angles[1:end]), sin.(FF_true.measurement_angles[1:end]), color=:white, label=nothing, marker=:dot, ms=1.5, markerstrokecolor=:black, markerstrokewidth=0.5)
        scatter!(cos.(FF_true.measurement_angles[n_sol:n_sol]), sin.(FF_true.measurement_angles[n_sol:n_sol]), color=:black, label=nothing, marker=:dot, ms=1.5, markerstrokecolor=:black, markerstrokewidth=0.5)

        plot!(size=(400, 300), dpi=1000)
        savefig("figures/forward_additional_$(n_sol).png")
    end
end

# plot measurements
let
    FF_true = ModelFunction(measurement_angles(), extraction_locations(), "circle_fine.msh")
    true_m_pars = project_function(FF_true.M, FF_true.pars, true_m_func).free_values

    measurements = FF_true(true_m_pars)
    plot()
    for i in 1:length(FF_true.extraction_locations)
        scatter!(FF_true.measurement_angles, measurements[:, i], label="j=$(i)", ms=1, markerstrokewidth=0.3, color=i)
    end
    xlabel!(L"angle $\theta^{(i)}$")
    ylabel!(L"observation $\Sigma^{(ji)}$")
    plot!(size=(400, 300), dpi=1000, legend_columns=2)
    savefig("figures/measurements.png")
end

# plot measurements extra
let
    FF_true = ModelFunction(measurement_angles(), extraction_locations(), "circle_fine.msh")
    true_m_pars = project_function(FF_true.M, FF_true.pars, true_m_func).free_values

    measurements = FF_true(true_m_pars)
    plot()
    vline!([FF_true.measurement_angles[51]], color=:gray, label=nothing, alpha=0.8)

    for i in 1:length(FF_true.extraction_locations)
        scatter!(FF_true.measurement_angles, measurements[:, i], label=nothing, ms=1, color=i, markerstrokewidth=0.0, alpha=0.3)
    end
    scatter!(FF_true.measurement_angles[:], measurements[:, 1], label=nothing, ms=1, markerstrokewidth=0.3, color=1)
    for i in 1:length(FF_true.extraction_locations)
        scatter!(FF_true.measurement_angles[51:51], measurements[51:51, i], label="j=$(i)", ms=2, markerstrokewidth=0.3, color=i)
    end

    ylims!(extrema(measurements))
    xlabel!(L"angle $\theta^{(i)}$")
    ylabel!(L"observation $\Sigma^{(ji)}$")
    plot!(size=(400, 300), dpi=1000, legend_columns=2)
    savefig("figures/measurements_extra_51.png")
end

let
    FF_true = ModelFunction(measurement_angles(), extraction_locations(),  "circle_fine.msh")
    FF_true_higher_order = ModelFunction(measurement_angles(), extraction_locations(), "circle_fine.msh"; order=2)
    
    # this is the same, no matter the order of U or V
    true_m_pars = project_function(FF_true.M, FF_true.pars, true_m_func).free_values

    # lets check if the models agree in the measurements..
    @show maximum(abs.(FF_true(true_m_pars) .- FF_true_higher_order(true_m_pars)))

    adj_sols = solve_adjoint(FF_true_higher_order, true_m_pars)
    θs, g_hats = project_to_boundary(FF_true_higher_order, adj_sols)


    # check if the measurements of the projections agree with the "original" measurements
    using Interpolations
    using HCubature
    g_hat_1_func = Interpolations.linear_interpolation(θs, g_hats[:, 1])

    @show hquadrature(θ -> excitation_func(FF_true.measurement_angles[130])((cos(θ), sin(θ))) * g_hat_1_func(θ), θs[1], θs[end])[1], FF_true_higher_order(true_m_pars)[130, 1]

    # plotting
    plot()
    for i in 3:2:20
        plot!(θs, θ -> excitation_func(FF_true.measurement_angles[i])((cos(θ), sin(θ))), color=:lightgray, linestyle=:dash, label=nothing)
    end
    plot!(θs, θ -> excitation_func(FF_true.measurement_angles[1])((cos(θ), sin(θ))), color=:black, linestyle=:dash, label=nothing)
    
    for i in 1:length(FF_true.extraction_locations)
        plot!(θs, g_hats[:, i], color=i, label="j=$(i)")
    end
    xlabel!(L"angle $\theta$")
    ylabel!(L"riesz representation $\nabla_n  \lambda^{(j)}$")

    plot!(size=(400, 300), dpi=1000, legend_columns=2, legend=:top)
    savefig("figures/boundary_projections.png")
end

# plot extraction
let
    FF_true = ModelFunction(measurement_angles(), extraction_locations(), "circle_fine.msh")
    true_m_pars = project_function(FF_true.M, FF_true.pars, true_m_func).free_values

    sols_adjoint = solve_adjoint(FF_true, true_m_pars)
    θs = 0:0.01:2π

    for n_sol in 1:length(FF_true.extraction_locations)
        plot_solution(sols_adjoint[n_sol], extrema(sols_adjoint[n_sol].free_values), 200, true)
        for i in 1:length(FF_true.extraction_locations)
            r = extraction_radius()
            xs = [FF_true.extraction_locations[i][1] .+ r*cos.(θs)]
            ys = [FF_true.extraction_locations[i][2] .+ r*sin.(θs)]
            clr = nothing
            if i == n_sol
                clr = :black
            else
                clr = :white
            end
            plot!(xs, ys, color=clr, linestyle=:dash, label=nothing)
            
            annotate!(FF_true.extraction_locations[i][1]-r/3, FF_true.extraction_locations[i][2]+r/3, ("$i", 8, :center, clr))
        end
        plot!(size=(400, 300), dpi=1000)
        savefig("figures/adjoint_forward_$(n_sol).png")
    end
end

let 
    FF = ModelFunction(measurement_angles(), extraction_locations(), "circle_middle.msh", iterative=true)
    true_m_pars = project_function(FF.M, FF.pars, true_m_func).free_values

    # benchmark setup (measurement calculation)
    set_params!(FF, true_m_pars)
    measurements_forward = Matrix{Float64}(undef, length(FF.measurement_angles), length(FF.extraction_locations))
    measurements_adjoint = Matrix{Float64}(undef, length(FF.measurement_angles), length(FF.extraction_locations))

    solv_and_precon_forward = _measure_forward_get_solver_and_precond(FF)
    solv_and_precon_adjoint = _measure_adjoint_get_solver_and_precond(FF)

    # benchmark (without setup)
    trial_non_adjoint_without_setup = @benchmark _measure_forward_gmres_precond!($measurements_forward, $FF, $solv_and_precon_forward)
    trial_adjoint_without_setup = @benchmark _measure_adjoint_gmres_precond!($measurements_adjoint, $FF, $solv_and_precon_adjoint)

    # benchmark (with setup, preconditioner, matrix-build, etc)
    trial_non_adjoint_with_setup = @benchmark measure_forward($FF, $true_m_pars)
    trial_adjoint_with_setup = @benchmark ($FF)($true_m_pars)

    @show "non_adjont (without setup)"
    display(trial_non_adjoint_without_setup)
    @show "adjoint (without setup)"
    display(trial_adjoint_without_setup)

    @show "non_adjoint (with setup)"
    display(trial_non_adjoint_with_setup)
    @show "adjoint (with setup)"
    display(trial_adjoint_with_setup)

    @show maximum(abs.(measurements_adjoint .- measurements_forward))
end

function finite_difference_grad(f, p, h)
    val = f(p)
    grad = similar(p)
    for i in eachindex(p)
        # @show i, length(p)
        p[i] += h
        grad[i] = (f(p) - val)/h
        p[i] -= h
    end
    return grad
end

let # gradient computation benchmarks 
    FF = ModelFunction(measurement_angles(), extraction_locations(), "circle_middle.msh", iterative=true)
    true_m_pars = project_function(FF.M, FF.pars, true_m_func).free_values
    true_measurements = FF(true_m_pars)
    mean_squared_error(p) = sum((FF(p) .- true_measurements).^2) # ./ length(true_measurements)
    # do not use this ! (too expensive)
    mean_squared_error_forward(p) = sum((measure_forward(FF, p) .- true_measurements).^2) #./ length(true_measurements)
    p0 = fill(0.5, size(true_m_pars))

    trial_FD_adjoint = @benchmark finite_difference_grad($(mean_squared_error), $(p0), $(1e-6))
    trial_FD_non_adjoint = @benchmark finite_difference_grad($(mean_squared_error_forward), $(p0), $(1e-6))
    trial_adjoint_adjoint = @benchmark Zygote.gradient($(mean_squared_error), $(p0))

    @show "FD_adjoint"
    display(trial_FD_adjoint)

    @show "FD_non_adjoint"
    display(trial_FD_non_adjoint)

    @show "adjoint_adjoint"
    display(trial_adjoint_adjoint)
end

# let # taylor remainder test
    FF = ModelFunction(measurement_angles(), extraction_locations(), "circle_middle.msh", iterative=false; order=1)
    true_m_pars = project_function(FF.M, FF.pars, true_m_func).free_values
    true_measurements = FF(true_m_pars)
    mean_squared_error(p) = sum((FF(p) .- true_measurements).^2) # ./ length(true_measurements)
    # do not use this ! (most expensive thing ever..)
    mean_squared_error_non_adjoint(p) = sum((measure_forward(FF, p) .- true_measurements).^2) #./ length(true_measurements)
    p0 = fill(0.5, size(true_m_pars))

    grad_FD_adjoint_05 = finite_difference_grad(mean_squared_error, p0, 1e-5)
    grad_FD_adjoint_03 = finite_difference_grad(mean_squared_error, p0, 1e-3)
    # grad_FD_non_adjoint = finite_difference_grad(mean_squared_error_non_adjoint, p0, 1e-5)
    grad_adjoint_adjoint = Zygote.gradient(mean_squared_error, p0)[1]

    # this is not optimized at all..
    # Σ_tangent = tangent_adjoint(FF, p0)
    # res = 2.0.*(FF(p0) .- true_measurements)
    # grad_non_adjoint_adjoint = zeros(num_free_dofs(FF.M))
    # for i in 1:num_free_dofs(FF.M)
    #     grad_non_adjoint_adjoint[i] = dot(Σ_tangent[:, :, i], res)
    # end

    p_perturbs = [randn(length(true_m_pars)) |> normalize for i in 1:50]
    seq1(h) = [abs(mean_squared_error(p0 .+ h.*p_perturb) - mean_squared_error(p0)) for p_perturb in p_perturbs] |> mean
    seq2(h, grad) = h.* [abs(mean_squared_error(p0 .+ h.*p_perturb) / h - mean_squared_error(p0) / h - dot(grad, p_perturb)) for p_perturb in p_perturbs] |> mean

# FEFunction(FF.M, grad1 .- grad2) |> plot_solution
# FEFunction(FF.M, grad2) |> plot_solution
# FEFunction(FF.M, grad3[1] .- grad1) |> plot_solution
# FEFunction(FF.M, grad4 .- grad1) |> plot_solution
# FEFunction(FF.M, grad3[1] .- grad4) |> plot_solution

    hs = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    plot(hs, seq1.(hs), xaxis=:log, yaxis=:log, marker=:o, label="1st rem.")
    plot!(hs, seq2.(hs, Ref(grad_adjoint_adjoint)), marker=:o, label="2nd rem. (Adjoint)")
    # plot!(hs, seq2.(hs, Ref(grad_non_adjoint_adjoint)), marker=:o, label="2nd Taylor remainder(nad_ad)")
    plot!(hs, seq2.(hs, Ref(grad_FD_adjoint_03)), marker=:o, label="2nd rem. (FD, Δ=1e-3)")
    plot!(hs, seq2.(hs, Ref(grad_FD_adjoint_05)), marker=:o, label="2nd rem. (FD, Δ=1e-5)")
    # plot!(hs, seq2.(hs, Ref(grad_FD_non_adjoint)), marker=:o, label="2nd Taylor remainder(FD_nad)")

    plot!(hs, 0.001.*hs, color=:gray, linestyle=:dash, label="1st order")
    # plot!(hs, 0.0001.*hs, color=:gray, linestyle=:dash, label=nothing)
    plot!(hs, 0.00005.*hs.^2, color=:gray, linestyle=:dashdot, label="2nd order")
    xlabel!(L"h")
    ylabel!(L"taylor remainder $\,$")
    plot!(size=(400, 300), dpi=1000, legend=:bottomright)
    savefig("figures/taylor_remainder.png")
# end


let # tangent gradient plots
    FF = ModelFunction(measurement_angles(), extraction_locations(), "circle_middle.msh")
    true_m_pars = project_function(FF.M, FF.pars, true_m_func).free_values
    set_params!(FF, true_m_pars)

    adj_sol = solve_adjoint(FF, true_m_pars)

    p_idx = rand(1:length(true_m_pars), 10)
    for i in 1:10
        p2 = plot()
        θs = 0:0.01:2π
        for n_adj_sol = 1:7
            dot_m = FEFunction(FF.M, zeros(length(true_m_pars)))
            dot_m.free_values[p_idx[i]] = 1.0
            tang_sol = solve_tangent(FF, adj_sol[n_adj_sol], dot_m)
            boundary_proj = project_to_boundary(FF, [tang_sol])
            p = plot_solution(tang_sol) #  cmap=cgrad([cgrad(:temperaturemap)[1], cgrad(:temperaturemap)[end]]))
            for i in 1:length(FF.extraction_locations)
                r = extraction_radius()
                xs = [FF.extraction_locations[i][1] .+ r*cos.(θs)]
                ys = [FF.extraction_locations[i][2] .+ r*sin.(θs)]
                clr = i == n_adj_sol ? :black : :lightgray
                plot!(xs, ys, color=clr, linestyle=:dash, label=nothing)
                
                annotate!(FF.extraction_locations[i][1]-r/3, FF.extraction_locations[i][2]+r/3, ("$i", 8, :center, clr))
            end
            plot!(size=(400, 300), dpi=1000)
            savefig("figures/tangent_gradient/gradient_tangent_$(n_adj_sol)_$(i).png")
            plot!(p2, boundary_proj[1], boundary_proj[2][:, 1])
        end
        plot!(p2, size=(400, 300), dpi=1000)
        savefig("figures/tangent_gradient/gradient_tangent_$(i)")
    end
end

let 
    FF = ModelFunction(measurement_angles(), extraction_locations(), "circle_fine.msh")
    true_m_pars = project_function(FF.M, FF.pars, true_m_func).free_values

    true_measurements = FF(true_m_pars)

    # mean_squared_error(p) = sum((FF(p) .- true_measurements).^2) ./ length(true_measurements)
    p0 = fill(0.5, size(true_m_pars))
    measurements = FF(p0)
    
    plot()
    for i in 1:length(FF.extraction_locations)
        scatter!(FF.measurement_angles, true_measurements[:, i], label="j=$(i)", ms=1, markerstrokewidth=0.2, color=i)
    end
    for i in 1:length(FF.extraction_locations)
        scatter!(FF.measurement_angles, measurements[:, i], label=nothing, ms=0.5, markerstrokewidth=0.1, alpha=0.5, color=i)
    end

    xlabel!(L"angle $\theta^{(i)}$")
    ylabel!(L"observation $\Sigma^{(ji)}$")
    plot!(size=(400, 300), dpi=1000, legend_columns=2)
    savefig("figures/measurements_difference.png")
end

let # adjoint gradient plots
    FF = ModelFunction(measurement_angles(), extraction_locations(), "circle_fine.msh")
    true_m_pars = project_function(FF.M, FF.pars, true_m_func).free_values
    true_measurements = FF(true_m_pars)

    mean_squared_error(p) = sum((FF(p) .- true_measurements).^2) ./ length(true_measurements)
    p0 = fill(0.5, size(true_m_pars))

    grad = Zygote.gradient(mean_squared_error, p0)

    # for i in 1:length(FF.extraction_locations)
    #     temp = FEFunction(FF.U, FF.temp_storage[:, i])
    #     plot_solution(temp)

    #     xlims!(-1.3, 1.35)
    #     ylims!(-1.3, 1.3)

    #     FF0 = FF(p0)

    #     rs = fill(1.1, length(FF.measurement_angles))
    #     xs = [rs[i]*cos(θ) for (i, θ) ∈ enumerate(FF.measurement_angles)]
    #     ys = [rs[i]*sin(θ) for (i, θ) ∈ enumerate(FF.measurement_angles)]

    #     # plot "coordinate system"
    #     plot!(xs[12:end], ys[12:end], color=:gray, linestyle=:dash, label=nothing)
    #     plot!(xs[1:9], ys[1:9], color=:black, linestyle=:solid, label=nothing, arrow=(:closed))
    #     plot!([1.1, 1.3], [0.0, 0.0], color=:black, linestyle=:solid, legend=nothing, arrow=(:closed))

    #     # plot measurement difference (with scatter)
    #     rs = 1.1 .+ (FF0 .- true_measurements)[:, i]
    #     xs = [rs[i]*cos(θ) for (i, θ) ∈ enumerate(FF.measurement_angles)]
    #     ys = [rs[i]*sin(θ) for (i, θ) ∈ enumerate(FF.measurement_angles)]

    #     scatter!([xs..., xs[1]], [ys..., ys[1]], color=i, linestyle=:solid, label=nothing, ms=1.5, markerstrokewidth=0.3)
    #     plot!(size=(400, 300), dpi=1000)
    #     savefig("figures/adjoint_gradient/adjoint_solution_$(i).png")
    # end

    p = plot_solution(FEFunction(FF.M, grad[1] .* FF.I_M))
    plot!(size=(400, 300), dpi=1000)
    savefig("figures/gradient.png")
end

# let # gradient plots


# end




# adj_sol = solve_adjoint(FF, true_m_pars)
# θs = 0:0.01:2π

# for i in 1:20
#     dot_m = FEFunction(FF.M, zeros(length(true_m_pars)))
#     dot_m.free_values[rand(1:length(true_m_pars))] = 1.0
#     tang_sol = solve_tangent(FF, adj_sol[7], dot_m)
#     boundary_proj = project_to_boundary(FF, [tang_sol])
#     p = plot_solution(tang_sol)
#     for i in 1:length(FF.extraction_locations)
#         r = extraction_radius()
#         xs = [FF.extraction_locations[i][1] .+ r*cos.(θs)]
#         ys = [FF.extraction_locations[i][2] .+ r*sin.(θs)]
#         plot!(xs, ys, color=:black, linestyle=:dash, label=nothing)
        
#         annotate!(FF.extraction_locations[i][1]-r/3, FF.extraction_locations[i][2]+r/3, ("$i", 8, :center, :black))
#     #    text!()
#     end
#     display(p)
#     sleep(1)
# end
# plot(boundary_proj[1], boundary_proj[2][:, 1])

# true_measurements = FF(true_m_pars)
# mean_squared_error(p) = sum((FF(p) .- true_measurements).^2) ./ length(true_measurements)

# p0 = fill(0.5, size(true_m_pars))

# grad = Zygote.gradient(mean_squared_error, p0)

# temp = FEFunction(FF.U, FF.temp_storage[:, 3])
# plot_solution(temp)

# xlims!(-1.3, 1.3)
# ylims!(-1.3, 1.3)

# FF0 = FF(p0)

# rs = 1.1 .+ (FF0 .- true_measurements)[:, 3]*0.0
# xs = [rs[i]*cos(θ) for (i, θ) ∈ enumerate(FF.measurement_angles)]
# ys = [rs[i]*sin(θ) for (i, θ) ∈ enumerate(FF.measurement_angles)]

# plot!(xs[12:end], ys[12:end], color=:black, linestyle=:dash, label=nothing)
# plot!(xs[1:9], ys[1:9], color=:black, linestyle=:dash, label=nothing, arrow=(:closed))
# plot!([1.1, 1.3], [0.0, 0.0], color=:black, linestyle=:dash, legend=nothing, arrow=(:closed))

# rs = 1.1 .+ (FF0 .- true_measurements)[:, 3]
# xs = [rs[i]*cos(θ) for (i, θ) ∈ enumerate(FF.measurement_angles)]
# ys = [rs[i]*sin(θ) for (i, θ) ∈ enumerate(FF.measurement_angles)]

# scatter!([xs..., xs[1]], [ys..., ys[1]], color=3, linestyle=:solid, label=nothing, ms=1.5, markerstrokewidth=0.3)
# #     else
# #         plot!(xs, ys, color=:gray, linestyle=:dash, label=nothing)
# #     end
# # end

# grad_fefunc = FEFunction(FF.M, grad[1])
# plot_solution(grad_fefunc)