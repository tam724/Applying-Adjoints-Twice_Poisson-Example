include("lib.jl")
include("test_problem.jl")

using ComponentArrays
using Lux
using Zygote
using Printf

struct WeightedSum <: Lux.LuxCore.AbstractExplicitLayer end
Lux.initialparameters(::AbstractRNG, layer::WeightedSum) = (; )
Lux.parameterlength(::WeightedSum) = 0
Lux.statelength(::WeightedSum) = 3
Lux.initialstates(::AbstractRNG, layer::WeightedSum) = (vals = Float32[0.1 0.4 0.9], )
(l::WeightedSum)(x, ps, st) = st.vals * x, st

struct NNTrans{MO, ST, LOC}
    model::MO
    st::ST
    xy::LOC
end

function create_NNTrans(f, rng)
    # x_coords = [x[1] for x = mean.(Gridap.get_cell_coordinates(f.M.fe_basis.trian.grid))]
    # y_coords = [x[2] for x = mean.(Gridap.get_cell_coordinates(f.M.fe_basis.trian.grid))]
    # x_coords = rand(2972)
    # y_coords = rand(2972)
    # xy = Float32.(Matrix(hcat([[0.5*(x_coords[i]+1), 0.5*(y_coords[i]+1)] for i in 1:length(x_coords)]...)))
    
    # xy = Float32.(hcat(Vector.(Gridap.get_array.())...))
    coords = mean.(Gridap.get_cell_coordinates(f.M.fe_basis.trian.grid))
    xy = Matrix{Float64}(undef, 2, length(coords))
    for (i, coord) in enumerate(coords)
        xy[1, i] = coord[1]
        xy[2, i] = coord[2]
    end

    model = Chain(Dense(2, 40, relu), Dense(40, 40, relu), Dense(40, 3), Lux.softmax, WeightedSum())
    # model = Chain(Dense(2, 15, relu), Dense(15, 15, relu), Dense(15, 1, LuxLib.sigmoid))
    ps, st = Lux.setup(rng, model)
    # vec, re = Optimisers.destructure(ps)

    return ComponentArray(ps), NNTrans{typeof(model), typeof(st), typeof(xy)}(model, st, xy)
end

# function material(trans::NNTrans, p, x)
#     ps = trans.re(p)
#     y = first(Lux.apply(trans.model, x, ps, trans.st))
#     return Float64.(y)
# end

function (trans::NNTrans)(ps)
    # local_parameters = Float32.(p)
    # ps = trans.re(local_parameters)
    outputs = Lux.apply(trans.model, trans.xy, ps, trans.st)[1][:]
    return outputs
end

function get_weightings(trans::NNTrans, ps)
    stripped_model = trans.model[1:end-1]
    remaining_layers = [:layer_1, :layer_2, :layer_3, :layer_4]
    stripped_ps = ps[remaining_layers]
    stripped_st = trans.st[remaining_layers]
    return Lux.apply(stripped_model, trans.xy, stripped_ps, stripped_st)[1]
end

FF_true = ModelFunction(measurement_angles(), extraction_locations(), "circle_fine.msh")
FF = ModelFunction(measurement_angles(), extraction_locations(), "circle_middle.msh", iterative=true)

#another_model(x) = sqrt(1.5*x[1]^2 + 0.5*x[2]^2) < 0.4 ? 0.1 : 0.9
#another_model2(x) = sin(5.0*x[1])*0.3 + 0.5
# function another_model3(x)
#     if x > 0.0
#         return 0.1
#     else
#         return 0.9
#     end

#     # if (sqrt(1.0*(x[1]+0.5)^2 + 1.0*(x[2]-0.5)^2) < 0.5)
#     #     return 0.4
#     # elseif (sqrt(1.5*(x[1]-0.3)^2 + 0.5*(x[2]-0.3)^2) < 0.3) || (sqrt(1.0*(x[1]+0.5)^2 + 1.0*(x[2]+0.3)^2) < 0.2) 
#     #     return 0.1
#     # else 
#     #     return 0.9
#     # end
# end

true_m_pars = project_function(FF_true.M, FF_true.pars, true_m_func).free_values
true_measurements_unnoised = FF_true(true_m_pars)
# true_measurements = true_measurements_unnoised .* (1.0 .+ 0.05*randn(size(true_measurements_unnoised)))
true_measurements = true_measurements_unnoised .+ 0.01*randn(size(true_measurements_unnoised))
p_true = plot_solution(FEFunction(FF_true.M, true_m_pars), (0, 1))

rng = Lux.Random.default_rng()
Lux.Random.seed!(rng, 12345) # try different seeds (this decides the initial condition)
p, p_trans3 = create_NNTrans(FF, rng)

mean_squared_error(p) = sum((FF(p) .- true_measurements).^2) ./ length(true_measurements)
mean_squared_error_non_adjoint(p) = sum((measure_forward(FF, p) .- true_measurements).^2) ./ length(true_measurements)
transformation(p) = p_trans3(p)
objective(p) = (mean_squared_error ∘ transformation)(p)
objective_non_adjoint(p) = (mean_squared_error_non_adjoint ∘ transformation)(p)

FEFunction(FF.M, FF.I_M .* Zygote.gradient(mean_squared_error, fill(0.5, num_free_dofs(FF.M)))[1]) |> plot_solution

function objective_g!(g, p)
    g .= Float64.(Zygote.gradient(objective, p)[1])
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

#mean_squared_error_non_adjoint(fill(0.5, num_free_dofs(FF.M)))
# @benchmark finite_difference_grad($objective_non_adjoint, $p, 1e-3) # 200s
# @benchmark finite_difference_grad($(objective), $(p), $(1e-3)) # 17.2s
# @benchmark Zygote.gradient(objective, p) #19ms 

function cb(tr)
    # p = plot_solution(FEFunction(FF.M, Float64.(transformation(tr[end].metadata["x"]))), (0.0, 1.0))
    # p2 = plot()
    # for i in 1:length(extraction_locations())
    #     plot!(measurement_angles(), true_measurements[:, i], label=nothing, color=i)
    #     optimized_measurements = FF(transformation(tr[end].metadata["x"]))
    #     plot!(measurement_angles(), optimized_measurements[:, i], label=nothing, color=i)
    # end
    # display(plot!(p, p2))
    # sleep(0.01)
    @show tr[end].iteration
    return false
end


# needs 4.8s 
@time res = optimize(objective, objective_g!, p, Optim.LBFGS(), Optim.Options(callback=cb, store_trace=true, extended_trace=true, iterations=2000, time_limit=200, g_abstol=1e-6, g_reltol=1e-6))
# @profview res = optimize(objective, objective_g!, p, Optim.LBFGS(), Optim.Options(callback=cb, store_trace=true, extended_trace=true, iterations=2000, time_limit=200, g_abstol=1e-6, g_reltol=1e-6))



weightings = get_weightings(p_trans3, res.trace[end].metadata["x"])

p1 = plot_solution(FEFunction(FF.M, weightings[1, :]))
p2 = plot_solution(FEFunction(FF.M, weightings[2, :]))
p3 = plot_solution(FEFunction(FF.M, weightings[3, :]))

plot(p1, p2, p3, layout=(1, 3))

# m_opti = transformation(res.trace[end].metadata["x"])
# p_opti = plot_solution(FEFunction(FF.M, m_opti), (0, 1))
# plot(p_true, p_opti)

# trace_it = res.trace[1]
@gif for trace_it in res.trace
    @show trace_it.iteration
    p1 = plot_solution(FEFunction(FF.M, transformation(trace_it.metadata["x"])), (0, 1))
    p2 = plot()
    current_measurements = FF(transformation(trace_it.metadata["x"]))
    for i in 1:7
        plot!(measurement_angles(), true_measurements[:, i], ms=1, markerstrokewidth=0.3, color=i, alpha=0.5, label=nothing)    
        scatter!(measurement_angles(), current_measurements[:, i], ms=1, markerstrokewidth=0.3, color=i, label="j=$(i)")
    end
    p1 = plot!(p1, size=(400, 300), dpi=1000)
    p2 = plot!(p2, size=(400, 300), dpi=1000, legend_columns=2)
    plot(p1, p2, layout=grid(1,2, widths=(1/2, 1/2)), size=(800,300))
    title!("iteration $(trace_it.iteration)")
end fps=5

let # plot hidden truth material
    θs = 0:0.01:2π
    plot_solution(FEFunction(FF_true.M, true_m_pars), (0, 1), 900)

    xlims!(-1.05, 1.05)
    ylims!(-1.05, 1.05)
    plot!(cos.(θs), sin.(θs), color=:gray, label=nothing)

    scatter!(cos.(measurement_angles()[1:end]), sin.(measurement_angles()[1:end]), color=:white, label=nothing, marker=:dot, ms=1.5, markerstrokecolor=:black, markerstrokewidth=0.5)

    extr_locs = extraction_locations()
    for i in eachindex(extr_locs)
        r = extraction_radius()
        xs = [extr_locs[i][1] .+ r*cos.(θs)]
        ys = [extr_locs[i][2] .+ r*sin.(θs)]
        plot!(xs, ys, color=:black, linestyle=:dash, label=nothing)
        
        annotate!(extr_locs[i][1]-r/3, extr_locs[i][2]+r/3, ("$i", 8, :center, :black))
    end
    plot!(size=(400, 300), dpi=1000)
    savefig("figures/material.png")
end

let # plot optimized material
    θs = 0:0.01:2π
    plot_solution(FEFunction(FF.M, transformation(res.trace[end].metadata["x"])), (0, 1), 700)

    xlims!(-1.05, 1.05)
    ylims!(-1.05, 1.05)
    plot!(cos.(θs), sin.(θs), color=:gray, label=nothing)

    # plot grid
    for line in FF.pars.dΩ.quad.trian.model.grid_topology.n_m_to_nface_to_mfaces[2, 1]
        x_start, y_start =  FF.pars.dΩ.quad.trian.model.grid_topology.vertex_coordinates[line[1]]
        x_end, y_end =  FF.pars.dΩ.quad.trian.model.grid_topology.vertex_coordinates[line[2]]
        plot!([x_start, x_end], [y_start, y_end], color=:black, label=nothing, linewidth=0.1)
    end

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
    savefig("figures/optimized.png")
end

let # plot optimized material animation
    anim = @animate for trace_it in res.trace
        @show trace_it.iteration

        θs = 0:0.01:2π

        plot_solution(FEFunction(FF.M, transformation(trace_it.metadata["x"])), (0, 1), 700)

        xlims!(-1.05, 1.05)
        ylims!(-1.05, 1.05)
        plot!(cos.(θs), sin.(θs), color=:gray, label=nothing)

        # plot grid
        for line in FF.pars.dΩ.quad.trian.model.grid_topology.n_m_to_nface_to_mfaces[2, 1]
            x_start, y_start =  FF.pars.dΩ.quad.trian.model.grid_topology.vertex_coordinates[line[1]]
            x_end, y_end =  FF.pars.dΩ.quad.trian.model.grid_topology.vertex_coordinates[line[2]]
            plot!([x_start, x_end], [y_start, y_end], color=:black, label=nothing, linewidth=0.1)
        end

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
        plot!(size=(400, 300), dpi=250)
    end fps=5
    gif(anim, "figures/mp4s/material_optimization.mp4")
    gif(anim, "figures/gifs/material_optimization.gif")
end

let # plot optimized material animation
    anim = @animate for trace_it in res.trace
        @show trace_it.iteration
        p2 = plot()
        current_measurements = FF(transformation(trace_it.metadata["x"]))
        for i in 1:7
            plot!(measurement_angles(), true_measurements[:, i], ms=1, markerstrokewidth=0.3, color=i, alpha=0.5, label=nothing)    
            scatter!(measurement_angles(), current_measurements[:, i], ms=1, markerstrokewidth=0.3, color=i, label="j=$(i)")
        end
        p2 = plot!(p2, size=(400, 300), dpi=250, legend_columns=2)
        xlabel!(L"angle $\theta^{(i)}$")
        ylabel!(L"observation $\Sigma^{(ji)}$")
        ylims!(extrema(true_measurements))
        annotate!(4, 0.3, ("MSE = $(@sprintf "%.2e" trace_it.value)", 8, :center, :black))
    end fps=5
    gif(anim, "figures/mp4s/measurement_optimization.mp4")
    gif(anim, "figures/gifs/measurement_optimization.gif")
end

let # plot final observations and noisy measurements
    current_measurements = FF(transformation(res.trace[end].metadata["x"]))
    plot()
    for i in 1:7
        plot!(measurement_angles(), true_measurements[:, i], ms=1, markerstrokewidth=0.3, color=i, alpha=0.5, label=nothing)    
        scatter!(measurement_angles(), current_measurements[:, i], ms=1, markerstrokewidth=0.3, color=i, label="j=$(i)")
    end
    plot!(size=(400, 300), dpi=1000, legend_columns=2)
    xlabel!(L"angle $\theta^{(i)}$")
    ylabel!(L"observations $\Sigma^{(ji)}$")
    savefig("figures/optimized_observations.png")
end

let # plot objective function
    scatter([it.value for it in res.trace], yaxis=:log, label=nothing)
    xlabel!("iteration")
    ylabel!("MSE")
    plot!(size=(400, 300), dpi=1000)
    savefig("figures/MSE.png")
end
