
∇ₙ(u, n) = dot(∇(u), n)
# original bilinear form
a_(u, v, m, (dΩ, dΓ, n, α_h)) = ∫(m*dot(∇(u), ∇(v)))dΩ - ∫(m*∇ₙ(u, n)*v + u*∇ₙ(v, n))dΓ + ∫(α_h[1] * u * v)dΓ # nitsche trick (stabilization)

# the part of the bilinear form that is linear in m (for speed)
a1_(u, v, m, (dΩ, dΓ, n, α_h)) = ∫(dot(∇(u), ∇(v)))dΩ - ∫(∇ₙ(u, n)*v)dΓ
# the part of the bilinear form that is affine in m (for speed)
a2_(u, v, m, (dΩ, dΓ, n, α_h)) = ∫(0.0*dot(∇(u), ∇(v)))dΩ - ∫(0.0*∇ₙ(u, n)*v + u*∇ₙ(v, n))dΓ + ∫(α_h[1] * u * v)dΓ # nitsche trick (stabilization)


# this is the original tangent bilinear form
dot_a_(u, v, m, dot_m, (dΩ, dΓ, n, α_h)) = ∫(dot_m*dot(∇(u), ∇(v)))dΩ - ∫(dot_m*dot(∇(u), n)*v)dΓ

# for speed (because dot_a is linear in m, we can precompute the matrices)
dot_a_temp(u, v, m, dot_m, (dΩ, dΓ, n, α_h)) = ∫(dot(∇(u), ∇(v)))dΩ - ∫(dot(∇(u), n)*v)dΓ

b_(v, m, g, (dΩ, dΓ, n, α_h)) = ∫(∇ₙ(v,  n)*g)dΓ - ∫(α_h[1] * (g * v))dΓ # nitsche trick (stabilization)
bnew_(v, m, g, (dΩ, dΓ, n, α_h)) = ∫(∇ₙ(v,  n)*g)dΓ
c_(u, m, μ, (dΩ, dΓ, n, α_h)) = ∫(μ*u)dΩ

true_ellipse_parameters() = (μ1=0.0, μ2=-0.3, r=π/4, a=0.3, b=0.8)
is_in_ellipse(x, (; μ1, μ2, r, a, b)) = ((x[1] - μ1)*cos(r) + (x[2] - μ2)*sin(r))^2/a^2 + ((x[1] - μ1)*sin(r) - (x[2] - μ2)*cos(r))^2/b^2 < 1.0

function true_m_func(x)
    if is_in_ellipse(x, true_ellipse_parameters())
        return 0.9
    elseif x[2] > 0.0
        return 0.1
    else
        return 0.4
    end
end

# dirchlet boundary
measurement_angles() = range(0, 2π, length=201)[1:200]
angle(x, y) = acos(dot(x, y) / sqrt(dot(x, x) / sqrt(dot(y, y))))
excitation_func(θ) = x -> exp(-10.0*angle(x, (cos(θ), sin(θ)))^2)

# extraction function
extraction_locations() = [[(0.5*cos(θ), 0.5*sin(θ)) for θ in range(0, 2π, length=7)[1:end-1]]..., (0.0, 0.0)]
extraction_radius() = sqrt(0.03)
extraction_func((x_loc, y_loc)) = x -> (((x[1] - x_loc)^2 + (x[2] - y_loc)^2) < extraction_radius()^2) ? 1.0 / (π*extraction_radius()^2) : 0.0
