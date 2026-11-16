# Samplers built on `rand` alone, whose stream `StableRNG` keeps fixed across Julia versions.

rexp(rng) = -log(rand(rng))

"A Poisson variable with mean `λ`."
function rpoisson(rng, λ)
    k = 0
    t = rexp(rng)
    while t < λ
        k += 1
        t += rexp(rng)
    end
    return k
end

"A gamma variable with integer shape `k` and mean one."
rgamma(rng, k::Integer) = sum(rexp(rng) for _ in 1:k) / k

"A standard normal variable."
rnormal(rng) = sqrt(2 * rexp(rng)) * cos(2π * rand(rng))

"An `m × n` matrix of positive expected values with rank `r` and unequal row and column sums."
function lowrank(rng, m, n, r)
    return (rand(rng, m, r) * rand(rng, r, n)) .* (0.2 .+ 3 .* rand(rng, m)) .* (0.2 .+ 3 .* rand(rng, n))'
end

"An `m × r` matrix with orthonormal columns."
orthonormal(rng, m, r) = Matrix(qr([rnormal(rng) for _ in 1:m, _ in 1:r]).Q)
