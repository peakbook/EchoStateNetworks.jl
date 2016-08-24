module EchoStateNetworks

abstract ESNArch
type NoTeacherForcing <: ESNArch
end
type TeacherForcing <: ESNArch
end

export EchoStateNetwork
export train!, predict!

type EchoStateNetwork{T<:AbstractFloat}
    Nr::Integer     # the number of reservoir neurons
    Ni::Integer     # the number of input dimension
    No::Integer     # the number of output dimension
    sparsity::T
    spectral_radius::T
    noise_level::T
    leaking_rate::T
    teacher_forcing::Bool
    Wr::Matrix{T}   # reservoir weights
    Wi::Matrix{T}   # input weights
    Wo::Matrix{T}   # readout weights
    Wf::Matrix{T}   # feedback weights
    state::Vector{T}
    input::Vector{T}
    output::Vector{T}
    rng::AbstractRNG
    function EchoStateNetwork(;Ni::Integer=1,
                              No::Integer=1,
                              Nr::Integer=100,
                              sparsity::AbstractFloat=0.95,
                              spectral_radius::AbstractFloat=0.95,
                              noise_level::AbstractFloat=0.001, 
                              leaking_rate::AbstractFloat=1.0, 
                              teacher_forcing::Bool=true,
                              rng::AbstractRNG=MersenneTwister(rand(UInt32)),
                             )
        esn = new()

        esn.Ni = Ni
        esn.No = No
        esn.Nr = Nr
        esn.sparsity = sparsity
        esn.spectral_radius = spectral_radius
        esn.teacher_forcing = teacher_forcing
        esn.noise_level = noise_level
        esn.leaking_rate = leaking_rate
        esn.rng = rng

        init_weights!(esn)

        return esn
    end
end

function init_weights!{T<:AbstractFloat}(esn::EchoStateNetwork{T})
    # init reservoir weight matrix
    esn.Wr = rand(esn.rng, T, esn.Nr, esn.Nr)-T(0.5)
    ## reduce connections based on `sparsity`
    Ns = round(Int,length(esn.Wr)*esn.sparsity)
    for i in [rand(esn.rng, 1:length(esn.Wr)) for i in 1:Ns]
        esn.Wr[i] = zero(T)
    end
    ## rescale the matrix to fit the `spectral radius`
    esn.Wr *= esn.spectral_radius/maximum(abs(eigvals(esn.Wr)))

    # init input weight matrix
    esn.Wi = rand(esn.rng, T, esn.Nr, esn.Ni+1)-T(0.5)

    # init feedback weight matrix
    esn.Wf = rand(esn.rng, T, esn.Nr, esn.No)-T(0.5)

    return 
end

function activate(f::Function, x::Array)
    return map(f, x)
end

function update{T<:AbstractFloat}(esn::EchoStateNetwork{T}, state::Vector{T}, 
                                  input::Vector{T}, output::Vector{T}, arch::NoTeacherForcing)
    return activate(tanh, esn.Wi*[one(T);input] + esn.Wr*state)
    + esn.noise_level*(rand(esn.rng, T, esn.Nr)-T(0.5))
end

function update{T<:AbstractFloat}(esn::EchoStateNetwork{T}, state::Vector{T},
                                  input::Vector{T}, output::Vector{T}, arch::TeacherForcing)
    return activate(tanh, esn.Wi*[one(T);input] + esn.Wr*state + esn.Wf*output)
    + esn.noise_level*(rand(esn.rng, T, esn.Nr)-T(0.5))
end

function train!{T<:AbstractFloat}(esn::EchoStateNetwork{T}, inputs::Matrix{T}, outputs::Matrix{T};
                                  discard::Integer=min(div(size(inputs,2),10), 100), reg::AbstractFloat=1e-8)

    @assert(size(inputs, 1) == esn.Ni)
    @assert(size(outputs, 1) == esn.No)
    @assert(size(inputs, 2) == size(outputs, 2))

    Nd = size(inputs, 2)
    states = zeros(T, (esn.Nr, Nd))
    arch = esn.teacher_forcing ? TeacherForcing() : NoTeacherForcing()
    for t = 2:size(inputs,2)
        states[:,t] = (esn.leaking_rate-one(T))*states[:,t-1] + esn.leaking_rate*update(esn, states[:,t-1], inputs[:,t], outputs[:,t-1], arch)
    end

    # extended system states
    X = zeros(1+esn.Ni+esn.Nr, Nd)
    X[1,:] = one(T)
    X[2:1+esn.Ni,:] = inputs
    X[2+esn.Ni:end,:] = states

    # discard initial transient
    Xe = X[:,discard+1:end]
    tXe = Xe'
    O = outputs[:,discard+1:end]

    # calc output weight matrix
    esn.Wo = O*tXe*pinv(Xe*tXe + reg*eye(size(Xe,1)))

    # store last states
    esn.state = states[:,end]
    esn.input = inputs[:,end]
    esn.output = outputs[:,end]

    return esn.Wo * X
end

function predict!{T<:AbstractFloat}(esn::EchoStateNetwork{T}, inputs::Matrix{T}; cont::Bool=true)
    Nd = size(inputs, 2)
    outputs = zeros(T, esn.No, Nd)
    state = zeros(T, esn.Nr)

    if cont
        inputs = hcat(esn.input, inputs)
        outputs = hcat(esn.output, outputs)
        state[:] = esn.state
    else
        inputs = hcat(zeros(T,esn.Ni), inputs)
        outputs = hcat(zeros(T,esn.No), outputs)
    end

    arch = esn.teacher_forcing ? TeacherForcing() : NoTeacherForcing()
    for t = 1:Nd
        state[:] = (esn.leaking_rate-one(T))*state + esn.leaking_rate*update(esn, state, inputs[:,t+1], outputs[:,t], arch)
        outputs[:,t+1] = esn.Wo*[one(T);inputs[:,t];state]
    end

    return outputs[:,2:end]
end

end # module
