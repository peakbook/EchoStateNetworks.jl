using EchoStateNetworks
using LinearAlgebra
using CSV
using Random

T = Float64
filepath = joinpath(dirname(@__FILE__),"MackeyGlass_t17.txt")
# if !isfile(filepath)
#     println("download data...")
#     run(`curl -o $(filepath) http://minds.jacobs-university.de/sites/default/files/uploads/mantas/code/MackeyGlass_t17.txt`)
#     println("finish.")
# end
data = convert(Array{T},CSV.read(filepath))'

train_range = 1:2000
test_range = 2001:4000
data_range = 1:4000

esn = EchoStateNetwork{T}(Ni=1, No=1, Nr=500,
                          spectral_radius=1.2,
                          sparsity=0.95,
                          leaking_rate=0.99,
                          teacher_forcing=true,
                          rng=MersenneTwister(0x59))

y_train = data[:,train_range]
x_train = ones(T, 1,length(train_range))
y_test = data[:,test_range]
x_test = ones(T, 1,length(test_range))

y_pred = train!(esn, x_train, y_train, reg=1.0e-8, discard=100)
println("train RMSE: $(norm(y_train-y_pred)/sqrt(length(train_range)))")
y_pred = predict!(esn, x_test)
println("test RMSE:  $(norm(y_test-y_pred)/sqrt(length(test_range)))")
