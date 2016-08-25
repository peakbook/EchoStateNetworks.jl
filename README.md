# EchoStateNetworks

A julia implementation of Echo State Networks (ESNs).

## Usage
``` julia
using EchoStateNetworks

T = Float64
data = convert(Array{T},readcsv("MackeyGlass_t17.txt")')

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
```

![ESN_example](http://peakbook.github.io/images/ESN_MackeyGlass.svg)

## References
- [Jaeger, Herbert. ``The "echo state" approach to analysing and training recurrent neural networks,`` GMD Report 148, 2001.](http://minds.jacobs-university.de/sites/default/files/uploads/papers/EchoStatesTechRep.pdf)
- <http://www.scholarpedia.org/article/Echo_state_network>
- [Lukoševičius, Mantas. ``A practical guide to applying echo state networks,`` Neural networks: Tricks of the trade. pp.659-686, 2012.](http://link.springer.com/chapter/10.1007/978-3-642-35289-8_36)
- <http://minds.jacobs-university.de/ESNresearch>
