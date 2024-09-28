import numpyro
import numpyro.distributions as dist

import jax.numpy as jnp
import jax._src.nn.functions as nnf

class NumPyroModel:
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, data_size, layer_prior_scale=1.0, output_prior_rate=1.0, output_prior_concentration=3.0, nonlin=nnf.relu):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_prior_scale = layer_prior_scale
        self.data_size = data_size
        self._layer_tuples = [(input_dim, hidden_dim)] + [(hidden_dim, hidden_dim)] * (num_layers - 2) + [(hidden_dim, output_dim)]
        self.nonlin = nonlin
        self.out_prior_rate = output_prior_rate
        self.out_prior_conc = output_prior_concentration
        
    def __str__(self):
        string_rep = ""
        for i, dims in enumerate(self._layer_tuples):
            string_rep += f"Layer {i+1}: {dims[0]} -> {dims[1]}\n"
        return string_rep
    
    def __call__(self, X, Y=None):
        for i, dims in enumerate(self._layer_tuples):
            w = numpyro.sample(
                f"w{i+1}",
                dist.Normal(
                    jnp.zeros(dims),
                    jnp.ones(dims)*self.layer_prior_scale
                    )
                )
            b = numpyro.sample(f"b{i+1}", dist.Normal(jnp.zeros(dims[1]), jnp.ones(dims[1])*self.layer_prior_scale))

            if i != self.num_layers-1:
                X = self.nonlin(jnp.dot(X, w) + b)
            else:
                X = jnp.dot(X, w) + b
        
        sigma = numpyro.sample("sigma", dist.Gamma(self.out_prior_conc, self.out_prior_rate).expand([self.output_dim]).to_event(1))
        sigma = jnp.diag(1.0/sigma)

        with numpyro.plate("data", self.data_size):
            numpyro.sample("Y", dist.MultivariateNormal(X, covariance_matrix=sigma), obs=Y)