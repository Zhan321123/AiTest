<h1 style="text-align:center;">Module</h1>

### Module

- **Linear**
$$ X_i_j = \sum_{k=1}^{Dim_I} W_{i,k} X_{k,j} + b_i $$
- **Convolution 2d**
$$ X_i_j = \sum_{k=1}^{Size_{kernel}^2} W_{i,k} X_{k,j} + b_i $$
- **Max Pooling**
$$ X_i_j = \max(0, X_{i,j}) $$
- **Batch Normalization 2d**
$$ X_i_j = \frac{X_{i,j} - \mu_i}{\sqrt{\sigma_i + \epsilon}} * \gamma_i + \beta_i $$

### Number of Parameters

- **Linear**
$$ N_w = Dim_I * Dim_O $$
$$ N_b = Dim_O $$
- **Convolution 2d**
$$ N_w = Size_{kernel}^2 * Channrl_I * Channel_O $$
$$ N_b = Channel_O $$
- **Flatten** $ N=0 $
- **Pooling** $ N=0 $
- **Batch Normalization 2d**
$$ N_\mu = Channel * 2 $$
$$ N_\sigma = Channel * 2 $$

### Parameters

- **Linear**
  - $ Dim_I $
  - $ Dim_O $
- **Convolution 2d**
  - $ Channel_I $
  - $ OutChannel_O $
  - $ Size_{kernel} $
  - $ Padding $
  - $ Stride $
- **Pooling**
  - $ Size_{kernel} $
  - $ Stride $
- **Batch Normalization 2d**
  - $ Channel $
  - $ Momentum $
- **Adaptive Avg Pool 2d**
  - $ W' $
  - $ H' $