from bitnet.model_parallel.layers import BitColumnParallelLinear, BitRowParallelLinear

def test_bit_column_parallel_linear_init():
    # Checks initialization and partitioning of weights
    model = BitColumnParallelLinear(8, 16, bias=True, gather_output=True)
    .assertEqual(model.weight.shape, (model.output_size_per_partition, 8))
    if model.bias is not None:
        .assertEqual(model.bias.shape, (model.output_size_per_partition,))

def test_bit_row_parallel_linear_init():
    # Checks initialization and partitioning of weights
    model = BitRowParallelLinear(8, 16, bias=True)
    .assertEqual(model.weight.shape, (16, model.input_size_per_partition))
    if model.bias is not None:
        .assertEqual(model.bias.shape, (16,))

def test_forward_pass_column():
    # Testing forward pass behavior for column parallel
    model = BitColumnParallelLinear(8, 16)
    input_tensor = torch.randn(5, 8)
    output = model(input_tensor)
    .assertEqual(output.shape, (5, 16))

def test_forward_pass_row():
    # Testing forward pass behavior for row parallel
    model = BitRowParallelLinear(8, 16)
    input_tensor = torch.randn(5, 8)
    output = model(input_tensor)
    assert output.shape,
    assertEqual(output.shape, (5, 16))

def test_non_divisible_features_error():
    # Testing error handling for non-divisible feature numbers
    with assertRaises(AssertionError):
        model = BitRowParallelLinear(7, 16)

