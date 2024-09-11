import torch
from utils.analysis import (
    get_model,
    get_sae_data,
    get_module,
    get_sae,
    get_config,
    get_samples,
    get_latents_and_sequences,
    parser)

args = parser.parse_args()
if args.end is not None:
    indices = range(args.start, args.end)
else:
    indices = [args.start]

pcfg, model, dataloader = get_model(args.model_dir)
module = get_module(model, 'res0')

def get_range_of_latent(sae, fidx, num_batches=1):
    activations = []
    if isinstance(fidx, int): fidx = [fidx]
    def hook(module, input, output):
        nonlocal activations
        latent, _ = sae(output)
        activations += (latent.index_select(-1, torch.tensor(fidx, dtype=torch.int32)))

    handle = module.register_forward_hook(hook)
    for _ in range(num_batches):
        batch, _ = next(iter(dataloader))
        model(batch)
    handle.remove()
    act = torch.stack(activations, 0)
    return act.min().item(), act.max().item()

def get_hook(sae, fidx, clamp_value):
    """fidx can be an index or a list of indices"""
    def hook(module, input, output):
        latent, _ = sae(output)
        latent.index_fill_(-1, torch.tensor(fidx), clamp_value)
        recon = sae.decoder(latent)
        return recon
    return hook

def get_complement_hook(sae, fidx, clamp_value=0):
    def hook(module, input, output):
        latent, _ = sae(output)
        indices = [i for i in range(latent.size(-1)) if i not in fidx]
        latent.index_fill_(-1, torch.tensor(indices), 0)
        recon = sae.decoder(latent)
        return recon
    return hook

def fuck_with_model(hook, measures, show_samples=False):
    """
    Compare the value of some measure on model generations
    before and after intervening to null out a certain feature.
    The measures must each accept a string (the sentence) and a list of logprobs
    and return a scalar.
    """
    clean_samples, clean_logprobs = get_samples(100)
    if show_samples:
        print("Clean:")
        print('\n'.join(clean_samples[:show_samples]))

    handle = module.register_forward_hook(hook)
    dirty_samples, dirty_logprobs = get_samples(100)
    handle.remove()
    if show_samples:
        print("Dirty:")
        print('\n'.join(dirty_samples[:show_samples]))

    clean_measures = [[measure(sample, logprobs)
                        for sample, logprobs in zip(clean_samples, clean_logprobs)]
                            for measure in measures]
    dirty_measures = [[measure(sample, logprobs)
                         for sample, logprobs in zip(dirty_samples, dirty_logprobs)]
                            for measure in measures]

    return clean_measures, dirty_measures