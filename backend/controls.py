import torch
import torch.nn as nn
from collections import OrderedDict
known_layers = {
    'Linear': nn.Linear,
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
    'Conv1d': nn.Conv1d,
    'ConvTranspose1d': nn.ConvTranspose1d,
    'BatchNorm1d': nn.BatchNorm1d,
    'Dropout': nn.Dropout,
    'Flatten': nn.Flatten,
    'MaxPool1d': nn.MaxPool1d,
    'AvgPool1d': nn.AvgPool1d,
}
sdl = {
    "type": "Sequential",
    "constraints": {
        "no_branches": True,
        "only_known_layers": True
    },
    "forbid_removal_if_contains": ["Conv1d", "Linear"]
}
def validate_model_against_sdl_for_frontend(model: nn.Sequential):
    result = {
        "is_valid": True,
        "errors": [],
        "hints": [
            "La rete deve essere composta da un nn.Sequential esterno con blocchi Sequential interni.",
            "Ogni blocco interno può contenere qualsiasi layer, se contiene Linear o Conv non è protetto dalla rimozione.",
            "Sono ammessi solo i layer noti nel sistema.",
        ],
        "invalid_blocks": []
    }

    if not isinstance(model, nn.Sequential):
        result["is_valid"] = False
        result["errors"].append("Il modello caricato non è un nn.Sequential al livello principale.")
        return result

    sequential_blocks = list(model.children())

    for i, block in enumerate(sequential_blocks):
        if isinstance(block, nn.Sequential):
            flat = list(block.modules())[1:]  # exclude the container itself
            layer_names = [type(m).__name__ for m in flat]

            unknown_layers = [name for name in layer_names if name not in known_layers]
            if sdl["constraints"].get("only_known_layers", False) and unknown_layers:
                result["is_valid"] = False
                result["errors"].append(
                    f"Blocco {i} contiene layer non supportati: {', '.join(unknown_layers)}"
                )
                result["invalid_blocks"].append({
                    "index": i,
                    "layers": layer_names
                })

            if not any(name in sdl.get("forbid_removal_if_contains", []) for name in layer_names):
                # Questo blocco è "non rimuovibile"
                continue  # tutto ok

        else:
            result["is_valid"] = False
            result["errors"].append(
                f"Elemento al livello principale non è un Sequential: {type(block).__name__} (blocco {i})"
            )
            result["invalid_blocks"].append({
                "index": i,
                "layers": [type(block).__name__]
            })

    return result



def recreate_module(jit_module):
    original_name = getattr(jit_module, "original_name", None)

    if original_name == 'Sequential':
        submodules = []
        for i, (name, submodule) in enumerate(jit_module.named_children()):
            submodules.append((str(i), recreate_module(submodule)))
        return nn.Sequential(OrderedDict(submodules))

    elif original_name == 'Linear':
        state = jit_module.state_dict()
        weight = state['weight']
        bias = 'bias' in state
        in_features = weight.shape[1]
        out_features = weight.shape[0]
        layer = nn.Linear(in_features, out_features, bias=bias)
        layer.load_state_dict(state)
        return layer

    elif original_name == 'Conv1d':
        state = jit_module.state_dict()
        weight = state['weight']
        bias = 'bias' in state
        out_channels, in_channels, kernel_size = weight.shape
        layer = nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias)
        layer.load_state_dict(state)
        return layer

    elif original_name == 'ConvTranspose1d':
        state = jit_module.state_dict()
        weight = state['weight']
        bias = 'bias' in state
        in_channels = weight.shape[1]
        out_channels = weight.shape[0]
        kernel_size = weight.shape[2]
        layer = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, bias=bias)
        layer.load_state_dict(state)
        return layer

    elif original_name == 'BatchNorm1d':
        state = jit_module.state_dict()
        num_features = state['weight'].shape[0] if 'weight' in state else state['running_mean'].shape[0]
        layer = nn.BatchNorm1d(num_features)
        layer.load_state_dict(state)
        return layer

    elif original_name == 'Dropout':
        try:
            p = jit_module.p if hasattr(jit_module, 'p') else 0.5
            return nn.Dropout(p)
        except:
            return nn.Dropout()

    elif original_name == 'Flatten':
        try:
            return nn.Flatten(start_dim=1)  # assumendo start_dim=1 per default
        except:
            return nn.Flatten()

    elif original_name == 'MaxPool1d':
        try:
            kernel_size = jit_module.kernel_size if hasattr(jit_module, 'kernel_size') else 2
            return nn.MaxPool1d(kernel_size)
        except:
            return nn.MaxPool1d(2)

    elif original_name == 'AvgPool1d':
        try:
            kernel_size = jit_module.kernel_size if hasattr(jit_module, 'kernel_size') else 2
            return nn.AvgPool1d(kernel_size)
        except:
            return nn.AvgPool1d(2)

    elif original_name in known_layers:
        try:
            return known_layers[original_name]()
        except Exception as e:
            print(f"[WARNING] Impossibile creare layer {original_name}: {e}")
            return nn.Identity()

    else:
        print(f"[WARNING] Tipo di layer non supportato: {original_name}")
        return nn.Identity()


def recreate_sequential_model_from_jit(model_path):
    jit_model = torch.jit.load(model_path)
    jit_model = jit_model.net if hasattr(jit_model, 'net') else jit_model
    return recreate_module(jit_model)
