def aggregate_weighted_updates(
    base_state,
    client_updates,
    clients,
    weights,
    device,
):
    aggregated_state = {
        name: tensor.detach().clone() for name, tensor in base_state.items()
    }

    for client, weight in zip(clients, weights):
        for name, tensor in client_updates[client].items():
            update = tensor.to(device)
            if aggregated_state[name].is_floating_point():
                aggregated_state[name].add_(update, alpha=weight)
            else:
                aggregated_state[name] = aggregated_state[name] + update * weight

    return aggregated_state
