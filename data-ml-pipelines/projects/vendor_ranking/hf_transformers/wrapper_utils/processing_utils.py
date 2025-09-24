import torch
import structlog

LOG: "structlog.stdlib.BoundLogger" = structlog.getLogger(__name__)


def get_offline_embedding(account_id, country_code):
    offline_embeddings = read_account_offline_embedding(account_id, country_code)

    if not offline_embeddings.empty:
        account_id = offline_embeddings.iloc[0]['account_id']
        offline_embedding = offline_embeddings.iloc[0]['embeddings']
        return offline_embedding
    else:
        return None


def convert_embedding_to_tensor(embedding_str, device='cpu'):
    embedding_list = embedding_str.strip('[]').split(',')
    embedding_floats = [float(x) for x in embedding_list]
    tensor_embedding = torch.tensor([embedding_floats], device=device)
    LOG.info("Offline Embedding converted to tensor")
    return tensor_embedding


def get_sorted_chain_ids(chain_id_vocab, loaded_logits):
    """
    Given the chain_id_vocab and loaded_logits, return a dictionary of sorted chain_ids with their logits.
    """
    # Create the mapping from tokenizer's vocabulary
    index_to_chain_id = {v: k for k, v in chain_id_vocab.items()}
    # Create a list of tuples (chain_id, logit)
    chain_id_logits = [(index_to_chain_id[index], logit.item()) for index, logit in
                       enumerate(loaded_logits.squeeze(0))]

    sorted_chain_ids_with_logits = sorted(chain_id_logits, key=lambda x: x[1], reverse=True)
    sorted_chain_ids = [chain_id for chain_id, logit in sorted_chain_ids_with_logits]

    return sorted_chain_ids


def get_sorted_chain_ids_with_logits(chain_id_vocab, loaded_logits):
    """
    Given the chain_id_vocab and loaded_logits, return a dictionary of sorted chain_ids with their logits.
    """
    # Create the mapping from tokenizer's vocabulary
    index_to_chain_id = {v: k for k, v in chain_id_vocab.items()}
    # Create a list of tuples (chain_id, logit)
    chain_id_logits = [(index_to_chain_id[index], logit.item()) for index, logit in
                       enumerate(loaded_logits.squeeze(0))]

    sorted_chain_ids_with_logits = sorted(chain_id_logits, key=lambda x: x[1], reverse=True)
    sorted_chain_ids_with_logits_dict = {chain_id: logit for chain_id, logit in sorted_chain_ids_with_logits}

    return sorted_chain_ids_with_logits_dict


def get_chain_id_rank(input_chain_id, sorted_chain_ids):
    """
    Function to get the rank of a specific input_chain_id
    """
    try:
        rank = sorted_chain_ids.index(input_chain_id) + 1
        return rank
    except ValueError:
        LOG.error(f"Chain ID '{input_chain_id}' not found in the sorted chain_ids list.")
        return None
