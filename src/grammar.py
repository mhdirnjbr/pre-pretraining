import fire
import numpy as np
import os
import random
from tqdm import trange


def generate_dyck(num_symbols, min_depth=1, max_depth=4, max_length=510, offset=None):
    """Generates a Dyck sequence with specified number of symbols and depth constraints.

    Args:
        num_symbols: The number of distinct symbol pairs (k in k-Dyck).
        min_depth: Minimum required depth of nested brackets.
        max_depth: Maximum allowed depth of nested brackets.
        max_length: The maximum length of the generated sequence.

    Returns:
        A list representing the Dyck sequence, or None if generation fails.
    """
    result = []
    stack = []

    if min_depth < 1:
        raise ValueError("min_depth must be at least 1.")

    if offset is None:
        offset = num_symbols

    # Initialize with minimum depth
    for _ in range(min_depth):
        opening_symbol = np.random.randint(0, num_symbols)
        result.append(opening_symbol)
        stack.append(opening_symbol)

    while len(result) < max_length:
        if (
            len(stack) < max_depth and random.random() < 0.5
        ):  # Try to open if under max depth
            if len(result) >= max_length - 1:
                closing_symbol = stack.pop() + offset
                result.append(closing_symbol)
                continue
            opening_symbol = np.random.randint(0, num_symbols)
            result.append(opening_symbol)
            stack.append(opening_symbol)
        else:  # Close existing bracket
            closing_symbol = stack.pop() + offset
            result.append(closing_symbol)
            if not stack:
                break

    # pop remaining stuff on the stack if any
    while stack:
        closing_symbol = stack.pop() + offset
        result.append(closing_symbol)

    return result if not stack else None


def generate_dyck_txt_file(
    file_dir, num_symbols=30, n=100000, target_length=2048, min_depth=1, max_depth=16, split=None
):
    """Generates a text file containing Dyck sequences.

    Args:
        file_dir: The directory to save the file.
        num_symbols: The number of distinct symbol pairs (k in k-Dyck).
        n: The number of sequences to generate.
        target_length: Target length of each sequence.
        min_depth: Minimum required depth of nested brackets.
        max_depth: Maximum allowed depth of nested brackets.
    """
    import os

    os.makedirs(file_dir, exist_ok=True)
    with open(
        f"{file_dir}/dyck_sequences_{num_symbols}_{min_depth}_{max_depth}_{split}.txt", "w"
    ) as f:
        for i in trange(n):
            result = []
            while len(result) < target_length:
                new_seq = generate_dyck(
                    num_symbols, min_depth=min_depth, max_depth=max_depth
                )
                if new_seq is None:
                    continue
                result.extend(new_seq)

            dyck_str = " ".join(
                [str(x) for x in result[:target_length]]
            )  # truncate to target length
            f.write(f"{dyck_str}\n")


def make_copy_tokens(
    num_symbols: int = 64, min_w_length: int = 10, max_w_length: int = 510
):
    """Generates a string of repeated random integers.

    The function creates a string of random integers between 0 and
    `num_symbols`, with a length between min_w_length and
    max_w_length. It then duplicates this string and concatenates
    the two copies, returning the resulting string.

    Args:
        num_symbols: The number of possible symbols (integers).
        min_w_length: The minimum length of the initial random string.
        max_w_length: The maximum length of the initial random string.

    Returns:
        A string of repeated random integers.

    Raises:
        ValueError: If min_w_length > max_w_length
    """
    if min_w_length > max_w_length:
        raise ValueError("min_w_length cannot be greater than max_w_length")

    length = random.randint(min_w_length, max_w_length)
    original_token_seq = np.random.randint(0, num_symbols, size=length).tolist()
    return original_token_seq + original_token_seq


def make_copy_str_file(
    file_dir,
    num_symbols: int = 64,
    n: int = 100000,
    seq_length: int = 2048,
    min_w_length: int = 10,
):
    """Generates a text file containing strings of repeated random integers, padded to a minimum length.

    The function generates `n` strings of repeated random integers, each
    with a length *at least* `min_length`.  The strings are saved to a
    text file in the specified directory.  Generated string is packed
    until it reaches the seq length.

    Args:
        file_dir: The directory to save the file.
        num_symbols: The number of possible symbols (integers).
        n: The number of strings to generate.
        seq_length: The target length of each sequence.
        min_w_length: The minimum length of each initial random string.
    """
    os.makedirs(file_dir, exist_ok=True)
    with open(f"{file_dir}/ww_sequences_{num_symbols}_{min_w_length}.txt", "w") as f:
        for _ in trange(n):
            sequence = make_copy_tokens(num_symbols, min_w_length=min_w_length)
            while len(sequence) < seq_length:
                sequence.extend(
                    make_copy_tokens(num_symbols, min_w_length=min_w_length)
                )
            repeated_str = " ".join(map(str, sequence[:seq_length]))
            f.write(f"{repeated_str}\n")


def generate_shuff_dyck(k, max_length=2048, p_open=0.5, max_depth=16):
    """
    Generate a k-shuffle Dyck sequence, truncated at max_length.
    When max depth is reached, close one bracket and continue.

    Args:
        k (int): Number of different types of brackets
        max_length (int): Target maximum length of the sequence
        p_open (float): Probability of opening a new bracket
        max_depth (int): Maximum nesting depth allowed

    Returns:
        list: Generated sequence where i represents opening bracket i
             and i+k represents closing bracket i

    Note: the final Dyck word may be invalid due to truncation, but
    we didn't find this to be an issue in practice.
    """
    sequence = []
    counts = [0] * k  # Track open brackets of each type

    while len(sequence) < max_length:
        depth = sum(counts)

        # Must open if all brackets are closed
        if depth == 0:
            bracket = random.randint(0, k - 1)
            sequence.append(bracket)
            counts[bracket] += 1
            continue

        # If at max depth, force a close
        if depth >= max_depth:
            open_brackets = [i for i, count in enumerate(counts) if count > 0]
            bracket = random.choice(open_brackets)
            sequence.append(bracket + k)
            counts[bracket] -= 1
            continue

        # Randomly choose to open or close
        if random.random() < p_open and depth < max_depth:
            bracket = random.randint(0, k - 1)
            sequence.append(bracket)
            counts[bracket] += 1
        else:
            # Close an existing bracket
            open_brackets = [i for i, count in enumerate(counts) if count > 0]
            bracket = random.choice(open_brackets)
            sequence.append(bracket + k)
            counts[bracket] -= 1

    return sequence


def generate_shuff_dyck_txt_file(
    file_dir, num_symbols=64, n=100000, target_length=2048, p=0.5
):
    """Generates a text file containing Dyck sequences with cross-serial dependencies.

    Args:
        file_dir: The directory to save the file.
        num_symbols: The number of distinct symbol pairs (k in k-Dyck).
        n: The number of sequences to generate.
        target_length: desired sequence length.
    """
    os.makedirs(file_dir, exist_ok=True)
    with open(
        f"{file_dir}/dyck_sequences_cross_serial_{num_symbols}_{p}.txt", "w"
    ) as f:
        for i in range(n):
            result = generate_shuff_dyck(num_symbols, target_length, p)
            dyck_str = " ".join([str(x) for x in result[:target_length]])
            f.write(f"{dyck_str}\n")


if __name__ == "__main__":
    fire.Fire(
        {
            "generate_dyck": generate_dyck_txt_file,
            "generate_shuff_dyck": generate_shuff_dyck_txt_file,
            "generate_ww": make_copy_str_file,
        }
    )
