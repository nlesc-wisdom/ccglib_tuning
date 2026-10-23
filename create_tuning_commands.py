#!/usr/bin/env python
from functools import cache
from pathlib import Path

from cuda.bindings import driver
from cuda.core import Device

device = Device()
device.set_current()


@cache
def get_max_memory():
    err, _free_mem, total_mem = driver.cuMemGetInfo()
    assert total_mem is not None, f"Failed to get size of GPU memory ({err})"
    return total_mem


def type_to_size(type_name):
    match type_name:
        case "bfloat16":
            type_size = 2
        case "int1":
            type_size = 1./8
        case "int32":
            type_size = 4
        case "float8e4m3":
            type_size = 1
        case "float8e5m2":
            type_size = 1
        case "float16":
            type_size = 2
        case "float32":
            type_size = 4
        case _:
            raise ValueError(f"Unknown input type: {type_name}")
    return type_size


def check_memory_requirement(b, m, n, k, type_in_size, type_out_size):
    max_size = get_max_memory()
    sizeA = 2 * b * m * k * type_in_size
    sizeB = 2 * b * n * k * type_in_size
    sizeC = 2 * b * m * n * type_out_size
    return (sizeA + sizeB + sizeC) < max_size


def construct_axis_search_space(type_in, type_out):
    type_in_size = type_to_size(type_in)
    type_out_size = type_to_size(type_out)

    batch_sizes = [1, 32, 1024]

    mnk_min = 16
    mnk_max = 8192

    mnk_typical = 256

    fixed = mnk_typical
    variable = mnk_min
    while variable <= mnk_max:
        for b in batch_sizes:
            if check_memory_requirement(b, variable, variable, variable, type_in_size, type_out_size):
                yield f"-b {b} -m {variable} -n {variable} -k {variable}"
            # fixed M
            if check_memory_requirement(b, fixed, variable, variable, type_in_size, type_out_size):
                yield f"-b {b} -m {fixed} -n {variable} -k {variable}"
            # fixed N
            if check_memory_requirement(b, variable, fixed, variable, type_in_size, type_out_size):
                yield f"-b {b} -m {variable} -n {fixed} -k {variable}"
            # fixed K
            if check_memory_requirement(b, variable, variable, fixed, type_in_size, type_out_size):
                yield f"-b {b} -m {variable} -n {variable} -k {fixed}"
            # only M variable
            if check_memory_requirement(b, variable, fixed, fixed, type_in_size, type_out_size):
                yield f"-b {b} -m {variable} -n {fixed} -k {fixed}"
            # only N variable
            if check_memory_requirement(b, fixed, variable, fixed, type_in_size, type_out_size):
                yield f"-b {b} -m {fixed} -n {variable} -k {fixed}"
            # only K variable
            if check_memory_requirement(b, fixed, fixed, variable, type_in_size, type_out_size):
                yield f"-b {b} -m {fixed} -n {fixed} -k {variable}"
        variable *= 2


if __name__ == "__main__":
    # assume ccglib is next to this repo
    ccglib_dir = Path(__file__).resolve().parent.parent / "ccglib"

    fixed_args = {
        "type_in": "float16",
        "type_out": "float32",
        "observer": "pmt",
        "kernel": "opt",
        "backend": "cupy",
        "ccglib": ccglib_dir,
        "name": f"cache/{device.name.lower().replace(' ', '_')}"
    }

    fixed_arg_str = " ".join((f"--{k}={v}" for k, v in fixed_args.items()))

    tuning_args = []
    for item in construct_axis_search_space(fixed_args["type_in"], fixed_args["type_out"]):
        # avoid duplicates
        if item not in tuning_args:
            tuning_args.append(item)

    # create command line for each run
    with open("tuning_commands.txt", "w") as fp:
        for item in tuning_args:
            fp.write(f"{ccglib_dir}/tuning/tune_gemm.py {fixed_arg_str} {item}\n")
        
