#!/usr/bin/env python

import argparse
import json
import re

import polars
import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_cache", nargs="+", help="Cachefiles to combine")
    args = parser.parse_args()

    # f"{name}_{kernel_name}_{type_in}_to_{type_out}_{batch_size}x{m_global}x{n_global}x{k_global}.json"
    fname_re = re.compile(r"_([a-z]+\d+)_to_([a-z]+\d+)_(\d+)x(\d+)x(\d+)x(\d+)\.json")

    # Use first cache file to build data structure
    fname = args.input_cache[0]
    ref = {}
    ref["type_in"], ref["type_out"] = fname_re.search(fname).group(1, 2)

    with open(fname, "r") as fp:
        cache = json.loads(fp.read())

    ref["device_name"] = cache["device_name"]
    ref["kernel_name"] = cache["kernel_name"]
    ref["tune_params_keys"] = cache["tune_params_keys"]

    derived_keys = ["batch_size", "m_global", "n_global", "k_global"]
    result_keys = list(next(iter(cache["cache"].values())).keys())

    
    data = {k: [] for k in derived_keys + result_keys}

    # add results from each cache file, with some sanity checks
    for fname in tqdm.tqdm(args.input_cache, desc="Cache files"):
        values = {}
        values["type_in"], values["type_out"], values["batch_size"], values["m_global"], values["n_global"], values["k_global"] = fname_re.search(fname).group(*range(1, 7))
        for key in ("type_in", "type_out"):
            assert values[key] == ref[key], f"{key}: {values[key]} != {ref[key]}"

        with open(fname, "r") as fp:
            try:
                cache = json.loads(fp.read())
            except json.decoder.JSONDecodeError:
                print(f"Failed to read {fname}:")
                raise

        for key in ("device_name", "kernel_name", "tune_params_keys"):
            assert cache[key] == ref[key], f"{key}: {cache[key]} != {ref[key]}"

        for result in cache["cache"].values():
            if "__error__" not in result:
                for key in derived_keys:
                    data[key].append(int(values[key]))
                for key in result_keys:
                    data[key].append(result[key])

    # Convert to parquet
    df = polars.DataFrame(data)
    output_fname = f"cache/{ref['device_name'].lower().replace(' ', '_')}_{ref['kernel_name']}_{ref['type_in']}_to_{ref['type_out']}.parquet"
    df.write_parquet(output_fname)
