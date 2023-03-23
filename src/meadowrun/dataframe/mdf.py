from __future__ import annotations

import datetime
import dataclasses
import gzip
import io
import os
import string
import asyncio
import pickle
import sys
import time
import uuid
from typing import (
    TypeVar,
    Callable,
    Sequence,
    Generic,
    Union,
    Iterable,
    List,
    Tuple,
    Any,
    Dict,
    Optional,
    AsyncIterable, Coroutine, Deque,
)

import boto3

import meadowrun
from meadowrun import StorageBucketSpec
from meadowrun.k8s_integration.storage_spec import S3BucketSpec
from meadowrun.abstract_storage_bucket import AbstractStorageBucket
from meadowrun.run_job import run_map
from meadowrun.shared import _chunker
from meadowrun.storage_grid_job import get_aws_s3_bucket

try:
    import pandas as pd
    import pandas.core.groupby.generic
    import numpy as np
except ImportError:
    print("Using mdf without pandas installed is not recommended")

_T = TypeVar("_T")
_U = TypeVar("_U")
_TPartName = TypeVar("_TPartName")
_TDataFrame = TypeVar("_TDataFrame")
_UDataFrame = TypeVar("_UDataFrame")
_VDataFrame = TypeVar("_VDataFrame")
_PandasDF = TypeVar("_PandasDF")


@dataclasses.dataclass(frozen=True)
class _SquareBracketProperty:
    other_self: Any
    func: Callable[[Any, Any], Any]

    def __getitem__(self, item: Any) -> Any:
        return self.func(self.other_self, item)


def _square_bracket_property(func: Callable[[Any, Any], Any]):
    return property(lambda self: _SquareBracketProperty(self, func))


class DataFrame(Generic[_UDataFrame]):
    def __init__(
        self,
        part_function: Callable[[int, _TPartName, _TDataFrame], _UDataFrame],
        part_names_and_values: Optional[Sequence[Tuple[_TPartName, _TDataFrame]]],
        previous_groupby: Optional[DataFrameGroupBy],
        previous_reduce: Optional[DataFrameReduce],
    ):
        # if not (bool(arguments) ^ bool(previous_groupby) ^ bool(previous_reduce)):
        #     raise ValueError("")
        self._part_function = part_function
        self._part_names_and_values = part_names_and_values
        self._previous_groupby = previous_groupby
        self._previous_reduce = previous_reduce

    @classmethod
    def from_map(
        cls,
        part_function: Callable[[int, _TPartName, _TDataFrame], _UDataFrame],
        part_names_and_values: Sequence[Tuple[_TPartName, _TDataFrame]],
    ) -> DataFrame:
        return cls(part_function, part_names_and_values, None, None)

    @classmethod
    async def from_s3(
        cls,
        name: str,
        storage_bucket: StorageBucketSpec,
        column_names: Optional[Sequence[str]] = None,
    ) -> DataFrame:
        # TODO same super incorrect problem
        async with await storage_bucket.get_storage_bucket("default") as sb:
            # TODO check for completeness? chop off prefix? convert to int?
            object_names = await sb.list_objects(f"{name}/")

        # it seems like sometimes S3 will think that the folder is an object
        object_names = [(object_name[len(name) + 1:], object_name) for object_name in object_names if not object_name.endswith("/")]
        object_names.sort()  # should be already sorted, but just to be sure

        return cls.from_map(
            lambda i, part_name, object_name: asyncio.run(
                _from_s3_helper(storage_bucket, object_name, column_names)
            ),
            object_names,
        )

    @_square_bracket_property
    def parts_loc(self, labels: Any) -> DataFrame[_UDataFrame]:
        if self._part_names_and_values is None:
            raise NotImplementedError(
                "Cannot call parts_loc on the result of a groupby"
            )
        # TODO this is a bit sketchy in the case where you do e.g.
        # mddf.map_parts(func_with_side_effects).parts_between(...) ... This gets treated
        # as mddf.parts_between().map_parts(func_with_side_effects)

        if not isinstance(labels, slice):
            labels = slice(labels, labels)
        if labels.step is not None:
            raise NotImplementedError("slices with steps are not supported")

        return DataFrame(
            self._part_function,
            [
                part_name
                for part_name, part_value in self._part_names_and_values
                if (labels.start is None or labels.start <= part_name)
                and (labels.stop is None or labels.stop >= part_name)
            ],
            self._previous_groupby,
        )

    @_square_bracket_property
    def parts_iloc(self, labels: Any) -> DataFrame[_UDataFrame]:
        if self._part_names_and_values is None:
            raise NotImplementedError(
                "Cannot call parts_iloc on the result of a groupby"
            )
        # TODO this is a bit sketchy in the case where you do e.g.
        # mddf.map_parts(func_with_side_effects).parts_between(...) ... This gets treated
        # as mddf.parts_between().map_parts(func_with_side_effects)

        if isinstance(labels, slice):
            new_parts = self._part_names_and_values[labels]
        else:
            new_parts = [self._part_names_and_values[labels]]

        return DataFrame(self._part_function, new_parts, self._previous_groupby, self._previous_reduce)

    @property
    def part_names(self) -> List[_T]:
        if self._part_names_and_values is None:
            raise NotImplementedError(
                "Cannot call part_names on the result of a groupby"
            )
        # TODO I think we should rename _arguments to part_name everywhere
        return [part_name for part_name, part_value in self._part_names_and_values]

    def map_parts_with_name(
        self, func: Callable[[int, _TPartName, _UDataFrame], _VDataFrame]
    ) -> DataFrame[_VDataFrame]:
        return DataFrame(
            lambda i, part_name, part_value: func(i, part_name, self._part_function(i, part_name, part_value)),
            self._part_names_and_values,
            self._previous_groupby,
            self._previous_reduce,
        )

    def map_parts(
        self, func: Callable[[_UDataFrame], _VDataFrame]
    ) -> DataFrame[_VDataFrame]:
        return self.map_parts_with_name(lambda i, part_name, part_value: func(part_value))

    # BUSTED
    # async def compute_with_index(
    #     self, *run_map_args, **run_map_kwargs
    # ) -> Sequence[Tuple[int, _PandasDF]]:
    #     arguments_list = list(self._part_names_and_values)
    #     # TODO make this work with _previous_groupby
    #     return [
    #         (arg, part)
    #         for ((i, arg), part) in zip(
    #             arguments_list,
    #             await run_map(
    #                 lambda arg: self._part_function(*arg),
    #                 arguments_list,
    #                 *run_map_args,
    #                 **run_map_kwargs,
    #             ),
    #         )
    #     ]

    async def compute(self, *run_map_args, **run_map_kwargs) -> Sequence[_UDataFrame]:
        if self._part_names_and_values is not None:
            part_names_and_values = self._part_names_and_values
        elif self._previous_groupby is not None:
            t0 = time.time()
            # TODO we should actually be able to read this off of the Host. And what if it's in run_map_args?
            storage_bucket = run_map_kwargs.pop("storage_bucket")
            # TODO probably just get rid of run_map_args. Also, make this work more generally
            if asyncio.iscoroutine(run_map_kwargs.get("deployment", None)):
                run_map_kwargs["deployment"] = await run_map_kwargs["deployment"]
            groups = await self._previous_groupby._base_df._to_groups(
                self._previous_groupby._column_names,
                self._previous_groupby._num_bits,
                storage_bucket,
                *run_map_args,
                **run_map_kwargs,
            )
            part_names_and_values = (
                (group_index, (group, storage_bucket))
                for group_index, group in groups.items()
            )
            print(f"Completed previous groupby in {time.time() - t0}")
        elif self._previous_reduce is not None:
            # TODO this should be the same as in _reduce_helper and others
            weird_job_id = str(uuid.uuid4())

            # TODO see comment above
            storage_bucket = run_map_kwargs.pop("storage_bucket")
            if asyncio.iscoroutine(run_map_kwargs.get("deployment", None)):
                run_map_kwargs["deployment"] = await run_map_kwargs["deployment"]

            t0 = time.time()
            chunked_intermediates = await self._previous_reduce._base_df._reduce_helper(
                storage_bucket, self._previous_reduce._branch_factor, *run_map_args, **run_map_kwargs
            )
            print(f"Completed stage 0 ({len(self._previous_reduce._base_df._part_names_and_values)} > {len(chunked_intermediates)}) in {time.time() - t0} {datetime.datetime.now()}")
            stage = 1  # TODO previous line is stage 0
            while len(chunked_intermediates) > self._previous_reduce._final_num_partitions:
                t0 = time.time()
                reducer_func = self._part_function  # TODO there's probably a better way to pass this around?
                await run_map(
                    lambda args: _to_intermediate(
                        reducer_func(args[0], args[0], (args[1], storage_bucket)),
                        f"TO_RENAME_INTERMEDIATES/{weird_job_id}/{stage}_{args[0]}.pkl",
                        storage_bucket,
                    ),
                    list(enumerate(chunked_intermediates)),
                    *run_map_args,
                    **run_map_kwargs
                )

                chunked_intermediates = list(_chunker(
                    (f"TO_RENAME_INTERMEDIATES/{weird_job_id}/{stage}_{i}.pkl" for i in
                     range(len(chunked_intermediates))), self._previous_reduce._branch_factor))
                print(
                    f"Completed stage {stage} ({len(chunked_intermediates)}) in {time.time() - t0} {datetime.datetime.now()}")

            part_names_and_values = [(i, (chunk, storage_bucket)) for i, chunk in enumerate(chunked_intermediates)]
        else:
            raise ValueError(
                "_part_names, _previous_groupby, and _previous_reduce are all None "
                "which should never happen!"
            )

        return await run_map(
            lambda args: self._part_function(*args),
            [(i, part_name, part_value) for i, (part_name, part_value) in enumerate(part_names_and_values)],
            *run_map_args,
            **run_map_kwargs,
        )

    async def head(self, *run_map_args, **run_map_kwargs) -> _UDataFrame:
        # TODO there should be special interaction with .from_s3().head()
        return (await self.parts_iloc[:1].map_parts(lambda df: df.head()).compute(*run_map_args, **run_map_kwargs))[0]

    async def to_pd(self, *run_map_args, **run_map_kwargs) -> _PandasDF:
        return pd.concat(await self.compute(*run_map_args, **run_map_kwargs))

    # TODO shouldn't be importing this from K8s
    async def to_s3(
        self,
        name: str,
        storage_bucket: StorageBucketSpec,
        *run_map_args,
        **run_map_kwargs,
    ):
        # TODO use {i} if {part_name} won't work? Or make it an option?
        await self.map_parts_with_name(
            lambda i, part_name, part_value: asyncio.run(_to_s3_helper(storage_bucket, part_value, f"{name}/{part_name}.parquet"))
        ).compute(*run_map_args, **run_map_kwargs)

    # TODO this API should be replaced by a plain .flatten() which writes to an
    # intermediate storage location if there is a subsequent .map_parts OR just writes
    # directly to a result storage bucket in the case of .flatten().to_s3()
    async def flatten_to_s3(
            self,
            name: str,
            storage_bucket: StorageBucketSpec,
            *run_map_args,
            **run_map_kwargs,
    ):
        await self.map_parts(
            lambda part_value: asyncio.run(_flatten_to_s3_helper(storage_bucket, part_value, name))
        ).compute(*run_map_args, **run_map_kwargs)

    async def _to_groups(
        self,
        column_names: Sequence[str],
        num_bits: int,
        storage_bucket: StorageBucketSpec,
        *run_map_args,
        **run_map_kwargs,
    ) -> Dict[int, List[Tuple[str, Tuple[int, int]]]]:
        # TODO need to clean these up later! if we get this id to line up with the
        # actual job id, the job code will clean this up for us?
        # TODO if we do that we probably want some sort of index to keep track of
        # different sub jobs
        # TODO add something to storage_keys
        job_id = str(uuid.uuid4())
        groups_by_source = await self.map_parts_with_name(
            lambda i, part_name, part_value: (
                i,
                asyncio.run(
                    _to_groups_helper(
                        part_value,
                        column_names,
                        num_bits,
                        f"TO_RENAME_INTERMEDIATES/{job_id}/{i}.pkl",
                        storage_bucket,
                    )
                ),
            )
        ).compute(*run_map_args, **run_map_kwargs)

        groups_by_dest = {}
        for source_index, source_groups in groups_by_source:
            for dest_index, byte_range in source_groups.items():
                if dest_index not in groups_by_dest:
                    groups_by_dest[dest_index] = []

                groups_by_dest[dest_index].append(
                    (f"TO_RENAME_INTERMEDIATES/{job_id}/{source_index}.pkl", byte_range)
                )

        return groups_by_dest

    # TODO rename
    async def _reduce_helper(self, storage_bucket: StorageBucketSpec,
                             branch_factor: int,
        *run_map_args,
        **run_map_kwargs) -> List[Sequence[str]]:

        # TODO see comment above
        job_id = str(uuid.uuid4())
        await self.map_parts_with_name(
            lambda i, part_name, part_value:
                _to_intermediate(
                    part_value,
                    f"TO_RENAME_INTERMEDIATES/{job_id}/{i}.pkl",
                    storage_bucket,
                ),
        ).compute(*run_map_args, **run_map_kwargs)

        return list(_chunker((f"TO_RENAME_INTERMEDIATES/{job_id}/{i}.pkl" for i in range(len(self._part_names_and_values))), branch_factor))

    # TODO probably rename to shuffle?
    def groupby(
        self,
        column_names: Sequence[str],
        num_partitions: int,
    ) -> DataFrameGroupBy:
        """
        num_partitions must be a power of 2
        """
        if num_partitions <= 0 or ((num_partitions - 1) & num_partitions) != 0:
            raise ValueError("num_partitions must be a power of 2")

        num_bits = num_partitions.bit_length() - 1
        return DataFrameGroupBy(self, column_names, num_bits)

    # def reduce_raw(self
    #     func2: Callable[[List[_UDataFrame]], _VDataFrame]
    # ):

    def reduce(
            self,
            func1: Callable[[_TDataFrame], _UDataFrame],
            func2: Callable[[_UDataFrame], _VDataFrame],
            branch_factor: int = 8,
            final_num_partitions: int = 1,
    ) -> DataFrame[_VDataFrame]:
        return DataFrame(
            lambda i, part_name, part_value: func2(asyncio.run(_read_intermediates(part_value[0], part_value[1]))),
            None, None, DataFrameReduce(self.map_parts(func1), branch_factor, final_num_partitions)
        )


import s3fs


async def _from_s3_helper(
    storage_bucket: StorageBucketSpec,
    object_name: str,
    column_names: Optional[Sequence[str]],
) -> Any:
    print(f"Staring _from_s3 {datetime.datetime.now()}")
    t0 = time.time()

    # ALT2
    # see https://www.programcreek.com/python/example/115410/s3fs.S3FileSystem
    fs = s3fs.S3FileSystem(client_kwargs={"region_name": "us-east-2"})
    result = pd.read_parquet(
        path=f"s3://meadowrun-us-east-2-034389035875/{object_name}",
        filesystem=fs,
        columns=column_names,
    )

    print(f"_from_s3 finished in {time.time() - t0}")

    return result

    # ALT 1
    # client = boto3.client("s3", region_name="us-east-2")
    # body = client.get_object(Bucket="meadowrun-us-east-2-034389035875", Key=object_name)["Body"]
    # with io.BytesIO(body.read()) as buffer:
    #     return pd.read_parquet(buffer)

    # ORIG IMPLEMENTATION
    # async with await storage_bucket.get_storage_bucket_in_cluster() as sb:
    #     with io.BytesIO(await sb.get_bytes(object_name)) as buffer:
    #         return pd.read_parquet(buffer)


async def _to_s3_helper(
    storage_bucket: StorageBucketSpec, part_value: _PandasDF, result_key: str,
) -> None:
    async with await storage_bucket.get_storage_bucket_in_cluster() as sb:
        await sb.write_bytes(part_value.to_parquet(**{
            'coerce_timestamps': 'us',
            'allow_truncated_timestamps': True,
        }), result_key)

    # async def to_pd(self, *run_map_args, **run_map_kwargs) -> _PandasDF:
    #     return pd.concat(await self.)


async def _flatten_to_s3_helper(
    storage_bucket: StorageBucketSpec, part_value: Iterable[Tuple[str, _PandasDF]], result_key_prefix: str,
) -> None:
    async with await storage_bucket.get_storage_bucket_in_cluster() as sb:
        for sub_part_name, sub_part_value in part_value:
            await sb.write_bytes(sub_part_value.to_parquet(), f"{result_key_prefix}/{sub_part_name}")


_TO_BE_NAMED_MAPPING = {
    np.dtype("int32"): "uint32",
    np.dtype("int16"): "uint16",
    np.dtype("int8"): "uint8",
    np.dtype("float32"): "uint32",
    np.dtype("float16"): "uint16",
}


def _as_uint64(s: pd.Series) -> pd.Series:
    # TODO deal with PYTHONHASHSEED
    if s.dtype == "uint64":
        return s
    if s.dtype in ("uint32", "uint16", "uint8", "bool"):
        return s.astype("uint64")
    if s.dtype in ("datetime64[ns]", "float64", "int64"):
        return s.view("uint64")
    if s.dtype in _TO_BE_NAMED_MAPPING:
        return s.view(_TO_BE_NAMED_MAPPING[s.dtype]).astype("uint64")

    # TODO not even clear we should be supporting this
    # ALSO PYTHONHASHSEED needs to be set for this to work
    if s.dtype == "object":
        return s.map(hash).view("uint64")

    # TODO add support for datetime64[ns, <tz>], period[<freq>], category, Sparse*,
    # interval*, string?. Are there more dtypes?
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#basics-dtypes
    raise NotImplementedError(f"Columns of dtype {s.dtype} are not yet supported")


async def _to_groups_helper(
    df: _PandasDF,
    column_names: Sequence[str],
    num_bits: int,
    name: str,
    storage_bucket: StorageBucketSpec,
) -> Dict[int, Tuple[int, int]]:
    t0 = time.time()
    t_serialize = 0
    t_io = 0

    # effectively https://github.com/python/cpython/blob/main/Objects/tupleobject.c#L319
    # vectorized TODO NOT ACTUALLY! ^^^

    # compute hash
    hash_column = pd.Series(
        np.repeat(np.array([2870177450012600261], dtype="uint64"), len(df))
    )

    # TODO document that indexes will get dropped
    df = df.reset_index(drop=True)

    for column_name in column_names:
        hash_column ^= _as_uint64(df.loc[:, column_name])

    # take fibonacci hash and top num_bits bits. See
    # https://probablydance.com/2018/06/16/fibonacci-hashing-the-optimization-that-the-world-forgot-or-a-better-alternative-to-integer-modulo/
    hash_column = np.right_shift(hash_column * 11400714819323198485, 64 - num_bits)

    i = 0
    results = {}
    with io.BytesIO() as buffer:
        for key, group_df in df.groupby(hash_column):
            t0_serialize = time.time()
            group_df.reset_index(drop=True).to_feather(buffer)
            t_serialize += time.time() - t0_serialize
            next_i = buffer.tell()
            results[key] = i, next_i
            i = next_i

        client = boto3.client("s3", region_name="us-east-2")
        t0_io = time.time()
        client.put_object(
            Bucket="meadowrun-us-east-2-034389035875", Key=name, Body=buffer.getvalue()
        )
        t_io += time.time() - t0_io
        # async with await storage_bucket.get_storage_bucket_in_cluster() as sb:
        #     await sb.write_bytes(buffer.getvalue(), name)

    print(f"to_groups completed in {time.time() - t0}. Serialization time {t_serialize}, io time {t_io}. Time is {datetime.datetime.now()}")

    return results


async def _read_groups(
    groups: List[Tuple[str, Tuple[int, int]]], storage_bucket: StorageBucketSpec,
) -> _TDataFrame:
    import datetime
    print(f"Starting _read_groups {datetime.datetime.now()}")

    t0 = time.time()
    t_io = 0
    t_deserialize = 0
    t_reducer_func = 0

    dfs = []
    storage_bucket = S3BucketSpec("us-east-2",
                                  "meadowrun-us-east-2-034389035875")  # TODO FIX, use actual parameter
    async with await storage_bucket.get_storage_bucket_in_cluster() as sb:
        i = 0
        for bs in asyncio.as_completed(
            [sb.get_byte_range(object_key, (byte_range[0], byte_range[1] - 1)) for object_key, byte_range in groups]
        ):
            t0_deserialize = time.time()
            with io.BytesIO(await bs) as buffer:
                dfs.append(pd.read_feather(buffer))

            i += 1
            if i % 25 == 0:
                print(f"Processed {i} {datetime.datetime.now()}")
            t_deserialize += time.time() - t0_deserialize
            # with io.BytesIO(await sb.get_byte_range(object_key, byte_range)) as buffer:
            #     dfs.append(pd.read_parquet(buffer))

    t1 = time.time()
    if len(dfs) == 1:
        result = dfs[0]
    else:
        result = pd.concat(dfs)
    t1 = time.time() - t1

    print(f"_read_groups completed in {time.time() - t0}. io was {t_io}, deserialize was {t_deserialize} pd.concat was {t1} reducer_func time was {t_reducer_func}")

    return result


def _to_intermediate(
    df: _PandasDF,
    name: str,
    storage_bucket: StorageBucketSpec,
) -> None:
    t0 = time.time()
    t_serialize = 0
    t_io = 0

    with io.BytesIO() as buffer:
        t0_serialize = time.time()
        df.reset_index(drop=True).to_feather(buffer)
        t_serialize += time.time() - t0_serialize

        client = boto3.client("s3", region_name="us-east-2")
        t0_io = time.time()
        client.put_object(
            Bucket="meadowrun-us-east-2-034389035875", Key=name, Body=buffer.getvalue()
        )
        t_io += time.time() - t0_io
        # async with await storage_bucket.get_storage_bucket_in_cluster() as sb:
        #     await sb.write_bytes(buffer.getvalue(), name)

    print(f"_to_intermediate completed in {time.time() - t0}. Serialization time {t_serialize}, io time {t_io}. Time is {datetime.datetime.now()}")


# TODO deduplicate with above code
async def _read_intermediates(
    groups: List[str], storage_bucket: StorageBucketSpec,
) -> _TDataFrame:
    import datetime
    print(f"Starting _read_groups {datetime.datetime.now()}")

    t0 = time.time()
    t_io = 0
    t_deserialize = 0
    t_reducer_func = 0

    dfs = []
    storage_bucket = S3BucketSpec("us-east-2",
                                  "meadowrun-us-east-2-034389035875")  # TODO FIX, use actual parameter
    async with await storage_bucket.get_storage_bucket_in_cluster() as sb:
        i = 0
        for bs in asyncio.as_completed(
            [sb.get_bytes(object_key) for object_key in groups]
        ):
            t0_deserialize = time.time()
            with io.BytesIO(await bs) as buffer:
                dfs.append(pd.read_feather(buffer))

            i += 1
            if i % 25 == 0:
                print(f"Processed {i} {datetime.datetime.now()}")
            t_deserialize += time.time() - t0_deserialize
            # with io.BytesIO(await sb.get_byte_range(object_key, byte_range)) as buffer:
            #     dfs.append(pd.read_parquet(buffer))

    t1 = time.time()
    if len(dfs) == 1:
        result = dfs[0]
    else:
        result = pd.concat(dfs)
    t1 = time.time() - t1

    print(f"_read_groups completed in {time.time() - t0}. io was {t_io}, deserialize was {t_deserialize} pd.concat was {t1} reducer_func time was {t_reducer_func}")

    return result


class DataFrameReduce(Generic[_TDataFrame]):
    def __init__(self, base_df: DataFrame, branch_factor: int, final_num_partitions: int):
        self._base_df = base_df
        self._branch_factor = branch_factor
        self._final_num_partitions = final_num_partitions


class DataFrameGroupBy(Generic[_TDataFrame]):
    def __init__(self, base_df: DataFrame, column_names: Sequence[str], num_bits: int):
        self._base_df = base_df
        self._column_names = column_names
        self._num_bits = num_bits

    # TODO would be nice to have apply_raw with an AsyncIterable

    def map_parts(
        self,
        pre_reduce_func: Optional[Callable[[_TDataFrame], _UDataFrame]],
        reduce_finish_func: Callable[[_UDataFrame], _VDataFrame],
    ) -> DataFrame[_VDataFrame]:
        # TODO what if they don't concat correctly?

        if pre_reduce_func is None:
            previous_groupby = self
        else:
            previous_groupby = DataFrameGroupBy(
                self._base_df.map_parts(pre_reduce_func),
                self._column_names,
                self._num_bits,
            )

        return DataFrame(
            lambda i, part_name, part_value: reduce_finish_func(asyncio.run(_read_groups(part_value[0], part_value[1]))),
            None,
            previous_groupby,
            None
        )
        # return self.apply_raw(lambda dfs: pd.concat(dfs)).map_parts(func)

    def apply(
        self,
        func1: Callable[[pd.core.groupby.generic.DataFrameGroupBy], _UDataFrame],
            # TYPE IS NOT RIGHT HERE
        func2: Optional[Callable[[pd.core.groupby.generic.DataFrameGroupBy], _UDataFrame]],
    ) -> DataFrame[_UDataFrame]:
        if func2 is None:
            func2 = func1

        return self.map_parts(
            lambda df: func1(df.groupby(self._column_names, as_index=False)),
            lambda df: func2(df.groupby(self._column_names)),
        )


def sample_str_column(n: int, strlen: int):
    return np.random.choice(list(string.ascii_letters), size=n * strlen).view(
        f"U{strlen}"
    )


def sample_df(num_rows: int):
    return pd.DataFrame(
        {
            "int": np.random.randint(0, 250, num_rows),
            "float": np.random.rand(num_rows),
            "str": sample_str_column(num_rows, 5),
        }
    )


async def test_adhoc():
    import meadowrun

    # storage_bucket = meadowrun.GenericStorageBucketSpec("meadowrunbucket",
    #                                           "http://localhost:9000",
    #                                           "http://minio-service:9000",
    #                                           "minio-credentials", )
    # host = meadowrun.Kubernetes(
    #     storage_bucket,
    #     kube_config_context="minikube",
    # )


    storage_bucket = S3BucketSpec("us-east-2", "meadowrun-us-east-2-034389035875")
    host = meadowrun.AllocEC2Instance()
    resources = meadowrun.Resources(1, 2)

    deployment = meadowrun.Deployment.mirror_local(
        # interpreter=meadowrun.PreinstalledInterpreter(MEADOWRUN_INTERPRETER)
        # interpreter=meadowrun.PipRequirementsFile(
        #     r"C:\source\scratch\directories\req-pd.txt", "3.9"
        # )
        interpreter=meadowrun.LocalPipInterpreter(sys.executable)
    )

    the_args = dict(
        host=host,
        deployment=deployment,
        num_concurrent_tasks=1,
        resources_per_task=resources,
    )

    # mddf = DataFrame.from_map(lambda i, arg: sample_df(500), range(5))
    # print(await mddf.map_parts_with_name(lambda i, df: df.sum()).compute(**the_args))
    # print(await mddf.to_s3("test_mdf", storage_bucket, **the_args))

    def foo(i, df):
        print(f">>> {i} {df}")
        return df

    def foo2(df):
        print(f">>> {df}")
        return df

    mddf = await DataFrame.from_s3("yellow_taxi", storage_bucket)

    # print((await mddf.map_parts_with_name(foo).map_parts_with_name(foo).compute(**the_args))[0])
    # print((await mddf.map_parts(foo2).map_parts(foo2).compute(**the_args))[0])

    #
    def add_cols(df):
        df["strkey"] = df["str"].str.slice(0, 1)
        return df

    print(
        await mddf.map_parts(add_cols)
        .groupby(["strkey"], 8)
        .apply(lambda df: df.sum())
        .to_pd(storage_bucket=storage_bucket, **the_args)
    )

    # print(await mddf.to_pd(**the_args))


async def test_benchmark1(n):
    import meadowrun
    from meadowrun.k8s_integration.storage_spec import S3BucketSpec

    storage_bucket = S3BucketSpec("us-east-2", "meadowrun-us-east-2-034389035875")
    host = meadowrun.AllocEC2Instance()
    resources = meadowrun.Resources(1, 8)
    deployment = await meadowrun.Deployment.mirror_local(
        interpreter=meadowrun.LocalPipInterpreter(sys.executable)
    )

    the_args = dict(
        num_concurrent_tasks=8,
        host=host,
        deployment=deployment,
        resources_per_task=resources,
        # retry_with_more_memory=True,
        # max_num_task_attempts=2,
    )

    mddf = await DataFrame.from_s3(
        "yellow_taxi",
        storage_bucket,
        # column_names=["PULocationID", "DOLocationID", "tip_amount", "total_amount"]
    )


    mddf = mddf.parts_iloc[-n:]

    print(len(mddf.part_names))

    # print(await mddf.map_parts(lambda df: df.head()).compute(**the_args))
    #
    # print(mddf.part_names)
    #
    # print(await mddf.map_parts(
    #     lambda df: (
    #         len(pickle.dumps(df)),
    #         len(pickle.dumps(df.assign(trip_distance_rounded=(df["trip_distance"] * 10).round().astype(int)).groupby(["PULocationID", "DOLocationID", "trip_distance_rounded"]).sum())),
    #         len(pickle.dumps(df.groupby(["PULocationID", "DOLocationID"]).sum())),
    #         len(pickle.dumps(df.groupby(["PULocationID", "DOLocationID", "passenger_count"]).sum())),
    #     )
    # ).compute(**the_args))

    t0 = time.time()

    print(
        await mddf.groupby(["PULocationID", "DOLocationID"], 1)
        .apply(lambda df: df.sum())
        .to_pd(storage_bucket=storage_bucket, **the_args)
    )

    print(time.time() - t0)


async def test_generate_data2():
    import meadowrun
    from meadowrun.k8s_integration.storage_spec import S3BucketSpec

    storage_bucket = S3BucketSpec("us-east-2", "meadowrun-us-east-2-034389035875")
    host = meadowrun.AllocEC2Instance()
    resources = meadowrun.Resources(1, 3)
    deployment = await meadowrun.Deployment.mirror_local(
        interpreter=meadowrun.LocalPipInterpreter(sys.executable)
    )

    the_args = dict(
        num_concurrent_tasks=16,
        host=host,
        deployment=deployment,
        resources_per_task=resources,
    )

    mddf = await DataFrame.from_s3(
        "yellow_taxi",
        storage_bucket,
        # column_names=["PULocationID", "DOLocationID", "tip_amount", "total_amount"]
    )


    # mddf = mddf.parts_iloc[-1:]

    print(len(mddf.part_names))

    def foo(x):
        x["date"] = pd.to_datetime(x["tpep_pickup_datetime"].dt.date)
        return [(str(date.date()), gr) for date, gr in x.groupby("date")]

    t0 = time.time()

    # print(await mddf.map_parts(foo).compute(**the_args))
    result = await mddf.map_parts(foo).flatten_to_s3("yellow_taxi_by_day", storage_bucket=storage_bucket, **the_args)
    print(result)

    print(time.time() - t0)
    return result


async def test_benchmark2(n):
    import meadowrun
    from meadowrun.k8s_integration.storage_spec import S3BucketSpec

    storage_bucket = S3BucketSpec("us-east-2", "meadowrun-us-east-2-034389035875")
    host = meadowrun.AllocEC2Instance()
    resources = meadowrun.Resources(1, 2)
    deployment = await meadowrun.Deployment.mirror_local(
        interpreter=meadowrun.LocalPipInterpreter(sys.executable)
    )

    the_args = dict(
        num_concurrent_tasks=128,
        host=host,
        deployment=deployment,
        resources_per_task=resources,
        # retry_with_more_memory=True,
        # max_num_task_attempts=2,
    )

    mddf = await DataFrame.from_s3(
        "yellow_taxi_by_day",
        storage_bucket,
        # column_names=["PULocationID", "DOLocationID", "tip_amount", "total_amount"]
    )


    mddf = mddf.parts_iloc[-n:]

    print(len(mddf.part_names))

    # print(await mddf.map_parts(lambda df: df.head()).compute(**the_args))
    #
    # print(mddf.part_names)
    #
    # print(await mddf.map_parts(
    #     lambda df: (
    #         len(pickle.dumps(df)),
    #         len(pickle.dumps(df.assign(trip_distance_rounded=(df["trip_distance"] * 10).round().astype(int)).groupby(["PULocationID", "DOLocationID", "trip_distance_rounded"]).sum())),
    #         len(pickle.dumps(df.groupby(["PULocationID", "DOLocationID"]).sum())),
    #         len(pickle.dumps(df.groupby(["PULocationID", "DOLocationID", "passenger_count"]).sum())),
    #     )
    # ).compute(**the_args))

    t0 = time.time()

    print(
        await mddf.groupby(["PULocationID", "DOLocationID"], 1)
        .apply(lambda df: df.sum())
        .to_pd(storage_bucket=storage_bucket, **the_args)
    )

    print(time.time() - t0)


async def fake_checkins():
    storage_bucket = S3BucketSpec("us-east-2", "meadowrun-us-east-2-034389035875")
    host = meadowrun.AllocEC2Instance()
    resources = meadowrun.Resources(1, 3)

    deployment = meadowrun.Deployment.mirror_local(
        # interpreter=meadowrun.PreinstalledInterpreter(MEADOWRUN_INTERPRETER)
        # interpreter=meadowrun.PipRequirementsFile(
        #     r"C:\source\scratch\directories\req-pd.txt", "3.9"
        # )
        interpreter=meadowrun.LocalPipInterpreter(sys.executable)
    )

    the_args = dict(
        host=host,
        deployment=deployment,
        num_concurrent_tasks=70,
        resources_per_task=resources,
    )

    # this data

    num_all_users = 300_000_000
    num_all_locations = 8_000_000
    all_states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

    # date_arg = pd.Timestamp("2022-01-01")

    def foo(i, date, date_arg):
        num_rows = np.random.randint(10_000_000, 17_500_000)
        dts = (date_arg + pd.Series(np.random.random(num_rows) * 60 * 24).round(3) * pd.Timedelta(seconds=1)).sort_values()
        user_ids = np.random.randint(0, num_all_users, num_rows)
        location_ids = np.random.randint(0, num_all_locations, num_rows)
        states = np.random.choice(all_states, num_rows)

        return pd.DataFrame.from_dict({
            "dt": dts,
            "user_id": user_ids,
            "location_id": location_ids,
            "states": states,
        }).reset_index(drop=True)

    await DataFrame.from_map(
        foo, [(str(d.date()), d) for d in pd.date_range("2021-01-01", "2022-12-31")]
    ).to_s3("fake_checkins", storage_bucket, **the_args)


async def test(n):
    storage_bucket = S3BucketSpec("us-east-2", "meadowrun-us-east-2-034389035875")
    host = meadowrun.AllocEC2Instance()
    resources = meadowrun.Resources(1.5, 4)

    deployment = await meadowrun.Deployment.mirror_local(
        interpreter=meadowrun.LocalPipInterpreter(sys.executable),
        environment_variables={"PYTHONHASHSEED": "0"}
    )

    the_args = dict(
        host=host,
        deployment=deployment,
        num_concurrent_tasks=64,
        resources_per_task=resources,
    )

    mddf = await DataFrame.from_s3(
        "fake_checkins",
        storage_bucket)
    #
    # def foo(df):
    #     result = (len(df), len(pickle.dumps(df)), len(pickle.dumps(df.groupby("states").count())))
    #     import resource
    #     return result, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # print(await mddf.map_parts(foo).parts_iloc[:64].compute(**the_args))

    mddf = mddf.parts_iloc[-n:]

    t0 = time.time()

    def foo(df):
        t1 = time.time()
        stuff = df.groupby("user_id", as_index=False)["location_id"].nunique()
        user_ids = stuff.loc[stuff["location_id"] > 45, "user_id"]
        result = df[df["user_id"].isin(user_ids)]
        print(f"foo finished in {time.time() - t1} {datetime.datetime.now()}")
        return result

    result = await mddf \
        .groupby(["user_id"], n) \
        .map_parts(lambda df: df, foo).compute(storage_bucket=storage_bucket, **the_args)

    print(time.time() - t0)

    result_df = pd.concat(result).sort_values(["user_id", "location_id"])
    print(result_df)
    pd.to_pickle(result_df, "temp.pkl")


async def temp_test():
    storage_bucket = S3BucketSpec("us-east-2", "meadowrun-us-east-2-034389035875")
    host = meadowrun.AllocEC2Instance()
    resources = meadowrun.Resources(1.5, 8)

    deployment = await meadowrun.Deployment.mirror_local(
        interpreter=meadowrun.LocalPipInterpreter(sys.executable),
        environment_variables={"PYTHONHASHSEED": "0"}
    )

    the_args = dict(
        host=host,
        deployment=deployment,
        num_concurrent_tasks=64,
        resources_per_task=resources,
    )

    mddf = await DataFrame.from_s3(
        "fake_checkins",
        storage_bucket)

    mddf = mddf.parts_iloc[-512:]

    t1 = time.time()
    # print(await mddf.map_parts(foo).compute(**the_args))

    # print(await mddf.groupby(["location_id"], 1)\
    #     # .map_parts2(
    #     #     lambda df: df.groupby(["location_id"], as_index=False)[["dt"]].count(),
    #     #     lambda df1, df2: pd.concat([df1, df2]).groupby(["location_id"], as_index=False).sum(),
    #     #     lambda df: df
    #     # )
    # .apply(
    #     lambda gr: gr[["dt"]].count(),
    #     lambda gr: gr.sum()
    # )

    print(await mddf.reduce(lambda df: df.groupby(["location_id"], as_index=False)[["dt"]].count(),
                            lambda df: df.groupby(["location_id"], as_index=False).sum(),
                            )
          .compute(storage_bucket=storage_bucket, **the_args))

    print(time.time() - t1)


if __name__ == "__main__":
    asyncio.run(test(256))
    print("That was 256")
    asyncio.run(test(128))
    print("That was 128")
    asyncio.run(test(64))
    print("That was 64")
