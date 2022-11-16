from __future__ import annotations

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
    AsyncIterable,
)

import boto3

from meadowrun import StorageBucketSpec
from meadowrun.abstract_storage_bucket import AbstractStorageBucket
from meadowrun.run_job import run_map

try:
    import pandas as pd
    import pandas.core.groupby.generic
    import numpy as np
except ImportError:
    print("Using mdf without pandas installed is not recommended")

_T = TypeVar("_T")
_U = TypeVar("_U")
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
        partition_from_argument: Callable[[int, _TDataFrame], _UDataFrame],
        arguments: Optional[Sequence[Tuple[int, _TDataFrame]]],
        previous_groupby: Optional[DataFrameGroupBy],
    ):
        # if not (bool(arguments) ^ bool(previous_groupby)):
        #     raise ValueError("")
        self._partition_from_argument = partition_from_argument
        self._arguments = arguments
        self._previous_groupby = previous_groupby

    @classmethod
    def from_map(
        cls,
        partition_from_argument: Callable[[int, _T], _TDataFrame],
        arguments: Sequence[_T],
    ) -> DataFrame:
        return cls(partition_from_argument, list(enumerate(arguments)), None)

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
        object_names = [name for name in object_names if not name.endswith("/")]
        object_names.sort()  # should be already sorted

        return cls.from_map(
            lambda i, object_name: asyncio.run(
                _from_s3_helper(storage_bucket, object_name, column_names)
            ),
            object_names,
        )

    @_square_bracket_property
    def parts_loc(self, labels: Any) -> DataFrame[_UDataFrame]:
        # TODO this doesn't work right now because of the stupid enumerate

        if self._arguments is None:
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
            self._partition_from_argument,
            [
                argument
                for argument in self._arguments
                if (labels.start is None or labels.start <= argument)
                and (labels.stop is None or labels.stop >= argument)
            ],
            self._previous_groupby,
        )

    @_square_bracket_property
    def parts_iloc(self, labels: Any) -> DataFrame[_UDataFrame]:
        if self._arguments is None:
            raise NotImplementedError(
                "Cannot call parts_iloc on the result of a groupby"
            )
        # TODO this is a bit sketchy in the case where you do e.g.
        # mddf.map_parts(func_with_side_effects).parts_between(...) ... This gets treated
        # as mddf.parts_between().map_parts(func_with_side_effects)

        if isinstance(labels, slice):
            new_arguments = self._arguments[labels]
        else:
            new_arguments = [self._arguments[labels]]

        return DataFrame(
            self._partition_from_argument,
            new_arguments,
            self._previous_groupby,
        )

    @property
    def part_names(self) -> List[_T]:
        if self._arguments is None:
            raise NotImplementedError(
                "Cannot call part_names on the result of a groupby"
            )
        # TODO I think we should rename _arguments to part_name everywhere
        return [arg for i, arg in self._arguments]

    def map_parts_with_index(
        self, func: Callable[[int, _UDataFrame], _VDataFrame]
    ) -> DataFrame[_VDataFrame]:
        return DataFrame(
            lambda i, arg: func(i, self._partition_from_argument(i, arg)),
            self._arguments,
            self._previous_groupby,
        )

    def map_parts(
        self, func: Callable[[_UDataFrame], _VDataFrame]
    ) -> DataFrame[_VDataFrame]:
        return self.map_parts_with_index(lambda i, arg: func(arg))

    async def compute_with_index(
        self, *run_map_args, **run_map_kwargs
    ) -> Sequence[Tuple[int, _PandasDF]]:
        arguments_list = list(self._arguments)
        # TODO make this work with _previous_groupby
        return [
            (arg, part)
            for ((i, arg), part) in zip(
                arguments_list,
                await run_map(
                    lambda arg: self._partition_from_argument(*arg),
                    arguments_list,
                    *run_map_args,
                    **run_map_kwargs,
                ),
            )
        ]

    async def compute(self, *run_map_args, **run_map_kwargs) -> Sequence[_TDataFrame]:
        if self._arguments is not None:
            arguments = list(self._arguments)
        elif self._previous_groupby is not None:
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
            arguments = list(
                (group_index, (group, storage_bucket))
                for group_index, group in groups.items()
            )
        else:
            raise ValueError(
                "Both _arguments and _previous_groupby are None which should never "
                "happen!"
            )

        return await run_map(
            lambda arg: self._partition_from_argument(*arg),
            arguments,
            *run_map_args,
            **run_map_kwargs,
        )

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
        await self.map_parts_with_index(
            lambda i, part: asyncio.run(_to_s3_helper(storage_bucket, part, name, i))
        ).compute(*run_map_args, **run_map_kwargs)

    async def flatten_to_s3(
            self,
            name: str,
            storage_bucket: StorageBucketSpec,
            *run_map_args,
            **run_map_kwargs,
    ):
        pass

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
        groups_by_source = await self.map_parts_with_index(
            lambda i, part: (
                i,
                asyncio.run(
                    _to_groups_helper(
                        part,
                        column_names,
                        num_bits,
                        f"{job_id}/{i}.pkl",
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
                    (f"{job_id}/{source_index}.pkl", byte_range)
                )

        return groups_by_dest

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


import s3fs


async def _from_s3_helper(
    storage_bucket: StorageBucketSpec,
    object_name: str,
    column_names: Optional[Sequence[str]],
) -> Any:
    # ALT2
    # see https://www.programcreek.com/python/example/115410/s3fs.S3FileSystem
    fs = s3fs.S3FileSystem(client_kwargs={"region_name": "us-east-2"})
    return pd.read_parquet(
        path=f"s3://meadowrun-us-east-2-034389035875/{object_name}",
        filesystem=fs,
        columns=column_names,
    )

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
    storage_bucket: StorageBucketSpec, part: Any, name: str, i: int
):
    async with await storage_bucket.get_storage_bucket_in_cluster() as sb:
        await sb.write_bytes(part.to_parquet(), f"{name}/{i}.parquet")

    # async def to_pd(self, *run_map_args, **run_map_kwargs) -> _PandasDF:
    #     return pd.concat(await self.)


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
    # effectively https://github.com/python/cpython/blob/main/Objects/tupleobject.c#L319
    # vectorized TODO NOT ACTUALLY! ^^^

    # compute hash
    hash_column = pd.Series(
        np.repeat(np.array([2870177450012600261], dtype="uint64"), len(df))
    )
    for column_name in column_names:
        hash_column ^= _as_uint64(df.loc[:, column_name])

    # take fibonacci hash and top num_bits bits. See
    # https://probablydance.com/2018/06/16/fibonacci-hashing-the-optimization-that-the-world-forgot-or-a-better-alternative-to-integer-modulo/
    hash_column = np.right_shift(hash_column * 11400714819323198485, 64 - num_bits)

    i = 0
    results = {}
    with io.BytesIO() as buffer:
        for key, group_df in df.groupby(hash_column):
            group_df.to_parquet(buffer)
            next_i = buffer.tell()
            results[key] = i, next_i
            i = next_i

        client = boto3.client("s3", region_name="us-east-2")
        client.put_object(
            Bucket="meadowrun-us-east-2-034389035875", Key=name, Body=buffer.getvalue()
        )
        # async with await storage_bucket.get_storage_bucket_in_cluster() as sb:
        #     await sb.write_bytes(buffer.getvalue(), name)

    return results


async def _read_groups(
    groups: Tuple[str, Tuple[int, int]], storage_bucket: StorageBucketSpec
) -> _TDataFrame:
    dfs = []

    client = boto3.client("s3", region_name="us-east-2")
    # async with await storage_bucket.get_storage_bucket_in_cluster() as sb:
    for object_key, byte_range in groups:
        body = client.get_object(
            Bucket="meadowrun-us-east-2-034389035875",
            Key=object_key,
            Range=f"bytes={byte_range[0]}-{byte_range[1] - 1}",
        )["Body"]
        bs = body.read()
        print(f"{object_key} {byte_range} {len(bs)}")
        with io.BytesIO(bs) as buffer:
            dfs.append(pd.read_parquet(buffer))
        # with io.BytesIO(await sb.get_byte_range(object_key, byte_range)) as buffer:
        #     dfs.append(pd.read_parquet(buffer))
    return pd.concat(dfs)


class DataFrameGroupBy(Generic[_TDataFrame]):
    def __init__(self, base_df: DataFrame, column_names: Sequence[str], num_bits: int):
        self._base_df = base_df
        self._column_names = column_names
        self._num_bits = num_bits

    # TODO would be nice to have apply_raw with an AsyncIterable

    def map_parts(
        self,
        pre_reduce_func: Optional[Callable[[_TDataFrame], _UDataFrame]],
        reducer_func: Callable[[_UDataFrame], _VDataFrame],
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
            lambda i, arg: reducer_func(asyncio.run(_read_groups(arg[0], arg[1]))),
            None,
            previous_groupby,
        )
        # return self.apply_raw(lambda dfs: pd.concat(dfs)).map_parts(func)

    def apply(
        self, func: Callable[[pd.core.groupby.generic.DataFrameGroupBy], _UDataFrame]
    ) -> DataFrame[_UDataFrame]:
        return self.map_parts(
            lambda df: func(df.groupby(self._column_names, as_index=False)),
            lambda df: func(df.groupby(self._column_names)),
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


async def test():
    import meadowrun

    # storage_bucket = meadowrun.GenericStorageBucketSpec("meadowrunbucket",
    #                                           "http://localhost:9000",
    #                                           "http://minio-service:9000",
    #                                           "minio-credentials", )
    # host = meadowrun.Kubernetes(
    #     storage_bucket,
    #     kube_config_context="minikube",
    # )

    from meadowrun.k8s_integration.storage_spec import S3BucketSpec

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
    # print(await mddf.map_parts_with_index(lambda i, df: df.sum()).compute(**the_args))
    # print(await mddf.to_s3("test_mdf", storage_bucket, **the_args))

    def foo(i, df):
        print(f">>> {i} {df}")
        return df

    def foo2(df):
        print(f">>> {df}")
        return df

    mddf = await DataFrame.from_s3("yellow_taxi", storage_bucket)

    # print((await mddf.map_parts_with_index(foo).map_parts_with_index(foo).compute(**the_args))[0])
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


async def test2(n):
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


if __name__ == "__main__":
    asyncio.run(test2(999))
    asyncio.run(test2(999))
    asyncio.run(test2(112))
    asyncio.run(test2(96))
    asyncio.run(test2(80))
    asyncio.run(test2(64))
    asyncio.run(test2(48))
    asyncio.run(test2(32))
    asyncio.run(test2(16))
