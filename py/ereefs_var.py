from argparse import ArgumentParser
from pathlib import Path

from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from virtualizarr import open_virtual_mfdataset
from virtualizarr.parsers import HDFParser


DEFAULT_SOURCE_PREFIX = "/g/data/fx3/gbr4_v2"
DEFAULT_PUBLIC_PREFIX = "https://thredds.nci.org.au/thredds/fileServer/fx3/gbr4_v2/"
DEFAULT_LOADABLE_VARS = ["time", "zc", "latitude", "longitude"]
DEFAULT_DROP_VARS = ["botz"]


parser = ArgumentParser(
    description=(
        "Build a parquet-backed kerchunk store for one eReefs/GBR4 variable "
        "from on-disk NCI files, with optional path rewriting for THREDDS publication."
    )
)
parser.add_argument("varlabel", help="Variable to keep, e.g. temp or salt")
parser.add_argument("outfile", help="Output parquet directory")
parser.add_argument("paths", nargs="+", help="One or more local NetCDF paths, e.g. /g/data/.../gbr4_simple_2024-01-16.nc")
parser.add_argument(
    "--source-prefix",
    default=DEFAULT_SOURCE_PREFIX,
    help=f"Local disk prefix for the source files (default: {DEFAULT_SOURCE_PREFIX})",
)
parser.add_argument(
    "--strip-prefix",
    default=DEFAULT_SOURCE_PREFIX.rstrip("/") + "/",
    help="Path prefix to strip out of parquet refs after writing",
)
parser.add_argument(
    "--add-prefix",
    default=DEFAULT_PUBLIC_PREFIX,
    help=f"Replacement prefix to insert into parquet refs (default: {DEFAULT_PUBLIC_PREFIX})",
)
parser.add_argument(
    "--no-rewrite-refs",
    action="store_true",
    help="Skip the post-write parquet ref rewrite step",
)
parser.add_argument(
    "--loadable-vars",
    nargs="*",
    default=DEFAULT_LOADABLE_VARS,
    help="Coordinate variables to materialize into the virtual dataset",
)
parser.add_argument(
    "--drop-vars",
    nargs="*",
    default=DEFAULT_DROP_VARS,
    help="Variables to drop before writing the kerchunk parquet store",
)
args = parser.parse_args()

outfile = Path(args.outfile)
if outfile.exists():
    raise SystemExit(f"outfile already exists: {outfile}")

paths = [Path(path).expanduser().resolve() for path in args.paths]
missing = [str(path) for path in paths if not path.exists()]
if missing:
    raise SystemExit(f"source files not found: {missing}")

for path in paths:
    if not str(path).startswith(args.source_prefix.rstrip("/") + "/"):
        raise SystemExit(
            f"source file {path} is outside --source-prefix {args.source_prefix!r}"
        )

registry = ObjectStoreRegistry({"file:///": LocalStore(prefix=args.source_prefix)})
parser = HDFParser()

vds = open_virtual_mfdataset(
    [path.as_uri() for path in paths],
    registry=registry,
    parser=parser,
    combine="nested",
    concat_dim="time",
    parallel=False,
    drop_variables=args.drop_vars,
    loadable_variables=args.loadable_vars,
)

if args.varlabel not in vds:
    raise SystemExit(
        f"variable {args.varlabel!r} not found; available variables: {list(vds.data_vars)}"
    )

vds[[args.varlabel]].vz.to_kerchunk(str(outfile), format="parquet")
print(f"wrote {outfile}")


def rewrite_parquet_paths(root: Path, strip_prefix: str, add_prefix: str) -> int:
    try:
        import polars as pl
    except ImportError as exc:
        raise SystemExit(
            "polars is required for parquet path rewriting; install it or rerun with --no-rewrite-refs"
        ) from exc

    if strip_prefix == add_prefix:
        return 0

    updates = 0
    for parquet_file in root.rglob("*.parq"):
        refs = pl.read_parquet(parquet_file)
        if "path" not in refs.columns:
            continue

        has_updates = refs.select(
            pl.col("path").str.starts_with(strip_prefix).fill_null(False).any()
        ).item()
        if not has_updates:
            continue

        refs.with_columns(
            pl.when(pl.col("path").str.starts_with(strip_prefix).fill_null(False))
            .then(
                pl.concat_str(
                    [pl.lit(add_prefix), pl.col("path").str.slice(len(strip_prefix))]
                )
            )
            .otherwise(pl.col("path"))
            .alias("path")
        ).write_parquet(parquet_file)
        updates += 1

    return updates


if args.no_rewrite_refs:
    print("skipped parquet ref rewrite (--no-rewrite-refs)")
else:
    updates = rewrite_parquet_paths(outfile, args.strip_prefix, args.add_prefix)
    print(
        "rewrote parquet refs "
        f"from {args.strip_prefix!r} to {args.add_prefix!r} in {updates} files"
    )
