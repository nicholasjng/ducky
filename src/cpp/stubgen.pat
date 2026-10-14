# nanobind stubgen pattern file.
#
# We use this to emit ``@typing.overload`` variants for methods whose return
# type depends on a boolean argument — something nb::sig can't express
# directly because each .def has a single signature string.

pl:
    \from typing import overload, Literal
    \import polars
    @overload
    def pl(self, lazy: Literal[False] = False) -> polars.DataFrame: ...
    @overload
    def pl(self, lazy: Literal[True]) -> polars.LazyFrame: ...
    def pl(self, lazy: bool = False) -> polars.DataFrame | polars.LazyFrame:
        \doc

# __exit__: rewrite the traceback annotation to use `TracebackType` directly.
# The default sig emits ``types.TracebackType``, which collides with the
# ``types`` property on Appender (and would similarly collide on any class
# that exposes a ``types`` member) — at class scope, ``types`` resolves to
# the descriptor rather than the imported module.
\.__exit__$:
    \from types import TracebackType
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        \doc

# iter_batches: dict values are bare ndarrays when with_validity=False,
# (values, mask) tuples when with_validity=True.
\.iter_batches$:
    @overload
    def iter_batches(self, columns: Iterable[str] | None = None, with_validity: Literal[False] = False) -> Iterator[dict[str, numpy.ndarray]]: ...
    @overload
    def iter_batches(self, columns: Iterable[str] | None = None, *, with_validity: Literal[True]) -> Iterator[dict[str, tuple[numpy.ndarray, numpy.ndarray | None]]]: ...
    def iter_batches(self, columns: Iterable[str] | None = None, with_validity: bool = False) -> Iterator[dict[str, numpy.ndarray] | dict[str, tuple[numpy.ndarray, numpy.ndarray | None]]]:
        \doc

# Chunk.mlx: the default sig emits `mlx.core.array`, which collides with the
# `mlx` method itself -- same class of bug as __exit__ above (at class scope,
# the bare name `mlx` resolves to the method, not the imported module).
# Import mlx.core under its conventional alias so the annotation never needs
# the bare `mlx` token.
\.mlx$:
    \import mlx.core as mx
    def mlx(self, key: int | str) -> mx.array:
        \doc

# iter_batches_mlx / to_mlx: use the same alias as Chunk.mlx, so the stub
# imports mlx.core once, under one name.
\.iter_batches_mlx$:
    \import mlx.core as mx
    def iter_batches_mlx(self, columns: Iterable[str] | None = None) -> Iterator[dict[str, mx.array]]:
        \doc

\.to_mlx$:
    \import mlx.core as mx
    def to_mlx(self, columns: Iterable[str] | None = None) -> dict[str, mx.array]:
        \doc
