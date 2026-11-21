# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ducky",
#     "jax>=0.10.1",
#     "mlx>=0.31.2; sys_platform == 'darwin'",
# ]
#
# [tool.uv.sources]
# ducky = { path = "../", editable = true }
# ///
"""Train a small character-level transformer on text that DuckDB tokenizes.

The language-model step after ``tabular_mlp.py``. DuckDB is the text pipeline
and the vector store around the model:

1. ``read_text`` loads a corpus by glob: the Python standard library's sources;
2. SQL splits it into characters, builds the vocabulary, and packs the token
   stream into fixed-length training windows, one ``INTEGER[block + 1]`` array
   per row, which arrive as one ``(windows, block + 1)`` array;
3. JAX or MLX trains a decoder-only transformer on the windows;
4. each file's embedding goes back into DuckDB as a ``FLOAT[d]`` column, and
   ``array_cosine_similarity`` answers nearest-neighbour queries, with the
   query vector bound as a NumPy parameter.

Run with ``--backend jax`` or ``--backend mlx``.
"""

from __future__ import annotations

import argparse
import sysconfig
import time
from typing import Any

import numpy as np

import ducky

VOCAB = 96  # the most frequent characters; the rest share one id
BLOCK = 64
DIM = 96
HEADS = 4
LAYERS = 2
BATCH = 64
LEARNING_RATE = 3e-3


def rows(con: ducky.Connection, query: str, parameters: Any = None) -> list[tuple[Any, ...]]:
    return con.execute(query, parameters).fetchall()


def prepare(con: ducky.Connection, files: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Tokenize the corpus in SQL; return train windows, validation windows and the vocabulary."""
    glob = sysconfig.get_paths()["stdlib"] + "/*.py"
    script = f"""
        CREATE TABLE docs AS
        SELECT row_number() OVER (ORDER BY filename) - 1 AS doc,
               parse_filename(filename) AS name, content
        FROM (FROM read_text('{glob}') ORDER BY size DESC LIMIT {files});

        -- One row per character, in order.
        CREATE TABLE chars AS
        SELECT doc, unnest(string_split(content, '')) AS ch,
               generate_subscripts(string_split(content, ''), 1) AS pos
        FROM docs;

        -- Ids by frequency. Id {VOCAB - 1} stands for every rare character.
        CREATE TABLE vocab AS
        SELECT ch, least(row_number() OVER (ORDER BY count(*) DESC, ch) - 1, {VOCAB - 1}) AS id
        FROM chars GROUP BY ch;

        CREATE TABLE tokens AS
        SELECT doc, pos, id::INTEGER AS id FROM chars JOIN vocab USING (ch);

        -- Non-overlapping windows of {BLOCK + 1} tokens: inputs and shifted targets.
        CREATE TABLE windows AS
        SELECT doc, list(id ORDER BY pos)::INTEGER[{BLOCK + 1}] AS ids
        FROM tokens GROUP BY doc, (pos - 1) // {BLOCK + 1}
        HAVING count(*) = {BLOCK + 1};
        """
    for statement in script.split(";"):
        if statement.strip():
            con.execute(statement)
    # Hold out whole files, so validation text is unseen.
    train = con.sql("SELECT ids FROM windows WHERE hash(doc) % 10 < 9").to_numpy()["ids"]
    val = con.sql("SELECT ids FROM windows WHERE hash(doc) % 10 >= 9").to_numpy()["ids"]
    chars = [ch for (ch,) in rows(con, f"SELECT ch FROM vocab WHERE id < {VOCAB - 1} ORDER BY id")]
    return train, val, [*chars, "?"]


def train_jax(train: np.ndarray, val: np.ndarray, steps: int) -> tuple[np.ndarray, Any]:
    import jax
    import jax.numpy as jnp

    def init(key: jax.Array) -> dict[str, Any]:
        def dense(key: jax.Array, m: int, n: int) -> jax.Array:
            return jax.random.normal(key, (m, n)) * 0.02

        keys = iter(jax.random.split(key, 3 + 4 * LAYERS))
        blocks = [
            {
                "qkv": dense(next(keys), DIM, 3 * DIM),
                "proj": dense(next(keys), DIM, DIM),
                "up": dense(next(keys), DIM, 4 * DIM),
                "down": dense(next(keys), 4 * DIM, DIM),
                "g1": jnp.ones(DIM),
                "g2": jnp.ones(DIM),
            }
            for _ in range(LAYERS)
        ]
        return {
            "tok": dense(next(keys), VOCAB, DIM),
            "pos": dense(next(keys), BLOCK, DIM),
            "head": dense(next(keys), DIM, VOCAB),
            "g": jnp.ones(DIM),
            "blocks": blocks,
        }

    def norm(x: jax.Array, gain: jax.Array) -> jax.Array:
        x = x - x.mean(-1, keepdims=True)
        return gain * x / jnp.sqrt((x * x).mean(-1, keepdims=True) + 1e-5)

    def forward(params: dict[str, Any], ids: jax.Array) -> jax.Array:
        t = ids.shape[1]
        x = params["tok"][ids] + params["pos"][:t]
        causal = jnp.tril(jnp.ones((t, t), bool))
        for block in params["blocks"]:
            h = norm(x, block["g1"]) @ block["qkv"]
            q, k, v = (
                part.reshape(-1, t, HEADS, DIM // HEADS).transpose(0, 2, 1, 3)
                for part in jnp.split(h, 3, axis=-1)
            )
            scores = q @ k.transpose(0, 1, 3, 2) / jnp.sqrt(DIM // HEADS)
            weights = jax.nn.softmax(jnp.where(causal, scores, -jnp.inf), axis=-1)
            x = x + (weights @ v).transpose(0, 2, 1, 3).reshape(-1, t, DIM) @ block["proj"]
            x = x + jax.nn.gelu(norm(x, block["g2"]) @ block["up"]) @ block["down"]
        return norm(x, params["g"]) @ params["head"]

    def loss(params: dict[str, Any], batch: jax.Array) -> jax.Array:
        logp = jax.nn.log_softmax(forward(params, batch[:, :-1]))
        return -jnp.take_along_axis(logp, batch[:, 1:, None], axis=-1).mean()

    @jax.jit
    def step(state: Any, batch: jax.Array) -> tuple[Any, jax.Array]:
        params, m, v, t = state
        value, grads = jax.value_and_grad(loss)(params, batch)
        t = t + 1
        m = jax.tree.map(lambda a, g: 0.9 * a + 0.1 * g, m, grads)
        v = jax.tree.map(lambda a, g: 0.99 * a + 0.01 * g * g, v, grads)
        scale = LEARNING_RATE * jnp.sqrt(1 - 0.99**t) / (1 - 0.9**t)
        params = jax.tree.map(lambda p, a, b: p - scale * a / (jnp.sqrt(b) + 1e-8), params, m, v)
        return (params, m, v, t), value

    evaluate = jax.jit(loss)
    params = init(jax.random.PRNGKey(0))
    zeros = jax.tree.map(jnp.zeros_like, params)
    state = (params, zeros, zeros, jnp.zeros((), jnp.int32))
    data, held_out = jnp.asarray(train), jnp.asarray(val[:512])
    rng = np.random.default_rng(0)
    start = time.perf_counter()
    for i in range(1, steps + 1):
        state, value = step(state, data[rng.integers(0, len(train), BATCH)])
        if i % 100 == 0 or i == steps:
            print(
                f"step {i:>4} | train loss {float(value):.3f} | val loss "
                f"{float(evaluate(state[0], held_out)):.3f} | {time.perf_counter() - start:.1f} s"
            )
    params = state[0]

    def next_logits(ids: np.ndarray) -> np.ndarray:
        return np.asarray(forward(params, jnp.asarray(ids[None, -BLOCK:]))[0, -1])

    return np.asarray(params["tok"]), next_logits


def train_mlx(train: np.ndarray, val: np.ndarray, steps: int) -> tuple[np.ndarray, Any]:
    import mlx.core as mx  # ty: ignore[unresolved-import]
    import mlx.optimizers as optim  # ty: ignore[unresolved-import]
    from mlx import nn  # ty: ignore[unresolved-import]

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok = nn.Embedding(VOCAB, DIM)
            self.pos = nn.Embedding(BLOCK, DIM)
            self.blocks = nn.TransformerEncoder(
                LAYERS, DIM, HEADS, mlp_dims=4 * DIM, norm_first=True, activation=nn.gelu
            )
            self.head = nn.Linear(DIM, VOCAB, bias=False)

        def __call__(self, ids: Any) -> Any:
            t = ids.shape[1]
            x = self.tok(ids) + self.pos(mx.arange(t))
            mask = nn.MultiHeadAttention.create_additive_causal_mask(t)
            return self.head(self.blocks(x, mask))

    model = Model()
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=LEARNING_RATE)

    def loss(model: Any, batch: Any) -> Any:
        return nn.losses.cross_entropy(model(batch[:, :-1]), batch[:, 1:], reduction="mean")

    value_and_grad = nn.value_and_grad(model, loss)
    data, held_out = mx.array(train), mx.array(val[:512])
    rng = np.random.default_rng(0)
    start = time.perf_counter()
    for i in range(1, steps + 1):
        batch = data[mx.array(rng.integers(0, len(train), BATCH))]
        value, grads = value_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, value)
        if i % 100 == 0 or i == steps:
            print(
                f"step {i:>4} | train loss {value.item():.3f} | val loss "
                f"{loss(model, held_out).item():.3f} | {time.perf_counter() - start:.1f} s"
            )

    def next_logits(ids: np.ndarray) -> np.ndarray:
        return np.asarray(model(mx.array(ids[None, -BLOCK:]))[0, -1])

    return np.asarray(model.tok.weight), next_logits


def sample(next_logits: Any, vocab: list[str], prompt: str, length: int) -> str:
    lookup = {ch: i for i, ch in enumerate(vocab)}
    ids = np.array([lookup.get(ch, VOCAB - 1) for ch in prompt], dtype=np.int32)
    rng = np.random.default_rng(0)
    for _ in range(length):
        logits = next_logits(ids).astype(np.float64) / 0.7
        p = np.exp(logits - logits.max())
        ids = np.append(ids, rng.choice(VOCAB, p=p / p.sum())).astype(np.int32)
    return "".join(vocab[i] for i in ids)


def store_embeddings(con: ducky.Connection, token_vectors: np.ndarray) -> None:
    """Embed each file as the mean of its tokens' vectors and store it in DuckDB."""
    counts = con.sql(
        "SELECT doc::INTEGER AS doc, id, count(*)::FLOAT AS n FROM tokens GROUP BY doc, id"
    ).to_numpy()
    histogram = np.zeros((int(counts["doc"].max()) + 1, VOCAB), dtype=np.float32)
    histogram[counts["doc"], counts["id"]] = counts["n"]
    vectors = (histogram / histogram.sum(axis=1, keepdims=True)) @ token_vectors
    vectors -= vectors.mean(axis=0)  # the corpus mean carries no signal
    con.execute(f"CREATE TABLE doc_vectors (doc INTEGER, vec FLOAT[{DIM}])")
    # A NumPy row binds as a LIST parameter; the cast makes it a FLOAT[d].
    con.executemany(
        f"INSERT INTO doc_vectors VALUES ($1, $2::FLOAT[{DIM}])",
        [(doc, vec) for doc, vec in enumerate(vectors)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("jax", "mlx"), default="jax")
    parser.add_argument("--files", type=int, default=120, help="largest stdlib modules to read")
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()

    con = ducky.connect()
    start = time.perf_counter()
    train, val, vocab = prepare(con, args.files)
    ((n_tokens,),) = rows(con, "SELECT count(*) FROM tokens")
    print(
        f"{args.files} files, {n_tokens:,} characters tokenized in SQL in "
        f"{time.perf_counter() - start:.1f} s: {len(train):,} train and {len(val):,} "
        f"validation windows of {BLOCK + 1} tokens"
    )

    token_vectors, next_logits = (train_jax if args.backend == "jax" else train_mlx)(
        train, val, args.steps
    )
    print("\nsample:\n" + sample(next_logits, vocab, "def ", 200) + "\n")

    store_embeddings(con, token_vectors)
    queries = rows(
        con,
        "SELECT name, vec FROM doc_vectors JOIN docs USING (doc) "
        "WHERE name IN ('argparse.py', 'tarfile.py') ORDER BY name",
    )
    for name, query in queries:
        nearest = rows(
            con,
            "SELECT name, "
            f"round(array_cosine_similarity(vec, $1::FLOAT[{DIM}])::DOUBLE, 3) AS score "
            "FROM doc_vectors JOIN docs USING (doc) WHERE name <> $2 ORDER BY score DESC LIMIT 4",
            [np.asarray(query, dtype=np.float32), name],
        )
        print(f"files nearest to {name}: {nearest}")
    con.close()


if __name__ == "__main__":
    main()
