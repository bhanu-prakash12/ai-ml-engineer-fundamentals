# 01 — Python Fundamentals for ML Engineering

## Learning objectives

By the end of this topic you can:
- Use Python's core data structures (list/tuple/set/dict) correctly, including when mutability matters.
- Write comprehensions, generators, and decorators instead of reaching for manual loops/boilerplate.
- Model data with classes, dataclasses, and magic methods the way ML codebases do (configs, batches, records).
- Reason about `asyncio` vs `threading` vs `multiprocessing` and know why the GIL makes that choice non-obvious.
- Use `pathlib`, `regex`, `logging`, and `json`/`csv` the way production ML pipelines do.

## Why it matters

Every ML framework (PyTorch `Dataset`/`DataLoader`, Hugging Face `datasets`, scikit-learn
pipelines) is Python wrapped around C/CUDA. If you're shaky on generators, decorators, or
`*args`/`**kwargs`, reading framework source code — or debugging why your training loop silently
drops the last batch — becomes much harder than it needs to be.

## Prerequisites

None — this is the starting point of the curriculum.

## Core concepts

- **Collections**: list (ordered, mutable) vs tuple (ordered, immutable) vs set (unordered, unique) vs dict (key→value). Indexing/slicing (`a[1:3]`, `a[::-1]`) and the mutability trap (`a = b = []`).
- **Comprehensions**: `[x**2 for x in xs if x > 0]` — the idiomatic replacement for build-and-append loops; same syntax for sets/dicts.
- **Functions**: positional/keyword args, `*args`/`**kwargs` for variadic signatures (every `nn.Module.forward(*args, **kwargs)` relies on this), `lambda` for throwaway callables, `map`/`filter`/`zip`/`enumerate` for functional-style iteration.
- **OOP**: classes, `__init__`, inheritance vs composition ("has-a" vs "is-a" — prefer composition for ML model components), `@dataclass` for data-holding classes without boilerplate, magic methods (`__repr__`, `__len__`, `__getitem__`, `__iter__`) — these are *exactly* what a PyTorch `Dataset` needs.
- **Iterators & generators**: the iterator protocol (`__iter__`/`__next__`), `yield` for lazy, memory-efficient sequences — the same idea behind streaming datasets.
- **Decorators**: functions that wrap functions (`@property`, `@staticmethod`, custom timing/caching decorators) — the same mechanism behind `@torch.no_grad()`.
- **Context managers**: `with` blocks and `__enter__`/`__exit__` — the same mechanism behind `with torch.no_grad():`.
- **Exceptions**: `try`/`except`/`finally`, custom exception classes, when to catch vs let propagate.
- **Typing**: type hints (`list[int]`, `Optional[str]`, `Callable`) for self-documenting, tool-checkable code.
- **Concurrency**: the GIL means CPU-bound Python threads don't parallelize — use `multiprocessing` for CPU-bound work (e.g. parallel data preprocessing) and `asyncio`/`threading` for I/O-bound work (e.g. concurrent API calls).

```mermaid
flowchart LR
  A[Collections & comprehensions] --> B[Functions, *args/**kwargs, lambda]
  B --> C[OOP: classes, dataclasses, magic methods]
  C --> D[Iterators & generators]
  D --> E[Decorators & context managers]
  E --> F[Exceptions & typing]
  F --> G[Concurrency: asyncio / threading / multiprocessing + GIL]
  G --> H[Ready for NumPy: vectorized data structures]
```

## Important APIs / functions

`list/tuple/set/dict`, comprehensions, `def`, `lambda`, `*args`/`**kwargs`, `map`/`filter`/`zip`/`enumerate`,
`class`, `@dataclass`, `__repr__`/`__len__`/`__getitem__`/`__iter__`, `yield`, `@property`,
`contextlib.contextmanager`, `try/except/finally`, `typing`, `json.load`/`dump`, `csv.reader/writer`,
`pathlib.Path`, `re.match/search/findall`, `logging.getLogger`, `asyncio.run/gather`,
`threading.Thread`, `multiprocessing.Pool`.

## Notebooks

| Notebook | Purpose | Expected output |
|---|---|---|
| `01_core_language.ipynb` | Collections, comprehensions, functions, OOP, dataclasses, magic methods | Printed demonstrations of each concept; a custom `Batch` dataclass and a custom iterable class with working `__len__`/`__getitem__`/`__iter__` |
| `02_generators_decorators_contextmanagers.ipynb` | Iterators/generators/`yield`, decorators, context managers, exceptions, typing | A lazy batch generator that never materializes the full dataset in memory; a timing decorator; a custom context manager; typed functions passing `mypy`-style hints |
| `03_stdlib_and_concurrency.ipynb` | `json`/`csv`/`pathlib`/`regex`/`logging`, `asyncio` vs `threading` vs `multiprocessing`, the GIL | A benchmark showing CPU-bound `multiprocessing` beats `threading` due to the GIL, and an `asyncio` I/O-bound demo beating sequential calls |

## Key takeaways

- Comprehensions and generators are not just "shorter loops" — generators give you O(1) memory for arbitrarily large sequences, which is why streaming datasets use them.
- `*args`/`**kwargs` is how every flexible ML API (optimizers, `forward()`, callbacks) stays extensible without breaking signatures.
- The GIL means threads don't help CPU-bound Python work — know when to reach for `multiprocessing` instead.
- Magic methods are the contract PyTorch `Dataset` and many other framework base classes expect you to implement.

## Common mistakes

- Using a mutable default argument (`def f(x=[])`) — the same list is reused across calls.
- Confusing `is` (identity) with `==` (equality).
- Writing CPU-bound parallel code with `threading` and being confused why it isn't faster (the GIL).
- Materializing a huge list with a list comprehension when a generator expression would do.
- Catching bare `except:` and silently swallowing real bugs.

## Interview questions

**Conceptual**
1. What is the GIL, and why doesn't `threading` speed up CPU-bound Python code?
2. What's the difference between a generator and a list, and when would memory usage force you to prefer one?
3. Why does composition tend to be preferred over inheritance in ML model code?
4. What problem do `*args`/`**kwargs` solve in a function signature?
5. What's the difference between a context manager and a `try/finally` block — and why use one over the other?

**Coding**
1. Implement a class that supports `len()` and index access (`obj[i]`) by implementing the relevant magic methods.
2. Write a generator function that lazily yields batches of size `n` from an iterable, without materializing the whole thing.
3. Write a decorator `@timed` that prints how long a function took to run.

**Debugging**
1. A function `def add_item(item, bucket=[])` keeps accumulating items across unrelated calls — why, and how do you fix it?
2. A `multiprocessing.Pool` worker silently fails with a `PicklingError` when passed a lambda — why, and what's the fix?

**Real-world engineering**
1. You need to preprocess 10M rows of text on 8 CPU cores. Would you use `threading` or `multiprocessing`, and why?
2. You're writing a client that calls 100 external APIs concurrently. Would you use `asyncio`, `threading`, or `multiprocessing`, and why?

## Exercises

1. Build a `@dataclass` called `TrainingExample` with fields `text: str`, `label: int`, and a method `__len__` returning `len(self.text)`.
2. Write a generator `batched(iterable, batch_size)` and prove it doesn't load the full input into memory by running it over `range(10**7)`.
3. Write a custom context manager (class-based or `@contextmanager`) that logs "entering"/"exiting" a block.
4. Benchmark squaring 2,000,000 numbers using a single process vs a `multiprocessing.Pool` with 4 workers, and explain the speedup (or lack of one).

## References

- [Python docs — Data structures](https://docs.python.org/3/tutorial/datastructures.html)
- [Python docs — Classes](https://docs.python.org/3/tutorial/classes.html)
- [Python docs — `dataclasses`](https://docs.python.org/3/library/dataclasses.html)
- [Python docs — Generators](https://docs.python.org/3/howto/functional.html#generators)
- [Python docs — `asyncio`](https://docs.python.org/3/library/asyncio.html)
- [Python docs — GIL](https://docs.python.org/3/glossary.html#term-global-interpreter-lock)
- [Real Python — Decorators](https://realpython.com/primer-on-python-decorators/)
