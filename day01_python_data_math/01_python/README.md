# 01 — Python Fundamentals for ML Engineering

> Python is the glue language of ML: master its data structures, laziness (generators), wrappers (decorators) and concurrency model, and every ML framework becomes readable.

| | |
|---|---|
| **Day** | 1 — Python + Data + Math |
| **Topic** | 01 — Python fundamentals for ML engineering |
| **Notebooks** | 3 (`01_core_language`, `02_generators_decorators_contextmanagers`, `03_stdlib_and_concurrency`) |
| **Builds on** | Nothing — the starting point |
| **Leads to** | [`02_numpy`](../02_numpy/README.md) (arrays), then everything else |

## 📖 Short forms used in this topic

| Term | Full form | What it is |
|---|---|---|
| **OOP** | Object-Oriented Programming | Organising code as classes (data + behaviour bundled together) and their instances. |
| **GIL** | Global Interpreter Lock | A lock in CPython so only one thread runs Python bytecode at a time; it is released during blocking I/O. |
| **CPU** | Central Processing Unit | The processor; "CPU-bound" work is limited by computation, not waiting. |
| **I/O** | Input / Output | Waiting on disk, network or other external resources ("I/O-bound" work). |
| **ML** | Machine Learning | Building models that learn patterns from data. |
| **JSON** | JavaScript Object Notation | A text format for nested key-value data (`json` module). |
| **CSV** | Comma-Separated Values | A plain-text table format, one row per line (`csv` module). |
| **API** | Application Programming Interface | The set of functions/classes/endpoints a library or service exposes. |
| **CUDA** | Compute Unified Device Architecture | NVIDIA's platform for running code on GPUs (mentioned as what ML frameworks wrap). |
| **GPU** | Graphics Processing Unit | Massively parallel processor used to train neural networks. |
| **regex** | Regular expression | A pattern language for matching text (`re` module). |
| **stdlib** | Standard library | Modules that ship with Python (`json`, `pathlib`, `logging`, ...). |
| **`*args` / `**kwargs`** | Positional / keyword arguments | Catch-all parameters: a tuple of extra positional args and a dict of extra keyword args. |

## 🧠 In plain words

Python is a readable language that ML libraries are written around. This topic covers the Python features ML code leans on: containers for data, short ways to transform it, classes to model it, lazy generators to stream it, wrappers (decorators and `with` blocks) to add behaviour, and tools to do many things at once.

## 🎒 Analogy to remember

Think of a **kitchen**. Collections are your containers (jar, tray, labelled spice rack). Generators are a chef cooking each dish **to order** instead of pre-making 10,000 meals. Decorators are packaging a dish passes through (timing sticker, caching box). Concurrency: threads are several cooks sharing **one stove burner** (the GIL) - they only help while someone waits for the oven (I/O); multiprocessing is opening more kitchens (CPU-bound work).
*Where it breaks down:* real cooks never "pickle" a recipe to send it to another kitchen - Python must, which is why lambdas fail in `multiprocessing`.

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
  subgraph S1["Data and functions"]
    A["Collections and comprehensions"] --> B["Functions, args/kwargs, lambda"]
    B --> C["OOP: classes, dataclasses, magic methods"]
  end
  subgraph S2["Lazy and wrapping"]
    D["Iterators and generators"] --> E["Decorators and context managers"]
  end
  subgraph S3["Robust and parallel"]
    F["Exceptions and typing"] --> G["Concurrency: asyncio, threading, multiprocessing, GIL"]
  end
  C --> D
  E --> F
  G --> H["Ready for NumPy"]
  classDef data fill:#DBEAFE,stroke:#2563EB,color:#1a1a1a;
  classDef lazy fill:#FEF3C7,stroke:#D97706,color:#1a1a1a;
  classDef safe fill:#EDE9FE,stroke:#7C3AED,color:#1a1a1a;
  classDef done fill:#DCFCE7,stroke:#16A34A,color:#1a1a1a;
  class A,B,C data;
  class D,E lazy;
  class F,G safe;
  class H done;
```

## 🔍 Concepts explained

**Category chips:** 🟦 Data structures · 🟩 Functions and OOP · 🟧 Lazy and wrapping · 🟥 Safety · 🟪 Concurrency

### 🟦 Collections

| Type | Ordered | Mutable | Duplicates | Lookup | Use when | Pitfall |
|---|---|---|---|---|---|---|
| `list` | yes | yes | yes | by index | sequence you will grow/change | `a = b = []` aliases one list |
| `tuple` | yes | no | yes | by index | fixed records, dict keys, return values | one-element needs a comma: `(1,)` |
| `set` | no | yes | no | membership test | de-duplicate, fast `in` | elements must be hashable |
| `dict` | insertion order | yes | unique keys | by key | label to value, configs | keys must be hashable |

> [!WARNING]
> Mutability is shared state. `def f(x=[])` evaluates the default **once**, so every call reuses the same list.

### 🟩 Functions and OOP

**`*args` / `**kwargs`**
- **Definition:** collect extra positional args into a tuple and extra keyword args into a dict.
- **Analogy:** a "bring anything" bag at the door.
- **Code idea:** `def forward(self, *args, **kwargs): return self.net(*args, **kwargs)`
- **Use when / watch out:** wrappers and extensible APIs; hides the real signature, so document it.

**Dataclass**
- **Definition:** `@dataclass` generates `__init__`, `__repr__`, `__eq__` from annotated fields.
- **Analogy:** a pre-printed form - you only fill in the fields.
- **Code idea:** `@dataclass class Batch: x: list; y: list`
- **Use when / watch out:** configs, records, batches. Use `field(default_factory=list)` for mutable defaults.

**Magic (dunder) methods**
- **Definition:** special methods Python calls for you (`len(obj)` calls `__len__`, `obj[i]` calls `__getitem__`, `for` calls `__iter__`).
- **Analogy:** the standard electrical socket - implement the plug shape and any appliance (the framework) works.
- **Use when / watch out:** `__len__` + `__getitem__` is the map-style PyTorch `Dataset` contract.

| Option | Relationship | Use when | Pitfall |
|---|---|---|---|
| Inheritance | "is-a" | true specialisation (`MyNet` is an `nn.Module`) | deep hierarchies get brittle |
| Composition | "has-a" | build models from parts (a net **has** layers) | slightly more wiring |

### 🟧 Lazy and wrapping

**Generator (`yield`)**
- **Definition:** a function that produces values one at a time and pauses between them.
- **Analogy:** cooking to order vs a buffet pre-made for everyone.
- **Code idea:** `def batched(it, n): ...  yield batch`
- **Use when / watch out:** data larger than memory, streaming. One-shot: it cannot be re-iterated.

**Decorator**
- **Definition:** a function that takes a function and returns a wrapped function; `@d` above `def f` means `f = d(f)`.
- **Analogy:** gift-wrapping - same gift, extra layer.
- **Code idea:** timing wrapper using `functools.wraps`.
- **Use when / watch out:** cross-cutting behaviour (timing, caching, `torch.no_grad()`); forget `functools.wraps` and you lose the name/docstring.

**Context manager (`with`)**
- **Definition:** an object with `__enter__`/`__exit__` that guarantees set-up and clean-up.
- **Analogy:** hotel key card: room opens on entry, locks on exit even if you leave in a hurry.
- **Use when / watch out:** files, locks, `torch.no_grad()`; `@contextmanager` turns a generator into one.

| Option | Memory | Re-iterable | Use when |
|---|---|---|---|
| List comprehension `[...]` | O(n) | yes | you need all items / random access |
| Generator expression `(...)` | O(1) | no | stream once, huge data |

### 🟥 Safety

- **Exceptions:** `try/except/finally` handle errors; catch specific types.
- **Typing:** hints (`list[int]`, `Optional[str]`) are not enforced at runtime; checkers like `mypy` read them.

> [!WARNING]
> A bare `except:` also swallows `KeyboardInterrupt` and real bugs. Catch the narrowest exception you can handle.

### 🟪 Concurrency - which tool?

| Tool | Parallel CPU work? | Best for | Pitfall |
|---|---|---|---|
| `threading` | No (GIL) | I/O-bound, blocking libraries | no speed-up on pure-Python CPU loops |
| `asyncio` | No (single thread) | many concurrent I/O waits (API calls) | needs `await`-compatible libraries |
| `multiprocessing` | Yes (separate processes) | CPU-bound work | start-up + pickling cost; lambdas not picklable |

```mermaid
flowchart TD
  Q{"What limits my task?"}
  Q -->|"CPU computation"| MP["multiprocessing: one process per core"]
  Q -->|"Waiting on I/O"| IO{"Library supports await?"}
  IO -->|"yes"| AS["asyncio"]
  IO -->|"no, blocking calls"| TH["threading"]
  classDef cpu fill:#FEE2E2,stroke:#DC2626,color:#1a1a1a;
  classDef io fill:#DBEAFE,stroke:#2563EB,color:#1a1a1a;
  classDef pick fill:#DCFCE7,stroke:#16A34A,color:#1a1a1a;
  classDef ask fill:#FEF3C7,stroke:#D97706,color:#1a1a1a;
  class Q,IO ask;
  class MP cpu;
  class AS,TH pick;
```

> [!TIP]
> NumPy and many C extensions release the GIL while computing, so threads can still help there. For pure-Python CPU loops, use processes.

> [!NOTE]
> The pattern throughout this topic: magic methods, generators, decorators and context managers are all "protocols" - you implement a small interface and the language or framework does the rest.

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
