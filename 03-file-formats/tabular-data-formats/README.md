# Tabular Data Formats on LUMI

Every training job on a shared HPC system does two things at once: it computes on the
GPU, and it retrieves the data to train on. If data can't arrive fast enough, the GPU 
sits idle; expensive hardware, allocated at real cost, doing nothing. For tabular data 
specifically (spreadsheet-style rows and columns, tabular ML features), format choice 
determines whether that data arrives fast or becomes the bottleneck.

Throughout this chapter, not every format retrieves data the same way, and this is 
important for interpreting the results below. CSV, in this chapter's benchmarks, is fully 
loaded into memory once, and after that, every further read comes from memory, not disk. 
HDF5 goes back to the real file on disk for every single request. Parquet sits in between, 
using a "memory map" that connects to the file without immediately loading it, then pulls 
in only the bytes actually requested. This difference is itself part of what's being measured, 
and is explained in full further down, in "What load time and dataloader time actually measure."

## What's actually being tested

Not just "which format is fastest", but also *how* the data will be read. This chapter 
tests two different access patterns, since they can favor completely different formats:

1. **Scanning and filtering** reading a full table, reading only specific columns, or
   reading only rows matching a condition. This is the pattern behind data exploration,
   feature engineering, and analytical queries.
2. **Random row-batch access** repeatedly grabbing small, randomly-ordered batches of
   rows, the way a model's DataLoader pulls training examples.

Testing only one of these patterns and generalizing to "the best format" is misleading. 
A format can be excellent at one and dramatically bad at the other.

## Dataset used

NYC Yellow Taxi trip records (one month, ~2.96 million rows, 19 columns), downloaded from the 
[NYC Taxi and Limousine Commission's public trip record data page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

## Formats compared

- **Parquet** a columnar, binary format where values are stored column-by-column rather 
than row-by-row, which is what lets it skip whole columns or use per-chunk summary statistics 
to skip irrelevant data entirely during a read.
- **CSV** Comma Separated Values, a plain text format, one row per line where values are 
separated by commas. It has no internal structure, and no compression. It was included as the 
no-optimization baseline.
- **HDF5** Hierarchical Data Format is a high-performance binary format, used for 
large structured scientific and numerical data, used here in "table" mode with per-column 
indexing enabled so it can support filtered and column-subset reads.

## How to run this

- Replace the account `project_xxxxxxxxx` in `Makefile` with your project id.
- Run `make download` to download the raw taxi data.
- Run `make venv` to install packages not included in the base container.
- Run `make convert` to build the same source data as Parquet, CSV, and HDF5.
- Run `make bench-scan` and `make bench-dataloader` to test the two access patterns
  described above. Run each of these multiple times to get a better estimate.
- Once you've run `make bench-scan` and `make bench-dataloader` multiple times each,
  run `make post` to compile the results. It prints two tables, one for scanning
  and filtering, one for random row-batch access each showing the average time,
  standard deviation, and number of runs, broken down by format..

## Results

| Format  | Size     |
|---------|----------|
| Parquet | 58.0 MB  |
| CSV     | 296.2 MB |
| HDF5    | 453.6 MB |

### Scanning and filtering

| Format  | Full scan      | Column subset (2 of 19) | Filtered      |
|---------|----------------|-------------------------|---------------|
| Parquet | 0.329 ± 0.102  | 0.021 ± 0.003           | 0.124 ± 0.005 |
| CSV     | 4.160 ± 0.287  | 1.529 ± 0.091           | 3.651 ± 0.233 |
| HDF5    | 5.510 ± 3.959* | 1.184 ± 0.056           | 1.172 ± 0.057 |

*\*HDF5's full-scan time decreased with each successive run (8.963s, then 6.377s, then
1.189s) explaining the large standard deviation.*

### Random row batch access (3 runs, seconds)

| Format  | Load time     | Dataloader time  |
|---------|---------------|------------------|
| Parquet | 1.635 ± 0.343 | 0.993 ± 0.052    |
| CSV     | 4.291 ± 0.099 | 0.688 ± 0.003    |
| HDF5    | 3.286 ± 0.280 | 220.007 ± 29.167 |

## What load time and dataloader time measure

These three formats don't do the same kind of work behind the scenes, even though the
benchmark asks the same question of all three. Each one splits the total cost between
load and dataloader differently

- **CSV** reads the entire file into memory during "load time." Its "dataloader
  time" afterward involves no disk access at all, it's just looking up already-loaded
  data in memory. As such, the Load time (4.291s) is the real cost, reading the entire file,
  parsing every row, and building the entire table in memory. This is the slowest load
  time of the three as it does the work upfront.
  Dataloader time (0,688s), at this point, the whole table is in memory and 10,000 random
  rows are grabbed with no disk involved. The real cost happened during load
- **HDF5** goes back to the real file on disk for every single random request. 
  "Dataloader time" here reflects real, repeated disk I/O.
  load time (3.286s) only opens the file, checks how many rows exist but doesnt touch actual
  row data at all.
  Dataloader time (220.007s), since nothing was preloaded, every random batches triggers
  a new querry back to the actual file on disk and this is why is it slower than the rest.
  The real cost is not paid once but with evry single batch.
- **Parquet** uses a "memory map," a middle-ground technique that connects to the file
  without fully loading it, then pulls in only the bytes actually requested, on demand
  spliting the cost evenly.
  Load time (1.635s) doesnt not fully read the file. it sets up a memory map, a lightweght
  connection to the file that doesnot pull in the actual data, but a fast way to reach it later.
  Dataloader (0.993s) since the real data wasnt pulled in during "load time", each random batch
  fetches its own rows through that memory-mapped connection.

## Decision framework

Given the results above, here is how to choose a format based on how you plan to read your data:

- If you mostly scan full tables, read specific columns, or filter rows (data
  exploration, feature engineering, analytical queries): use Parquet. It was
  fastest roughly 13x faster than CSV for a full scan, and over 70x faster than either 
  alternative for reading just a couple of columns.
- If you mostly grab small, randomly-ordered batches of rows (feeding a model
  during training): use CSV or Parquet, avoid HDF5. HDF5 was measured at
  roughly 300x slower than CSV for this specific access pattern, despite performing
  reasonably well at scanning. A format's scanning performance does not reliably
  predict its random-access performance, as this result demonstrates directly.
- If your table is of a very different size, column count, or shape than the ~2.96
  million row, 19-column table tested here, treat these exact numbers as
  directional, not definitive, re-running `make bench-scan` and
  `make bench-dataloader` against your own data is the most reliable way to confirm
  which format is actually justified for your case.

## Going Deeper: Does Parquet's row Group size matter?

In the random-access experiments above, the row group size was determined automatically 
by Parquet's default configuration and was not explicitly tested. As a follow-up, we 
conducted a separate analysis to evaluate the effect of row group size on performance.

### What is a row group

A Parquet file isn't stored as one giant, solid block. It's split internally into
chunks called row groups, think of it as a large stack of paper cut into
several smaller, separate stacks. Each stack also carries its own small summary
of what's inside it (for every column, the minimum and maximum value present in
that stack). That summary is exactly what lets Parquet skip a whole stack
instantly if it can't possibly contain what you're filtering for no need to
open it at all.

**Row group size** simply means: how many rows go into each stack before Parquet
start a new one. For smaller row groups, they create more metadata and more 
independently compressed blocks, which may increase file size but can also allow 
Parquet to skip irrelevant data more precisely during filtered reads. Larger row 
groups reduce metadata overhead and can improve compression, but may require more 
data to be read when only a small portion of the dataset is needed.

The goal of this experiment is therefore to measure the tradeoff between these
effects and answer two practical questions:

1. Which row-group configuration provides the fastest read performance for random 
row-batch access?
2. Which configuration uses the least storage space, and how much storage overhead 
is introduced when many small row groups are created?

We generated Parquet files with three different row-group configurations and compared 
both their on-disk size and their read performance.

- Small, force many small stacks (1,000 rows each)
- Default, let Parquet choose its own stack size automatically
- Large, force fewer, bigger stacks (500,000 rows each)

### What to expect: the tradeoff between these three

There are two separate costs that can move in different directions depending on
stack size:

- **Storage cost** More stacks means more repeated per-column summaries stored
  throughout the file, plus slightly less efficient compression (each stack is
  compressed on its own, separately from the others). More stacks generally
  means a little more disk space used, even for the exact same data.
- **Compute cost** This depends entirely on *how* you are reading the file. 
  Reading in order (scanning) behaves differently than reading in a random, 
  scattered order. Smaller stacks can help one of these patterns and hurt the 
  other, since there's a real cost to managing many separate stacks versus a real
  cost to opening one very large stack just to get a few rows out of it.

For this dataset which has 2.96-million-row taxi data, once the row groups are applied
This is what it to expect:

| Setting | Row groups created | Rows per group (approx.) | File size |
|---------|--------------------|--------------------------|-----------|
| small   | 2,965              | ~999                     | 89.9 MB   |
| default | 3                  | ~988,000                 | 58.0 MB   |
| large   | 6                  | ~494,000                 | 58.2 MB   |

Notice that "default" and "large" both land in almost the same territory,
while "small" is genuinely different, nearly a thousand times more stacks. 
That's worth keeping in mind before you guess at the result: two of these 
three settings are more similar to each other than you might expect from 
their names alone.

### Why this connects back to our two original tests

- **Scanning and filtering** reading the whole table, or just specific columns,
  or just rows matching a condition. This is the pattern that benefits
  from row groups' per-stack summaries where more stacks can mean more 
  opportunities to skip work entirely, but also more individual stacks to
  manage and check.
- **Random row-batch access** Here, stack size interacts with randomness 
  differently where grabbing one random row from a huge stack means touching a lot 
  of data just to get a little; grabbing it from a tiny stack means touching very 
  little, but you now have more stacks to organize and jump between.

### Why your own results might look completely different from ours

Its important to note that your results will depends heavily on specifics of your own data 

- How many rows your table has, In this experiment, our "small" setting created
 ~3,000 stacks specifically because we have ~3 million rows. A much smaller table 
  might barely be affected by any of these three settings at all.
- How many columns you have, and how wide each row is, the per-stack
  summary cost scales with column count, so a table with far fewer or far more
  columns than the 19 in this dataset will see a different balance of costs.
- How you actually plan to read the data, if you only ever scan the whole
  table in order and never grab random batches, this whole row-group-size
  question may matter far less to you than it does for a training pipeline.

### Try it yourself

```bash
make convert-rowgroups
make bench-rowgroups
```

`convert-rowgroups` builds all three versions from the same source data.
`bench-rowgroups` runs the same random-batch-access test against all three,
you can see which setting actually wins and by how much.

**Note:** this experiment only covers Parquet.

## Scope and limitations

This is a single-node, single-user benchmark on LUMI's shared Lustre filesystem at a
specific point in time. Results will vary with concurrent cluster load and are not a
substitute for multi-node or distributed I/O testing. See "Decision framework" above 
for guidance on how to weigh these results depending on how you plan to read your own 
data, and how your table compares in size and shape to the one tested here.

### Table of contents
- [Home](..#readme)
- [01. QuickStart](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/01-quickstart#readme)
- [02. Setting up your own environment](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/02-setting-up-environment#readme)
- [03. File formats for training data](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/03-file-formats#readme)
- [04. Data Storage Options](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/04-data-storage#readme)
- [05. Multi-GPU and Multi-Node Training](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/05-multi-gpu-and-node#readme)
- [06. Monitoring and Profiling jobs](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/06-monitoring-and-profiling#readme)
- [07. TensorBoard visualization](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/07-TensorBoard-visualization#readme)
- [08. MLflow visualization](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/08-MLflow-visualization#readme)
- [09. Wandb visualization](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/09-Wandb-visualization#readme)
- [10. LLM Inference](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/10-LLM-inference#readme)
