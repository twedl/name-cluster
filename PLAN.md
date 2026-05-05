# Plan

Goal: take a list of business names and cluster them into groups, where all the names in a group represent the same entity. The list may be in the millions. There may be other characteristics associated with each name, like country, product (or list of products). The names are those that would be recorded as the importer or exporter on customs forms in, e.g., the U.S. (form CBP 7501) or Canada (CARM or formerly form B3 import documentation).

A general example is this:

[
  (IBM USA, W), 
  (International Business Machines, X), 
  (Apple Computer Co., Y), 
  (Apple, Z),
]

with output

[
  {(IBM USA, W), 0}, 
  {(International Business Machines, X), 0}, 
  {(Apple Computer Co., Y), 1},
  {(Apple, Z), 1},
]

IBM and International Business Machines USA are the same company, and Apple Computer Co. and Apple are the same company.

The names may contain spurious characters, typos, individual names, extra information possibly related to local conditions or personnel, country-specific information. For example, Chinese companies have specific naming conventions that usually include the company's city that other countries do not follow. For example, a company with many locations in the U.S. may include a suffix or prefix to the generic name, or some may use abbreviations and others don't. For example, a company may accidentally include a meaningless prefix to their name on forms due to printing or input errors like "00 IBM", "000 IBM".

## Tools:
Python package with rust backend (possibly through existing python wrappers to existing rust crates). Published to PyPI.

## Considerations

The preference ordering of library characteristics is: 

1. UX
2. correctness and accuracy, 
3. performance, 
4. verifiability and observability.

This will eventually run on a kubernetes notebook inside a firewall with only access to PyPI through an internal artifactory mirror. The input list or table may be millions of elements with perhaps less than 100 column characteristics. There will not be a GPU available. Memory and CPU are limited.

The input and output will likely be either parquet datasets or in-memory duckdb or polars datasets read from a python script or REPL.

This package must include one or more of the following: an example generator, a dataset of company names and characteristics.
