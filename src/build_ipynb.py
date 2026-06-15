"""Convert the Databricks source notebook (.py with `# COMMAND ----------`)
into a Jupyter .ipynb so it renders nicely in the workspace preview AND can
be imported into Databricks Community Edition as-is.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.abspath(os.path.join(HERE, "..", "notebooks",
                                    "databricks_crypto_pipeline.py"))
DST  = os.path.abspath(os.path.join(HERE, "..", "notebooks",
                                    "databricks_crypto_pipeline.ipynb"))

with open(SRC) as f:
    text = f.read()

# Strip the leading "# Databricks notebook source" marker
text = re.sub(r"^# Databricks notebook source\s*\n", "", text)

# Split on COMMAND separators
chunks = [c.strip("\n") for c in re.split(r"\n# COMMAND -+\n", text) if c.strip()]

cells = []
md_prefix = re.compile(r"^# MAGIC %md\s*\n?", re.MULTILINE)
sql_prefix = re.compile(r"^# MAGIC %sql\s*\n?", re.MULTILINE)
magic_line = re.compile(r"^# MAGIC ?", re.MULTILINE)

for chunk in chunks:
    if chunk.lstrip().startswith("# MAGIC %md"):
        body = md_prefix.sub("", chunk)
        body = magic_line.sub("", body)
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": body.splitlines(keepends=True),
        })
    elif chunk.lstrip().startswith("# MAGIC %sql"):
        body = sql_prefix.sub("", chunk)
        body = magic_line.sub("", body)
        # Render as a code cell with %%sql-style header for clarity
        body = "%%sql\n" + body
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": body.splitlines(keepends=True),
            "outputs": [],
            "execution_count": None,
        })
    else:
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": chunk.splitlines(keepends=True),
            "outputs": [],
            "execution_count": None,
        })

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "databricks": {"notebookName": "databricks_crypto_pipeline"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open(DST, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", DST, "with", len(cells), "cells")
