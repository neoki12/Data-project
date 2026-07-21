import time, os
import pandas as pd
import numpy as np

sizes = [100, 1_000, 10_000, 100_000, 1_000_000]
df_full = pd.read_csv("synthetic_sales.csv")

results = []
for n in sizes:
df = df_full.sample(n=n, random_state=42) if n < len(df_full) else df_full

csv_path, parquet_path = f"tmp_{n}.csv", f"tmp_{n}.parquet"

t0 = time.perf_counter(); df.to_csv(csv_path, index=False); t_csv_write = time.perf_counter() - t0
t0 = time.perf_counter(); df.to_parquet(parquet_path); t_parquet_write = time.perf_counter() - t0

t0 = time.perf_counter(); pd.read_csv(csv_path); t_csv_read = time.perf_counter() - t0
t0 = time.perf_counter(); pd.read_parquet(parquet_path); t_parquet_read = time.perf_counter() - t0

results.append({
"rows": n,
"csv_size_kb": os.path.getsize(csv_path) / 1024,
"parquet_size_kb": os.path.getsize(parquet_path) / 1024,
"csv_write_s": t_csv_write, "parquet_write_s": t_parquet_write,
"csv_read_s": t_csv_read, "parquet_read_s": t_parquet_read,
})

pd.DataFrame(results)