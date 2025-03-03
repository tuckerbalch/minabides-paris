import os, sys

""" This script collects all of the individual logs from a large experimental
    batch into a single CSV data file (in ./csvs) for plotting or analysis.
    It then moves the output and log files into named subdirectories to tidy up. """

if len(sys.argv) < 3:
    print ("Usage: python collect_logs.py <output_name> <log_files...>")
    exit()

csv = sys.argv[1]
files = sys.argv[2:]
first = True

logdir = f"log/{csv}"
outdir = f"output/{csv}"

os.system(f"mkdir {logdir}")
os.system(f"mkdir {outdir}")

for file in files:
    if first:
        os.system(f"head -n1 {file} > csvs/{csv}.csv")
        first = False

    os.system(f'cat {file} | grep -v "exp,exp_help" >> csvs/{csv}.csv')

    outfile = file.replace("log/", "output/")

    print("Complete:", file)
    os.system(f"mv {file} {logdir}")
    os.system(f"mv {outfile} {outdir}")

