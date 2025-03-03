
""" This script produces a table of descriptive statistics from the results
    of an experiment, after grouping by one or more columns.  It expects a CSV
    file that was collected by collect_logs.py.

    For example, you can see profit statistics by agent type, or by
    learning rate, or by multiple columns at once.  Check the CSV header row
    to see what is available in your logs.

    It also produces a distribution plot of the first three columns given to -c,
    which are assigned to the column, row, and hue of the grid of plots, and
    separately one column name to -x for the categorical x-axis of each subplot.

    For example, "-c mode date agent -x symbol" will produce a 2-D grid of plots
    by mode (across) and date (down).  Each plot will break out agent type by
    hue, and also draw separate x-axis bars for each symbol.  More than three
    columns can be given to -c for the tabular stats, but will be ignored for
    the plot.
"""
import argparse
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams.update({'font.size': 12})

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:,.2f}'.format)
pd.set_option('display.multi_sparse', False)


# Read command-line parameters.
parser = argparse.ArgumentParser(description='Flexible stats/plotting tool for logs.')

parser.add_argument('-f', '--file', required=True, metavar='CSV_FILE',
                    help='CSV file to parse (generated from collect_logs.py).')
parser.add_argument('-c', '--columns', nargs='+', default=['mode','symbol'], metavar='COLUMN_NAMES',
                    help='One or more columns on which to group.  Ex: mode, date, agent.  '
                         'Determines column, row, hue for plots.  Only first three affect plot.')
parser.add_argument('-x', '--x_var', default=None, metavar='COLUMN_NAME',
                    help='This column will be the x-axis of each plot.  The y-axis is always profit.  '
                         'Affects plot only.  Optional but usually recommended.')

args = parser.parse_args()

# Parse CSV to DataFrame.
df = pd.read_csv(args.file)

# Stringify columns that should not get float formatted.
str_cols = [ x for x in ['lr','tau'] if x in df.columns ]
for col in str_cols:  df[col] = df[col].astype(str)

# Print stats distribution grouped by requested columns, showing profit.
# Usually want to include mode to avoid mixing training and testing results.
print(df.groupby(args.columns)['profit'].describe())


# Decide which columns to assign to which plot attributes.
cols = args.columns
col, row, hue = None, None, None
match len(cols):
    case 1: col = cols[0]
    case 2: col = cols[0]; row = cols[1]
    case _: col = cols[0]; row = cols[1]; hue = cols[2]

# Draw, annotate, and save the plot.
g = sns.catplot(data=df, x=args.x_var, y="profit", col=col, row=row, hue=hue, kind="box",
                aspect=1.0, margin_titles=True, sharey=False, legend_out=False)
g.set_ylabels("Daily Profit (USD)")
g.tick_params(axis='x', rotation=90)
g.tight_layout()

out = "_".join(cols)
g.savefig(f'images/plot_{out}.pdf')

