# encoding: utf-8
"""
This script uses the "cloc" program (which should be installed on your system
before running this script) to generate a figure containing a breakdown of all
the scripts in this folder (code, whitespace, comments).

Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
import subprocess
import pandas
from matplotlib import pyplot as plt
from matplotlib import font_manager
import numpy as np

csv = subprocess.Popen(
    'cloc --by-file --csv ??_*.py figure_*.py dodo.py config.py',
    shell=True,
    stdout=subprocess.PIPE
).stdout
loc = pandas.read_csv(csv, skiprows=4, usecols=[1, 2, 3, 4], index_col=0)
time = pandas.read_csv('time.csv', index_col=0)
loc = loc.join(time)
loc = pandas.concat((loc[:2], loc[2:].sort_index(ascending=False)))

# Coding font
ticks_font = font_manager.FontProperties(family='consolas', size=10)

# Make stack bar graph of the lines of code
plt.figure(figsize=(7, 4))
plt.barh(np.arange(len(loc)), loc['code'])
plt.barh(np.arange(len(loc)), loc['comment'], left=loc['code'])
plt.barh(np.arange(len(loc)), loc['blank'], left=loc['code'] + loc['comment'])
plt.xlabel('Number of lines')
plt.legend(['code', 'comment', 'blank'], loc='upper right')

# Annotate the bars with the running time of each script

# Put the names of the scripts on the y-axis
plt.yticks(np.arange(len(loc)), loc.index.values)
ax = plt.gca()
for label in ax.get_yticklabels():
    label.set_fontproperties(ticks_font)
ax.tick_params(left=False)


# Some fine tuning of the plot
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.ylim(-0.6, len(loc) - 0.5)
plt.tight_layout()

# Save the PDF version
plt.savefig('../paper/figures/code_summary.pdf')

# Print mean loc (excluding dodo.py and config.py)
selection = loc[(loc.index != 'dodo.py') & (loc.index != 'config.py')]['code']
print('Mean loc: %f (std. %f)' % (selection.mean(), selection.std()))
