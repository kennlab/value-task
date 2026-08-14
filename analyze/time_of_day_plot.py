import sqlite3

import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


conn = sqlite3.connect("merged.db")
df = pd.read_sql_query("SELECT * FROM data", conn)
df['dt']=pd.to_datetime(df['date'] + ' ' + df['time'])
df['td_15'] = (df['dt'].dt.floor('15min') - df['dt'].dt.floor('D')) + pd.to_datetime(0)  # Time of day rounded to nearest 15 minutes

fig, ax = plt.subplots(figsize=(10, 4))

sns.kdeplot(
    data=df.query('dt < "2026-05-19"'),
    x="td_15",
    ax=ax,
)

# Remove grid
ax.grid(False)

# Remove top and right spines
sns.despine(ax=ax)

# X-axis formatting: hour labels only
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # every 2 hours
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_xlabel("Time of day")

plt.tight_layout()
plt.show()