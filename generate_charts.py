import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

FEAR_C  = '#E05252'
GREED_C = '#3BAA72'
NEU_C   = '#8888AA'
SEG_PAL = {'Consistent Winners':'#3B6FBA','Hyper-Active Traders':'#E09030','Passive / Infrequent':'#7BAD6E'}

plt.rcParams.update({
    'figure.facecolor':'white','axes.facecolor':'#F9F9FB',
    'axes.spines.top':False,'axes.spines.right':False,
    'font.family':'DejaVu Sans','font.size':11,
    'axes.titlesize':13,'axes.titleweight':'bold',
    'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,
    'grid.color':'#E0E0E8','grid.linewidth':0.6,
})

print("Loading data...")
trades  = pd.read_csv("historical_data.csv")
fg      = pd.read_csv("fear_greed_index.csv")

trades['date'] = pd.to_datetime(trades['Timestamp IST'], dayfirst=True).dt.normalize()
fg['date']     = pd.to_datetime(fg['date'])

def simplify(c):
    if 'Fear' in c: return 'Fear'
    if 'Greed' in c: return 'Greed'
    return 'Neutral'
fg['sentiment'] = fg['classification'].apply(simplify)

merged = trades.merge(fg[['date','classification','sentiment','value']], on='date', how='inner')
merged['is_win']  = merged['Closed PnL'] > 0
merged['is_long'] = merged['Direction'].str.lower().str.contains('buy|long')

print("Engineering metrics...")
daily = (merged.groupby(['Account','date','sentiment'])
         .agg(
             daily_pnl    =('Closed PnL','sum'),
             n_trades     =('Closed PnL','count'),
             win_rate     =('is_win','mean'),
             avg_size_usd =('Size USD','mean'),
             long_ratio   =('is_long','mean'),
         ).reset_index())

daily = daily.sort_values(['Account','date'])
daily['cum_pnl']     = daily.groupby('Account')['daily_pnl'].cumsum()
daily['running_max'] = daily.groupby('Account')['cum_pnl'].cummax()
daily['drawdown']    = daily['cum_pnl'] - daily['running_max']

trader = (daily.groupby('Account').agg(
    total_pnl    =('daily_pnl','sum'),
    avg_win_rate =('win_rate','mean'),
    avg_trades   =('n_trades','mean'),
    pnl_std      =('daily_pnl','std'),
    avg_size     =('avg_size_usd','mean'),
    avg_long     =('long_ratio','mean'),
    n_days       =('date','count'),
    med_drawdown =('drawdown','median'),
).reset_index())
trader['pnl_std'] = trader['pnl_std'].fillna(0)

def segment(row):
    if row['avg_win_rate'] >= 0.45: return 'Consistent Winners'
    elif row['avg_trades'] >= 80:   return 'Hyper-Active Traders'
    else:                            return 'Passive / Infrequent'
trader['segment'] = trader.apply(segment, axis=1)

fear_d  = daily[daily['sentiment']=='Fear']
greed_d = daily[daily['sentiment']=='Greed']
seg_order = ['Consistent Winners','Hyper-Active Traders','Passive / Infrequent']
sentiment_order = ['Fear','Neutral','Greed']
colors = [FEAR_C, NEU_C, GREED_C]

# CHART 1 — Performance distributions
print("Generating chart 1...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Chart 1 — Trader Performance: Fear vs Greed Days', fontsize=15, fontweight='bold', y=1.02)

for ax, col, label, cap in zip(
    axes,
    ['daily_pnl','win_rate','drawdown'],
    ['Daily PnL ($)','Win Rate','Drawdown Proxy ($)'],
    [50000, 1, 50000]
):
    data_f = fear_d[col].clip(-cap, cap)
    data_g = greed_d[col].clip(-cap, cap)
    bp = ax.boxplot([data_f, data_g], patch_artist=True, widths=0.5,
                    medianprops=dict(color='white', linewidth=2.5),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker='o', markersize=2, alpha=0.3))
    for patch, color in zip(bp['boxes'], [FEAR_C, GREED_C]):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set_xticks([1,2]); ax.set_xticklabels(['Fear','Greed'])
    ax.set_title(label); ax.yaxis.grid(True)
    _, p = stats.mannwhitneyu(data_f.dropna(), data_g.dropna())
    sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else f'p={p:.2f}'))
    ax.text(0.5, 0.96, sig, transform=ax.transAxes, ha='center', fontsize=12, color='#444')

plt.tight_layout()
plt.savefig('chart1_performance.png', dpi=150, bbox_inches='tight')
plt.close()


# CHART 2 — Behavior shifts
print("Generating chart 2...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Chart 2 — Trader Behavior Shifts Across Sentiment Regimes', fontsize=15, fontweight='bold', y=1.02)

ax = axes[0]
means = [daily[daily['sentiment']==s]['n_trades'].mean() for s in sentiment_order]
bars = ax.bar(sentiment_order, means, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
for bar, v in zip(bars, means):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{v:.1f}', ha='center', fontsize=10, fontweight='bold')
ax.set_title('Avg Trades per Day'); ax.set_ylabel('Trades'); ax.yaxis.grid(True)

ax = axes[1]
means = [daily[daily['sentiment']==s]['long_ratio'].mean() for s in sentiment_order]
bars = ax.bar(sentiment_order, means, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
ax.axhline(0.5, color='#555', linestyle='--', linewidth=1, label='50/50 line')
for bar, v in zip(bars, means):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{v:.1%}', ha='center', fontsize=10, fontweight='bold')
ax.set_title('Long Position Ratio'); ax.set_ylabel('Proportion Long'); ax.yaxis.grid(True); ax.legend(fontsize=9)

ax = axes[2]
means = [daily[daily['sentiment']==s]['avg_size_usd'].mean() for s in sentiment_order]
bars = ax.bar(sentiment_order, means, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
for bar, v in zip(bars, means):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100, f'${v:,.0f}', ha='center', fontsize=10, fontweight='bold')
ax.set_title('Avg Trade Size (USD)'); ax.set_ylabel('USD'); ax.yaxis.grid(True)

plt.tight_layout()
plt.savefig('chart2_behavior.png', dpi=150, bbox_inches='tight')
plt.close()


# CHART 3 — Segment profiles
print("Generating chart 3...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Chart 3 — Trader Segment Profiles', fontsize=15, fontweight='bold', y=1.02)
seg_colors = [SEG_PAL[s] for s in seg_order]

ax = axes[0]
medians = [trader[trader['segment']==s]['total_pnl'].median() for s in seg_order]
bars = ax.bar(seg_order, medians, color=seg_colors, alpha=0.85, edgecolor='white', linewidth=1.5)
for bar, v in zip(bars, medians):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5000, f'${v/1e3:.0f}K', ha='center', fontsize=10, fontweight='bold')
ax.set_title('Median Total PnL'); ax.set_ylabel('USD')
ax.set_xticklabels(seg_order, rotation=15, ha='right'); ax.yaxis.grid(True)

ax = axes[1]
means = [trader[trader['segment']==s]['avg_win_rate'].mean() for s in seg_order]
bars = ax.bar(seg_order, means, color=seg_colors, alpha=0.85, edgecolor='white', linewidth=1.5)
ax.axhline(0.5, color='#555', linestyle='--', linewidth=1)
for bar, v in zip(bars, means):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{v:.1%}', ha='center', fontsize=10, fontweight='bold')
ax.set_title('Avg Win Rate'); ax.set_ylabel('Win Rate')
ax.set_xticklabels(seg_order, rotation=15, ha='right'); ax.yaxis.grid(True)

ax = axes[2]
means = [trader[trader['segment']==s]['avg_trades'].mean() for s in seg_order]
bars = ax.bar(seg_order, means, color=seg_colors, alpha=0.85, edgecolor='white', linewidth=1.5)
for bar, v in zip(bars, means):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f'{v:.0f}', ha='center', fontsize=10, fontweight='bold')
ax.set_title('Avg Trades / Day'); ax.set_ylabel('Trades')
ax.set_xticklabels(seg_order, rotation=15, ha='right'); ax.yaxis.grid(True)

plt.tight_layout()
plt.savefig('chart3_segments.png', dpi=150, bbox_inches='tight')
plt.close()


# CHART 4 — Heatmap
print("Generating chart 4...")
fig, ax = plt.subplots(figsize=(9, 4))
fig.suptitle('Chart 4 — Median Daily PnL by Segment x Sentiment', fontsize=14, fontweight='bold')

daily2 = daily.merge(trader[['Account','segment']], on='Account', how='left')
pivot = daily2.groupby(['segment','sentiment'])['daily_pnl'].median().unstack(fill_value=0)
pivot = pivot.reindex(index=seg_order, columns=['Fear','Neutral','Greed'], fill_value=0)

sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn', center=0,
            linewidths=0.5, linecolor='white', ax=ax,
            cbar_kws={'label':'Median Daily PnL ($)'})
ax.set_xlabel('Sentiment'); ax.set_ylabel('Segment')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.tight_layout()
plt.savefig('chart4_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# CHART 5 — Time series
print("Generating chart 5...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={'height_ratios':[3,1]})
fig.suptitle('Chart 5 — Market-wide Daily PnL vs Fear/Greed Index Over Time', fontsize=14, fontweight='bold')

daily_agg = merged.groupby('date').agg(total_pnl=('Closed PnL','sum'), value=('value','first')).reset_index()
daily_agg = daily_agg.sort_values('date')
daily_agg['roll_pnl'] = daily_agg['total_pnl'].rolling(7).mean()

ax1.fill_between(daily_agg['date'], 0, daily_agg['roll_pnl'],
                 where=daily_agg['roll_pnl']>=0, alpha=0.4, color=GREED_C, label='Profitable days')
ax1.fill_between(daily_agg['date'], 0, daily_agg['roll_pnl'],
                 where=daily_agg['roll_pnl']<0, alpha=0.4, color=FEAR_C, label='Loss days')
ax1.plot(daily_agg['date'], daily_agg['roll_pnl'], color='#333', linewidth=1.2)
ax1.axhline(0, color='#888', linewidth=0.8, linestyle='--')
ax1.set_ylabel('7-day Rolling PnL ($)'); ax1.legend(fontsize=9); ax1.yaxis.grid(True)

ax2.plot(daily_agg['date'], daily_agg['value'], color='#6060C0', linewidth=1.2)
ax2.fill_between(daily_agg['date'], 50, daily_agg['value'],
                 where=daily_agg['value']>=50, alpha=0.3, color=GREED_C)
ax2.fill_between(daily_agg['date'], daily_agg['value'], 50,
                 where=daily_agg['value']<50, alpha=0.3, color=FEAR_C)
ax2.axhline(50, color='#888', linewidth=0.8, linestyle='--')
ax2.set_ylabel('F&G Index'); ax2.set_xlabel('Date'); ax2.yaxis.grid(True)

plt.tight_layout()
plt.savefig('chart5_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()

# CHART 6 — Segment x Sentiment
print("Generating chart 6...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Chart 6 — Segment Behavior: Fear vs Greed Days', fontsize=14, fontweight='bold')

fg_only = daily2[daily2['sentiment'].isin(['Fear','Greed'])]
x = np.arange(len(seg_order)); w = 0.35

ax = axes[0]
pivot_wr = fg_only.groupby(['segment','sentiment'])['win_rate'].mean().unstack()
pivot_wr = pivot_wr.reindex(seg_order)
ax.bar(x-w/2, pivot_wr['Fear'],  width=w, color=FEAR_C,  alpha=0.8, label='Fear',  edgecolor='white')
ax.bar(x+w/2, pivot_wr['Greed'], width=w, color=GREED_C, alpha=0.8, label='Greed', edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(seg_order, rotation=15, ha='right')
ax.set_ylabel('Avg Win Rate'); ax.set_title('Win Rate by Segment & Sentiment')
ax.legend(); ax.yaxis.grid(True)

ax = axes[1]
pivot_tr = fg_only.groupby(['segment','sentiment'])['n_trades'].mean().unstack()
pivot_tr = pivot_tr.reindex(seg_order)
ax.bar(x-w/2, pivot_tr['Fear'],  width=w, color=FEAR_C,  alpha=0.8, label='Fear',  edgecolor='white')
ax.bar(x+w/2, pivot_tr['Greed'], width=w, color=GREED_C, alpha=0.8, label='Greed', edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(seg_order, rotation=15, ha='right')
ax.set_ylabel('Avg Trades/Day'); ax.set_title('Trade Frequency by Segment & Sentiment')
ax.legend(); ax.yaxis.grid(True)

plt.tight_layout()
plt.savefig('chart6_seg_sentiment.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✅ All 6 charts generated successfully!")
print("Files created: chart1_performance.png through chart6_seg_sentiment.png")
