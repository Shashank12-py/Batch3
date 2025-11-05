# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(layout="wide", page_title="Olympic Dataset Analyzer")

st.title("Olympic Dataset Analysis — Streamlit App")
st.markdown(
    """
Upload your Olympic dataset (TSV or CSV). The app will try to infer columns,
but you can map them manually if needed.
"""
)

@st.cache_data
def load_default_df(path="olympic_data.csv"):
    # try tab separated first, then comma separated
    try:
        df = pd.read_csv(path, sep="\t", encoding="utf-8")
    except Exception:
        df = pd.read_csv(path, encoding="utf-8")
    return df

uploaded_file = st.file_uploader("Upload TSV/CSV file", type=["csv", "tsv", "txt"])
if uploaded_file is None:
    st.info("No file uploaded — attempting to read `olympic_data.csv` from working directory.")
    try:
        df = load_default_df()
        st.success("Loaded olympic_data.csv")
    except Exception as e:
        st.error(f"Could not load default file: {e}")
        st.stop()
else:
    # try to detect delimiter
    try:
        df = pd.read_csv(uploaded_file, sep="\t", encoding="utf-8")
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="utf-8")

st.subheader("Raw preview")
st.dataframe(df.head(10))

# --- Column mapping UI ---
st.sidebar.header("Column mapping (map your CSV columns to expected fields)")
cols = list(df.columns)
def col_select(label, default=None):
    return st.sidebar.selectbox(label, options=["<none>"] + cols, index=(1 + cols.index(default)) if default in cols else 0)

# Default guesses (based on typical names)
guess = { 'name': None, 'age': None, 'country': None, 'year': None, 'date': None,
         'sport': None, 'gold': None, 'silver': None, 'bronze': None, 'total': None }

# try to auto-guess from column names
lowercols = [c.lower() for c in cols]
for c in cols:
    lc = c.lower()
    if 'name' in lc and guess['name'] is None: guess['name'] = c
    if lc=='age' and guess['age'] is None: guess['age'] = c
    if 'country' in lc and guess['country'] is None: guess['country'] = c
    if 'year' in lc and guess['year'] is None: guess['year'] = c
    if 'date' in lc and guess['date'] is None: guess['date'] = c
    if 'sport' in lc or 'sports' in lc: guess['sport'] = c
    # medal columns
    if 'gold' in lc and guess['gold'] is None: guess['gold'] = c
    if 'silver' in lc and guess['silver'] is None: guess['silver'] = c
    if 'bronze' in lc and guess['bronze'] is None: guess['bronze'] = c
    if 'total' in lc and guess['total'] is None: guess['total'] = c
    # some datasets put medal counts in numeric order - try to catch them heuristically:
    if lc in ('g','golds') and guess['gold'] is None: guess['gold']=c

st.sidebar.write("Detected columns (auto-guess). Change if incorrect.")
mapped = {}
mapped['name'] = col_select("Athlete name", guess['name'])
mapped['age'] = col_select("Age (optional)", guess['age'])
mapped['country'] = col_select("Country", guess['country'])
mapped['year'] = col_select("Year", guess['year'])
mapped['date'] = col_select("Date (optional)", guess['date'])
mapped['sport'] = col_select("Sport", guess['sport'])
mapped['gold'] = col_select("Gold count", guess['gold'])
mapped['silver'] = col_select("Silver count", guess['silver'])
mapped['bronze'] = col_select("Bronze count", guess['bronze'])
mapped['total'] = col_select("Total count (optional)", guess['total'])

# Build a cleaned dataframe with standard column names
def build_clean_df(df, mapping):
    df2 = df.copy()
    rename_map = {}
    for std, col in mapping.items():
        if col and col != "<none>":
            rename_map[col] = std
    df2 = df2.rename(columns=rename_map)
    # ensure required columns exist
    required = ['name','country','year','sport']
    for r in required:
        if r not in df2.columns:
            st.error(f"Required column '{r}' not mapped / missing. Please map it in the sidebar.")
            st.stop()
    # numeric conversions for medals and year, age
    for m in ['gold','silver','bronze','total','age','year']:
        if m in df2.columns:
            # remove stray characters, coerce to numeric
            df2[m] = pd.to_numeric(df2[m].astype(str).str.replace(r'[^\d\.-]', '', regex=True), errors='coerce').fillna(0).astype(float)
    # If total missing, compute
    if 'total' not in df2.columns or df2['total'].sum() == 0:
        if all(c in df2.columns for c in ['gold','silver','bronze']):
            df2['total'] = df2[['gold','silver','bronze']].sum(axis=1)
        else:
            df2['total'] = 0
    # Standardize sport and country strings
    df2['sport'] = df2['sport'].astype(str).str.strip()
    df2['country'] = df2['country'].astype(str).str.strip()
    df2['name'] = df2['name'].astype(str).str.strip()
    # ensure year integer
    try:
        df2['year'] = df2['year'].astype(int)
    except Exception:
        # try to extract 4-digit year
        df2['year'] = df2['year'].astype(str).str.extract(r'(\d{4})').astype(float).fillna(0).astype(int)
    return df2

df_clean = build_clean_df(df, mapped)
st.subheader("Cleaned data preview")
st.dataframe(df_clean.head(10))

# --- Analysis helpers ---
def athlete_total_medals(df):
    return df.groupby('name', as_index=False)[['gold','silver','bronze','total']].sum().sort_values('total', ascending=False)

def top_gold_athlete(df):
    agg = df.groupby('name', as_index=False)['gold'].sum()
    top = agg.loc[agg['gold'].idxmax()]
    return top

def country_total_medals(df):
    return df.groupby('country', as_index=False)[['gold','silver','bronze','total']].sum().sort_values('total', ascending=False)

def total_medal_count_per_year(df):
    return df.groupby('year', as_index=False)[['gold','silver','bronze','total']].sum().sort_values('year')

def medals_by_country_sport(df):
    pivot = df.pivot_table(index=['country','sport'], values=['gold','silver','bronze','total'], aggfunc='sum').reset_index()
    return pivot

def sport_total_medals(df):
    return df.groupby('sport', as_index=False)[['gold','silver','bronze','total']].sum().sort_values('total', ascending=False)

def athletes_multi_years(df, min_years=2):
    counts = df.groupby('name')['year'].nunique().reset_index(name='distinct_years')
    names = counts[counts['distinct_years']>=min_years]['name'].tolist()
    return df[df['name'].isin(names)].groupby('name', as_index=False)[['gold','silver','bronze','total']].sum()

def medal_percentages(df):
    totals = df[['gold','silver','bronze']].sum()
    overall = totals.sum()
    pct = (totals / overall * 100).round(2)
    return pd.DataFrame({'medal': ['gold','silver','bronze'], 'count': totals.values, 'percent': pct.values})

def avg_age(df):
    if 'age' in df.columns and df['age'].replace(0, np.nan).notna().sum()>0:
        return df.loc[df['age']>0,'age'].mean()
    return None

def medals_by_country_year(df, country):
    return df[df['country']==country].groupby('year', as_index=False)[['gold','silver','bronze','total']].sum().sort_values('year')

def athletes_same_as(df, athlete_name):
    athlete_total = athlete_total_medals(df)
    target_row = athlete_total[athlete_total['name']==athlete_name]
    if target_row.empty:
        return pd.DataFrame()
    target_total = float(target_row['total'].iloc[0])
    return athlete_total[athlete_total['total']==target_total]

def medal_to_athlete_ratio(df):
    # number of unique athletes per country vs total medals
    athletes_per_country = df.groupby('country')['name'].nunique().reset_index(name='num_athletes')
    medals_per_country = df.groupby('country', as_index=False)[['gold','silver','bronze','total']].sum()
    merged = medals_per_country.merge(athletes_per_country, on='country')
    merged['medals_per_athlete'] = merged['total'] / merged['num_athletes']
    return merged.sort_values('total', ascending=False)

# --- Sidebar filters ---
st.sidebar.header("Filters")
selected_years = st.sidebar.multiselect("Select years", options=sorted(df_clean['year'].unique()), default=sorted(df_clean['year'].unique()))
selected_countries = st.sidebar.multiselect("Select countries (empty = all)", options=sorted(df_clean['country'].unique()), default=[])
selected_sports = st.sidebar.multiselect("Select sports (empty = all)", options=sorted(df_clean['sport'].unique()), default=[])

df_filtered = df_clean.copy()
if selected_years:
    df_filtered = df_filtered[df_filtered['year'].isin(selected_years)]
if selected_countries:
    df_filtered = df_filtered[df_filtered['country'].isin(selected_countries)]
if selected_sports:
    df_filtered = df_filtered[df_filtered['sport'].isin(selected_sports)]

# --- Display analyses ---
st.header("Key analyses & visualizations")

# 1. How many total medals (Gold + Silver + Bronze) did each athlete win?
with st.expander("1) Total medals per athlete"):
    at_tot = athlete_total_medals(df_filtered)
    st.dataframe(at_tot.head(50))
    st.markdown("Top performers:")
    st.table(at_tot.head(10).loc[:, ['name','gold','silver','bronze','total']])

# 2. Which athlete won the highest number of gold medals?
with st.expander("2) Athlete with highest number of gold medals"):
    topg = top_gold_athlete(df_filtered)
    st.write(f"Top gold medalist: **{topg['name']}** with **{int(topg['gold'])}** gold medals.")

# 3. Which country has won the most total medals?
with st.expander("3) Country with most total medals"):
    ctot = country_total_medals(df_filtered)
    st.dataframe(ctot.head(20))
    if not ctot.empty:
        st.write(f"Top country: **{ctot.iloc[0]['country']}** with **{int(ctot.iloc[0]['total'])}** total medals.")

# 4. Total medal count per year
with st.expander("4) Total medal count per year"):
    per_year = total_medal_count_per_year(df_filtered)
    st.dataframe(per_year)
    fig = px.bar(per_year, x='year', y='total', title='Total medals per year')
    st.plotly_chart(fig, use_container_width=True)

# 5. How many medals has each country won in each sport?
with st.expander("5) Medals per country per sport"):
    pivot_cs = medals_by_country_sport(df_filtered)
    st.dataframe(pivot_cs.head(200))

# 6. Which sport contributes the most medals overall?
with st.expander("6) Sport with most medals"):
    sport_tot = sport_total_medals(df_filtered)
    st.dataframe(sport_tot)
    if not sport_tot.empty:
        st.write(f"Top sport by total medals: **{sport_tot.iloc[0]['sport']}** ({int(sport_tot.iloc[0]['total'])} medals)")

# 7. Which athlete has shown consistent performance across multiple Olympics?
with st.expander("7) Athletes with consistent performance across multiple Olympics"):
    consistent = athletes_multi_years(df_clean, min_years=2)
    st.write("Athletes who won medals in multiple distinct years (>=2):")
    st.dataframe(consistent.sort_values('total', ascending=False).head(50))

# 8. Total number of medals won by each country over all years
with st.expander("8) Total medals per country (all years)"):
    st.dataframe(country_total_medals(df_clean))

# 9. Percentage of total medals that were gold, silver, bronze
with st.expander("9) Medal type percentages"):
    pct = medal_percentages(df_clean)
    st.dataframe(pct)
    fig2 = px.pie(pct, values='count', names='medal', title='Medal type distribution')
    st.plotly_chart(fig2, use_container_width=True)

# 10. Average age of medal winners
with st.expander("10) Average age of medal winners"):
    avg = avg_age(df_clean)
    if avg is not None:
        st.write(f"Average age (over rows with age): **{avg:.2f}** years")
    else:
        st.write("Age column not available or not usable.")

# 11. Relationship between athlete age and number of medals
with st.expander("11) Relationship between age and medals"):
    if 'age' in df_clean.columns:
        agg_age = df_clean.groupby('name', as_index=False).agg({'age':'mean','total':'sum'})
        st.dataframe(agg_age.sort_values('total', ascending=False).head(30))
        fig3 = px.scatter(agg_age, x='age', y='total', hover_name='name', title='Average age vs total medals per athlete')
        st.plotly_chart(fig3, use_container_width=True)
        corr = agg_age['age'].corr(agg_age['total'])
        st.write(f"Pearson correlation (age vs total medals): {corr:.3f}")
    else:
        st.write("No age data available to analyze correlation.")

# 12. Which year had the highest number of gold medals awarded?
with st.expander("12) Year with most gold medals"):
    year_gold = df_clean.groupby('year', as_index=False)['gold'].sum().sort_values('gold', ascending=False)
    st.dataframe(year_gold)
    if not year_gold.empty:
        st.write(f"Year with most gold medals: **{int(year_gold.iloc[0]['year'])}** ({int(year_gold.iloc[0]['gold'])} golds)")

# 13. How does the medal count of the United States compare with other countries?
with st.expander("13) Compare a country (e.g., United States) with others"):
    country_choice = st.selectbox("Pick a country to compare", options=['United States'] + sorted(df_clean['country'].unique().tolist()))
    c_df = medals_by_country_year(df_clean, country_choice)
    st.dataframe(c_df)
    # show top countries side-by-side
    top_n = country_total_medals(df_clean).head(10)
    fig4 = px.bar(top_n, x='country', y='total', title=f"Top countries by total medals (top 10)")
    st.plotly_chart(fig4, use_container_width=True)

# 14. Which athletes have won medals in multiple Olympic years?
with st.expander("14) Athletes with medals in multiple years"):
    multi = athletes_multi_years(df_clean, min_years=2)
    st.dataframe(multi.sort_values('distinct_years', ascending=False) if 'distinct_years' in multi.columns else multi.head(200))

# 15. Total number of medals per sport category
with st.expander("15) Total medals per sport"):
    st.dataframe(sport_tot)

# 16. Which country dominates in swimming events?
with st.expander("16) Country dominance in a sport (e.g., Swimming)"):
    sport_sel = st.selectbox("Select sport", options=sorted(df_clean['sport'].unique()), index=0)
    sport_df = df_clean[df_clean['sport']==sport_sel]
    if not sport_df.empty:
        sport_country = sport_df.groupby('country', as_index=False)[['gold','silver','bronze','total']].sum().sort_values('total', ascending=False)
        st.write(f"Top countries in {sport_sel}:")
        st.dataframe(sport_country.head(30))
    else:
        st.write("No rows for that sport.")

# 17. Which athletes have the same number of medals as Michael Phelps?
with st.expander("17) Athletes with same total medals as a chosen athlete"):
    athlete_choice = st.text_input("Enter athlete name (e.g., Michael Phelps)", value="Michael Phelps")
    same_as = athletes_same_as(df_clean, athlete_choice)
    if same_as.empty:
        st.write("No match or athlete not found in aggregated totals.")
    else:
        st.dataframe(same_as)

# 18. How many total medals were won in each Olympic year?
with st.expander("18) Total medals per Olympic year (detailed)"):
    st.dataframe(total_medal_count_per_year(df_clean))

# 19. Which countries improved their performance in consecutive Olympic Games?
with st.expander("19) Countries with improvements over consecutive Olympics"):
    # compute consecutive differences per country
    ct = df_clean.groupby(['country','year'], as_index=False)[['total']].sum().sort_values(['country','year'])
    ct['prev_total'] = ct.groupby('country')['total'].shift(1)
    ct['change'] = ct['total'] - ct['prev_total']
    improved = ct[ct['change']>0]
    st.write("Examples of positive improvements by country and year:")
    st.dataframe(improved.head(200))

# 20. Average number of medals per athlete
with st.expander("20) Average medals per athlete"):
    avg_per_athlete = athlete_total_medals(df_clean)['total'].mean()
    st.write(f"Average medals per athlete: **{avg_per_athlete:.2f}**")

# 21. How many athletes won at least one gold medal?
with st.expander("21) Number of athletes with at least one gold medal"):
    gcount = df_clean.groupby('name', as_index=False)['gold'].sum()
    num = (gcount['gold']>0).sum()
    st.write(f"Number of athletes with at least one gold medal: **{int(num)}**")

# 22. Which sport has the most diverse set of medal-winning countries?
with st.expander("22) Sport diversity (num of distinct countries winning medals)"):
    divers = df_clean.groupby('sport')['country'].nunique().reset_index(name='num_countries').sort_values('num_countries', ascending=False)
    st.dataframe(divers)

# 23. Medal-to-athlete ratio for each country
with st.expander("23) Medal-to-athlete ratio by country"):
    mratio = medal_to_athlete_ratio(df_clean)
    st.dataframe(mratio.head(50))

# 24. Country with highest average gold medals per year
with st.expander("24) Country: highest average gold medals per year"):
    cyr = df_clean.groupby(['country','year'], as_index=False)['gold'].sum()
    avg_gold_per_country = cyr.groupby('country', as_index=False)['gold'].mean().sort_values('gold', ascending=False)
    st.dataframe(avg_gold_per_country.head(20))

# 25. Proportion of gold medals to total medals per country
with st.expander("25) Gold proportion per country"):
    ctot_all = df_clean.groupby('country', as_index=False)[['gold','total']].sum()
    ctot_all['gold_prop'] = (ctot_all['gold'] / ctot_all['total']).fillna(0).round(3)
    st.dataframe(ctot_all.sort_values('gold_prop', ascending=False).head(30))

# 26. How many athletes participated for each country?
with st.expander("26) Number of athletes per country"):
    ath_by_country = df_clean.groupby('country')['name'].nunique().reset_index(name='num_athletes').sort_values('num_athletes', ascending=False)
    st.dataframe(ath_by_country)

# 27. Top performers by total medal count
with st.expander("27) Top athletes by total medals"):
    st.dataframe(athlete_total_medals(df_clean).head(50))

# 28. Trend of total medals for each sport over time
with st.expander("28) Trend: total medals per sport over time"):
    sport_time = df_clean.groupby(['year','sport'], as_index=False)['total'].sum()
    sport_choice = st.selectbox("Pick sport to plot trend", options=['All'] + sorted(df_clean['sport'].unique().tolist()))
    if sport_choice!='All':
        st.dataframe(sport_time[sport_time['sport']==sport_choice])
        fig5 = px.line(sport_time[sport_time['sport']==sport_choice], x='year', y='total', title=f"{sport_choice} medals over time")
        st.plotly_chart(fig5, use_container_width=True)
    else:
        # show stacked area for top 6 sports
        top_sports = sport_total_medals(df_clean).head(6)['sport'].tolist()
        df_area = sport_time[sport_time['sport'].isin(top_sports)].pivot(index='year', columns='sport', values='total').fillna(0)
        st.dataframe(df_area)
        fig6 = px.area(df_area.reset_index(), x='year', y=df_area.columns.tolist(), title='Top sports medal trend (area)')
        st.plotly_chart(fig6, use_container_width=True)

# 29. How many medals did each country win in a particular year (e.g., 2012)?
with st.expander("29) Country medal counts in a particular year"):
    year_for_query = st.number_input("Enter year", value=int(df_clean['year'].min() if not df_clean.empty else 2000))
    ct_year = df_clean[df_clean['year']==int(year_for_query)].groupby('country', as_index=False)[['gold','silver','bronze','total']].sum().sort_values('total', ascending=False)
    st.write(f"Medal counts for year {int(year_for_query)}")
    st.dataframe(ct_year)

# 30. How many total medals were won by female vs male athletes? (requires gender column)
with st.expander("30) Gender split (if gender column exists)"):
    gender_col = None
    for c in df_clean.columns:
        if c.lower() in ('gender','sex'):
            gender_col = c
            break
    if gender_col:
        gsplit = df_clean.groupby(gender_col)[['gold','silver','bronze','total']].sum().reset_index()
        st.dataframe(gsplit)
    else:
        st.info("No gender column found. If you have a gender column name, remap it to 'gender' using the sidebar.")

# 31. Youngest and oldest medal winner
with st.expander("31) Youngest and oldest medal winners"):
    if 'age' in df_clean.columns and df_clean['age'].replace(0, np.nan).notna().sum()>0:
        min_age = df_clean['age'].min()
        max_age = df_clean['age'].max()
        st.write(f"Youngest (min age): {min_age} — athletes:")
        st.dataframe(df_clean[df_clean['age']==min_age][['name','age','country','year','sport']])
        st.write(f"Oldest (max age): {max_age} — athletes:")
        st.dataframe(df_clean[df_clean['age']==max_age][['name','age','country','year','sport']])
    else:
        st.write("No usable age data found.")

# 32. Athletes who won only bronze medals
with st.expander("32) Athletes who won only bronze medals"):
    agg = df_clean.groupby('name', as_index=False)[['gold','silver','bronze','total']].sum()
    only_bronze = agg[(agg['gold']==0) & (agg['silver']==0) & (agg['bronze']>0)]
    st.dataframe(only_bronze)

# 33. Athletes who won medals in multiple sports
with st.expander("33) Athletes who won medals in multiple sports"):
    multi_sport = df_clean.groupby('name')['sport'].nunique().reset_index(name='num_sports')
    multi_sport = multi_sport[multi_sport['num_sports']>1].sort_values('num_sports', ascending=False)
    st.dataframe(multi_sport)

# 34. How gold counts vary across countries and years (heatmap)
with st.expander("34) Heatmap: gold counts by country and year"):
    heat = df_clean.groupby(['country','year'], as_index=False)['gold'].sum()
    heat_pivot = heat.pivot(index='country', columns='year', values='gold').fillna(0)
    st.dataframe(heat_pivot.head(50))
    fig7, ax = plt.subplots(figsize=(10,6))
    # plot a small heatmap using imshow (no external libs)
    im = ax.imshow(heat_pivot.values, aspect='auto')
    ax.set_yticks(range(len(heat_pivot.index)))
    ax.set_yticklabels(heat_pivot.index)
    ax.set_xticks(range(len(heat_pivot.columns)))
    ax.set_xticklabels(heat_pivot.columns, rotation=45)
    ax.set_title("Gold medals by country (rows) and year (columns)")
    st.pyplot(fig7)

st.sidebar.markdown("---")
st.sidebar.write("App by ChatGPT — load data, map columns, then explore the expanders above.")
