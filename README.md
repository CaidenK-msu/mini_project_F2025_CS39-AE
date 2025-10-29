# Streaming Platform insight Dashboard

Mini-Project (Lab 4.3):

Author: Caiden Kopcik

App URL: https://miniprojectf2025cs39-ae-jfasnjsvxu3c8mjahbqzmz.streamlit.app/

github Repo link: https://github.com/CaidenK-msu/mini_project_F2025_CS39-AE 

# Overview
This Streamlit dashboard should help to vizulize "possible synthetic" or local CSV data representing viewing activity on a streaming platform. It should allow users to interactively explore watch-time patterns by date range, genre, region, and provide clear visuals for both summary metrics and trends.

# Features
Sidebar controls:
- Date range picker
- Genre multiselect
- Region selector
- Map style and color-intensity options
  
Visuals & KPIs:
- Total watch hours over time (Altair area chart)
- Regional distribution of viewing time (PyDeck map)
- Key metrics for subscribers, average watch hours, and top genre
- Summary of hot genres and active days
- Top-10 user watch totals (table)

Data Handling:
- Reads mainly from local CSV (data/viewing_activity.csv)
- Auto-generates synthetic data for any date range when no local data exist
