import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def normalize_topic_name(topic):
    if pd.isna(topic):
        return None
    normalized = str(topic).replace('#', '').strip()
    if normalized.startswith(tuple(str(i) for i in range(10))):
        normalized = normalized.split(':', 1)[1].strip()
    if normalized == 'Possible Points':
        return None
    if normalized == 'Indigenous Tenure':
        return 'Indigenous Tenure'
    if normalized == 'Marine carbon sinks':
        return 'Marine Carbon Sinks'
    return normalized

def process_score_sheet(file_path):
    try:
        # Try different encodings
        encodings = ['utf-8', 'cp1252', 'latin1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, header=None, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            st.error(f"Could not read {file_path} with any encoding")
            return None
            
        # Find the judge name
        judge_row = df[df[0].str.contains('Scored by:', na=False)]
        judge_name = judge_row[0].iloc[0].replace('Scored by:', '').strip() if not judge_row.empty else 'Unknown'
        
        # Find the criteria row
        criteria_idx = df[df[0] == 'Criteria'].index[0]
        
        # Get topics (column headers)
        topics = df.iloc[criteria_idx, 1:].apply(normalize_topic_name)
        topics = topics[topics.notna()]
        
        # Extract scores for each criterion
        scores = {}
        for criterion in ['Behavioral Necessity', 'Impact on Nature', 'Impact on Human Welfare']:
            criterion_row = df[df[0].str.contains(criterion, na=False, regex=False)]
            if not criterion_row.empty:
                scores[criterion] = pd.to_numeric(criterion_row.iloc[0, 1:len(topics)+1], errors='coerce')
        
        # Skip if any criterion is missing
        if len(scores) != 3:
            st.warning(f"Missing criteria in {file_path}")
            return None
            
        # Create DataFrame with scores
        scores_df = pd.DataFrame(scores)
        scores_df['Topic'] = topics.values
        scores_df['Judge'] = judge_name
        
        # Calculate total
        scores_df['Total'] = scores_df[['Behavioral Necessity', 'Impact on Nature', 'Impact on Human Welfare']].sum(axis=1)
        
        # Drop rows with any NaN values
        scores_df = scores_df.dropna()
        
        return scores_df
    
    except Exception as e:
        st.error(f"Error processing {file_path}: {str(e)}")
        return None

def adjust_scores(all_scores_df):
    # Calculate judge averages for numeric columns only
    numeric_cols = ['Behavioral Necessity', 'Impact on Nature', 'Impact on Human Welfare', 'Total']
    judge_avgs = all_scores_df.groupby('Judge')[numeric_cols].mean()
    
    # Calculate global averages
    global_avgs = all_scores_df[numeric_cols].mean()
    
    # Create adjusted scores
    adjusted_df = all_scores_df.copy()
    
    for criterion in numeric_cols:
        for judge in judge_avgs.index:
            mask = adjusted_df['Judge'] == judge
            adjustment = judge_avgs.loc[judge, criterion] - global_avgs[criterion]
            adjusted_df.loc[mask, criterion] -= adjustment
    
    return adjusted_df

def create_heatmap_table(topic_avgs, selected_metric):
    # Sort by selected metric
    sorted_data = topic_avgs.sort_values(selected_metric, ascending=False)
    
    # Create a Plotly table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Rank', 'Topic', 'Behavioral', 'Nature', 'Human', 'Total'],
            font=dict(size=12, color='black'),
            align='left',
            fill_color='lightgray'
        ),
        cells=dict(
            values=[
                range(1, len(sorted_data) + 1),
                sorted_data.index,
                sorted_data['Behavioral Necessity'].round(1),
                sorted_data['Impact on Nature'].round(1),
                sorted_data['Impact on Human Welfare'].round(1),
                sorted_data['Total'].round(1)
            ],
            font=dict(size=11),
            align='left',
            format=[None, None, '.1f', '.1f', '.1f', '.1f'],
            fill_color=[
                'white',
                'white',
                get_color_scale(sorted_data['Behavioral Necessity']),
                get_color_scale(sorted_data['Impact on Nature']),
                get_color_scale(sorted_data['Impact on Human Welfare']),
                ['lightgray'] * len(sorted_data)
            ]
        )
    )])
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=35 * len(sorted_data)
    )
    
    return fig

def get_color_scale(values):
    min_val = values.min()
    max_val = values.max()
    normalized = (values - min_val) / (max_val - min_val)
    
    colors = []
    for n in normalized:
        r = int(255 * (1 - n))
        g = int(255 * n)
        colors.append(f'rgb({r}, {g}, 0)')
    
    return colors

def main():
    st.set_page_config(layout="wide")
    st.title('Topic Rankings')
    
    # List of score sheet files
    files = [
        'Score Sheet_Larissa.csv', 'Score Sheet_Michelle.csv', 'Score Sheet_MID.csv',
        'Score Sheet_Natalia.csv', 'Score Sheet_Nikita Patelv2.csv', 'Score Sheet_Philipe.csv',
        'Score Sheet_Sam.csv', 'Score Sheet_Sania.csv', 'Score Sheet_Zach Hoffman.csv',
        'Score Sheet_Kristi.csv', 'Score Sheet_Tony.csv', 'Score Sheet_Kevin.csv',
        'Score Sheet_Fel.csv', 'Score Sheet_KatieH.csv', 'Score Sheet_Kate M.csv',
        'Score Sheet_Tanmatra.csv', 'Score Sheet_Travis.csv', 'Score Sheet_ Rakhim.csv',
        'Score Sheet_Anam.csv'
    ]
    
    # Process all score sheets
    all_scores = []
    for file in files:
        scores_df = process_score_sheet(file)
        if scores_df is not None:
            all_scores.append(scores_df)
    
    # Combine all scores
    if all_scores:
        all_scores_df = pd.concat(all_scores, ignore_index=True)
        
        # Calculate adjusted scores
        adjusted_scores = adjust_scores(all_scores_df)
        
        # Calculate topic averages
        topic_avgs = adjusted_scores.groupby('Topic').mean()
        
        # Create tabs for different metrics
        metric = st.radio(
            "Select Score Type",
            ["Overall Score", "Behavioral Necessity", "Impact on Nature", "Impact on Human Welfare"],
            horizontal=True
        )
        
        # Map display names to column names
        metric_map = {
            "Overall Score": "Total",
            "Behavioral Necessity": "Behavioral Necessity",
            "Impact on Nature": "Impact on Nature",
            "Impact on Human Welfare": "Impact on Human Welfare"
        }
        
        # Display the heatmap table
        fig = create_heatmap_table(topic_avgs, metric_map[metric])
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("No valid data was processed")

if __name__ == "__main__":
    main()