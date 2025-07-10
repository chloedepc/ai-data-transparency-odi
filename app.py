import pandas as pd
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Load cleaned dataset
df_dashboard = pd.read_csv("data/epochai_transparency.csv")

# Initialize app
app = dash.Dash(__name__)
app.title = "AI Data Transparency (EpochAI Notable AI Models Analysis)"

# ----------------------
# Create figures & tables
# ----------------------
transparency_hist_fig = px.histogram(df_dashboard, x="transparency_score")

# ---Summary data for models trained from scratch----

# Filter out 'Unknown' and finetuned models (i.e. those with base_model or finetune compute)
df_filtered = df_dashboard[
    (df_dashboard["org_region"] != "Unknown") &
    (df_dashboard["org_category"] != "Unknown") &
    (df_dashboard["base_model"] == "Unspecified") &
    (df_dashboard["finetune_compute_disclosed"] == 0)
].copy()

# Number of models trained from scratch 
total_models = len(df_dashboard)
trained_models_count = len(df_filtered)
percent_trained = trained_models_count / total_models * 100

# Compute group summaries
summary = []

summary.append(["Models Trained from Scratch", f"{trained_models_count} ({percent_trained:.1f}% of total models)"])


# Overall average transparency score
overall_avg = df_filtered["transparency_score"].mean()
summary.append(["Overall Average Transparency Score", f"{overall_avg:.2f}/4"])

# Highest/lowest rating org_category
by_category = df_filtered.groupby("org_category")["transparency_score"].mean()
summary.append(["Highest Rating Developer Organization Category", by_category.idxmax()])
summary.append(["Lowest Rating Developer Organization Category", by_category.idxmin()])

# Highest/lowest rating region
by_region = df_filtered.groupby("org_region")["transparency_score"].mean()
summary.append(["Highest Rating Developer Region", by_region.idxmax()])
summary.append(["Lowest Rating Developer Region", by_region.idxmin()])

# Results by model openness
# Define access types
open_weights_models = [
    "Open weights (non-commercial)",
    "Open weights (unrestricted)",
    "Open weights (restricted use)"
]

closed_models = [
    "API access",
    "Hosted access (no API)",
    "Unreleased"
]

# Compute average score for each group
open_avg = df_filtered[df_filtered["model_accessibility"].isin(open_weights_models)]["transparency_score"].mean()
closed_avg = df_filtered[df_filtered["model_accessibility"].isin(closed_models)]["transparency_score"].mean()

summary.append(["Average Transparency Score - Open Weights Models", f"{open_avg:.2f}"])
summary.append(["Average Transparency Score - Closed Models (API/Unreleased)", f"{closed_avg:.2f}"])

# % models disclosing each component
components = {
    "Parameters Transparency": df_filtered["parameters_disclosed"].mean(),
    "Training Compute Transparency": df_filtered["training_compute_disclosed"].mean(),
    "Training Dataset Transparency": df_filtered["training_data_disclosed"].mean(),
    "Training Dataset Size Transparency": df_filtered["training_dataset_size_disclosed"].mean()
}

for label, val in components.items():
    summary.append([f"% {label}", f"{val * 100:.1f}%"])

# Create summary DataFrame
summary_df = pd.DataFrame(summary, columns=["Trained Model Breakdown", "Value"])

# ----- Summary data for finetuned models ------
# Identify finetuned models (base model or finetune compute disclosed)
is_finetuned = (
    (df_dashboard["finetune_compute_disclosed"] == 1) |
    (df_dashboard["base_model"] != "Unspecified")
)

finetuned_df = df_dashboard[is_finetuned].copy()

# Total models and finetuned model count
finetuned_count = len(finetuned_df)
percent_finetuned = finetuned_count / total_models * 100

# Component-level disclosures
base_model_specified = (finetuned_df["base_model"] != "Unspecified").sum()
finetune_disclosed = (finetuned_df["finetune_compute_disclosed"] == 1).sum()
training_disclosed = (finetuned_df["training_data_disclosed"] == 1).sum()
training_size_disclosed = (finetuned_df["training_dataset_size_disclosed"] == 1).sum()
parameters_disclosed = (finetuned_df["parameters_disclosed"] == 1).sum()


# Create summary DataFrame
summary_finetuned = pd.DataFrame([
    ["Finetuned Models", f"{finetuned_count} ({percent_finetuned:.1f} of total models%)"],
    ["% Base Model Transparency", f"{base_model_specified / finetuned_count * 100:.1f}%"],
    ["% Finetune Compute Transparency", f"{finetune_disclosed / finetuned_count * 100:.1f}%"],
    ["% Training Dataset Transparency", f"{training_disclosed / finetuned_count * 100:.1f}%"],
    ["% Training Dataset Size Transparency", f"{training_size_disclosed / finetuned_count * 100:.1f}%"],
    ["% Parameters Disclosed Transparency", f"{parameters_disclosed / finetuned_count * 100:.1f}%"],
], columns=["Finetuned Model Breakdown", "Value"])

summary_table1 = dash_table.DataTable(
    columns=[{"name": i, "id": i} for i in summary_df.columns],
    data=summary_df.to_dict("records"),
    style_table={'overflowX': 'auto'},
    style_header={
        'backgroundColor': '#072589',
        'color': 'white',
        'fontWeight': 'bold',
        'textAlign': 'left',
        'fontFamily': 'Helvetica Neue, Helvetica, Arial, sans-serif'
    },
    style_data={
        'backgroundColor': 'white',
        'color': 'black',
        'textAlign': 'left',
        'fontFamily': 'Helvetica Neue, Helvetica, Arial, sans-serif',
        'fontSize': '14px'
    },
    style_cell={
        'padding': '8px'
    },
    style_data_conditional=[
        {
            'if': {'column_id': summary_df.columns[0]},  # Left column
            'fontWeight': 'bold'
        }
    ]
    
)

summary_table2 = dash_table.DataTable(
    columns=[{"name": i, "id": i} for i in summary_finetuned.columns],
    data=summary_finetuned.to_dict("records"),
    style_table={'overflowX': 'auto'},
    style_header={
        'backgroundColor': '#072589',
        'color': 'white',
        'fontWeight': 'bold',
        'textAlign': 'left',
        'fontFamily': 'Helvetica Neue, Helvetica, Arial, sans-serif'
    },
    style_data={
        'backgroundColor': 'white',
        'color': 'black',
        'textAlign': 'left',
        'fontFamily': 'Helvetica Neue, Helvetica, Arial, sans-serif',
        'fontSize': '14px'
    },
    style_cell={
        'padding': '8px'
    },
    style_data_conditional=[
        {
            'if': {'column_id': summary_finetuned.columns[0]},  # Left column
            'fontWeight': 'bold'
        }
    ]
)

# -------- Transparency Score Across Model Accessibility Box Plot -------

# Custom color list from your palette (can extend or truncate as needed)
custom_colors = [
    "#2254F4", "#178CFF", "#00B6FF", "#08DEF9", "#1DD3A7",
    "#0DBC37", "#67EE67", "#C0E236", "#F9BC26", "#FF6700",
    "#D60303", "#D73058", "#EF3AAB", "#E6007C", "#B13198", "#722EA5"
]

# Color mapping for accessibility types
color_map = {
    "API access": "#2254F4",
    "Hosted access (no API)": "#00B6FF",
    "Unreleased": "#D60303",
    "Open weights (non-commercial)": "#67EE67",
    "Open weights (unrestricted)": "#F9BC26",
    "Open weights (restricted use)": "#EF3AAB"
}

# Filter the data
df_by_access = df_dashboard[df_dashboard["model_accessibility"] != "Unspecified"]

# Build box traces
traces = []
for access_type in df_by_access["model_accessibility"].unique():
    subset = df_by_access[df_by_access["model_accessibility"] == access_type]
    traces.append(go.Box(
        y=subset["transparency_score"],
        name=access_type,
        marker_color=color_map.get(access_type, "#CCCCCC"),
        boxpoints="outliers",
        jitter=0.3,
        pointpos=0,
        customdata=np.stack((subset["model"], subset["organization"]), axis=-1),
        hovertemplate="<b>Model</b>: %{customdata[0]}<br>" +
                      "<b>Developer Organization</b>: %{customdata[1]}<extra></extra>",
        showlegend=False
    ))

# Create figure
accessibility_box_plot = go.Figure(data=traces)

# Update layout
accessibility_box_plot.update_layout(
    title="Transparency Score by Model Accessibility Type",
    xaxis_title="Model Accessibility",
    yaxis_title="Transparency Score",
    plot_bgcolor='white',
    xaxis=dict(showgrid=False,
               tickfont=dict(size=14),  
               title_font=dict(size=16)), 
    yaxis=dict(showgrid=True, gridcolor='lightgrey',
               tickfont=dict(size=14), 
               title_font=dict(size=16)),
    font=dict(family='Helvetica Neue, Helvetica, Arial, sans-serif'),
    title_font_size=18,
    height=600,
    width=800
)

# ----------------------
# Define constants and mappings
# ----------------------
dropdown_options = {
    "Developer Region": "org_region",
    "Developer Organization Type": "org_category",
    "Model Accessibility": "model_accessibility"
}

column_title_map = {
    "org_category": "Developer Organization Type",
    "org_region": "Developer Region",
    "model_accessibility": "Model Accessibility"
}

field_label_map = {
    "parameters_disclosed": "Parameters",
    "training_data_disclosed": "Training Data",
    "training_dataset_size_disclosed": "Dataset Size",
    "training_compute_disclosed": "Training Compute"
}

transparency_fields = list(field_label_map.keys())

# ----------------------
# Layout
# ----------------------
app.layout = html.Div([
    # ------- Header & Intro ---------
    html.Div([
        html.H1("AI Data Transparency: A Deep-Dive into EpochAI's Notable Models Dataset"), 
        html.H4("Overview"),
        html.P("This dashboard provides an overview of transparency characteristics across AI foundation models based on the EpochAI Notable Models dataset. It highlights trends across model types, organizations, and geographic regions.")
    ], className="full-width-text"),

    # ------- Section 1: Tables with Results Summary on the left ---------
    html.Div([
        # Sidebar narrative:
        html.Div([
            html.H2("Results Summary"),
            dcc.Markdown("""
            **Key insights**:
            - Models trained from scratch vs finetuned  
            - Component-level transparency  
            - Regional and organizational trends  
            - Variations in model openness
            """)
        ], className="sidebar"),

        # Main content (tables)
        html.Div([
            html.H4("Models Trained from Scratch", className="table-title"),
            summary_table1,
            html.Br(),
            html.H4("Finetuned Models", className="table-title"),
            summary_table2
        ], className="visual")  
    ], className="section"),


    # ------- Section 2: Histogram with side narrative --------
    html.Div([
        # Transparency Score Distribution Description
        html.Div([
            html.H2("Transparency Score Distribution"),
            dcc.Markdown("""
            This histogram shows how AI models are distributed across different transparency scores.
            Use this to identify clusters, outliers or potential skewness in upstream reporting patterns.
         """)
        ], className="sidebar"),

        # Transparency Score Distribution Visual
        html.Div([
            dcc.Graph(figure=transparency_hist_fig)
        ], className='visual')
    ], className='section'),


    # ------ Section 3: Key Insight Box Visual 1 --------
    html.Div([
        html.H4("Key Insight"),
        html.P("Most open models tend to have higher transparency scores.")
    ], className='insight-box'),

    # ------ Section 4: Component-Level Transparency Heatmap -----

    html.Div([
        html.Div([
            html.H2("Component-Level Transparency Heatmap"),
            dcc.Markdown("""
            Text Placeholder
         """)
        ], className="sidebar"),
    
        html.Div([
            dcc.Dropdown(
                id="comparison-dropdown",
                options=[{"label": k, "value": v} for k, v in dropdown_options.items()],
                value="org_region",  # Default selection
                clearable=False,
                # style={"width": "400px", "margin-bottom": "20px"}?
            ),
            dcc.Graph(id="heatmap-graph")
        ], className='visual')
    ], className='section'),

    # ------ Section 5: Key Insight Box Visual 2 --------
    html.Div([
        html.H4("Key Insight"),
        html.P("Most open models tend to have higher transparency scores.")
    ], className='insight-box'),

   # ------ Section 6: Transparency Across Model Accessibility -----

    html.Div([
        html.Div([
            html.H2("Transparency Score Distributions Across Model Openness"),
            dcc.Markdown("""
            Text Placeholder
         """)
        ], className="sidebar"),
    
        html.Div([
            dcc.Graph(figure=accessibility_box_plot)
        ], className='visual')
    ], className='section'),

    # ------ Section 7: Key Insight Box Visual 3 --------
    html.Div([
        html.H4("Key Insight"),
        html.P("Most open models tend to have higher transparency scores.")
    ], className='insight-box')



])


# ------------ Define Callbacks ----------------


# Callback
@app.callback(
    Output("heatmap-graph", "figure"),
    Input("comparison-dropdown", "value")
)
def update_heatmap(comparison_column):
    # Filter out 'Unknown' and 'Unspecified'
    df_filtered = df_dashboard[
        df_dashboard[comparison_column].notna() &
        (~df_dashboard[comparison_column].astype(str).isin(["Unknown", "Unspecified"]))
    ].copy()

    # Compute mean disclosure
    heatmap_data = (
        df_filtered
        .groupby(comparison_column)[transparency_fields]
        .mean()
        .transpose()
        * 100
    )

    heatmap_data.rename(index=field_label_map, inplace=True)

    # Sorth columns by average
    column_order = heatmap_data.mean(axis=0).sort_values(ascending=False).index
    heatmap_data = heatmap_data[column_order]

    transparency_heatmap_fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Viridis',
        colorbar=dict(title="% Transparent"),
        hovertemplate=f"{column_title_map[comparison_column]}: "+"%{x}<br>Field: %{y}<br>% Transparent: %{z:.1f}%<extra></extra>"
    ))

    transparency_heatmap_fig.update_layout(
        title=f"Component-Level Transparency by {column_title_map[comparison_column]}",
        xaxis_title=column_title_map[comparison_column],
        yaxis_title="Transparency Component",
        xaxis=dict(tickangle=45, tickfont=dict(size=14), title_font=dict(size=16)),
        yaxis=dict(tickfont=dict(size=14), title_font=dict(size=16)),
        font=dict(family='Helvetica Neue, Helvetica, Arial, sans-serif'),
        title_font_size=20,
        height=600
    )
    return transparency_heatmap_fig


if __name__ == '__main__':
    app.run()