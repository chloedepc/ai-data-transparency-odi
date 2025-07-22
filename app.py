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
# Create tables
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
summary.append(["Overall Average Transparency Score", f"{overall_avg:.2f}/4.0"])

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
    ["Finetuned Models", f"{finetuned_count} ({percent_finetuned:.1f}% of total models)"],
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


#----------------------------
# Data Preparation for figures
# ---------------------------

# Custom color list from your palette (can extend or truncate as needed)
custom_colors = [
    "#2254F4", "#178CFF", "#00B6FF", "#08DEF9", "#1DD3A7",
    "#0DBC37", "#67EE67", "#C0E236", "#F9BC26", "#FF6700",
    "#D60303", "#D73058", "#EF3AAB", "#E6007C", "#B13198", "#722EA5"
]

# Region color mapping
region_colors = [
    "#2254F4",  # deep blue
    "#FF6700",  # bright orange
    "#1DD3A7",  # teal
    "#D73058",  # magenta (red-purple)
    "#C0E236",  # yellow-green
    "#722EA5"   # purple (for Cross‑Regional Collaboration)
]

# Color mapping for accessibility types
access_color_map = {
    "API access": "#2254F4",             # deep blue
    "Hosted access (no API)": "#08DEF9", # slightly darker than current "#00B6FF"   
    "Unreleased": "#D60303",                    # bold red
    "Open weights (non-commercial)": "#0DBC37", # green
    "Open weights (unrestricted)": "#F9BC26",   # yellow-orange
    "Open weights (restricted use)": "#EF3AAB"  # magenta
}

# Organization type custom color mapping
org_color_map = {
    "Academia": "#2254F4",                          # deep blue
    "Industry": "#FF6700",                          # vivid orange
    "Industry-Academia Collaboration": "#1DD3A7",   # teal
    "Government / Public Sector": "#D60303",        # red
    "Research Collective": "#722EA5",               # deep purple
    "Cross-sector Collaboration": "#0DBC37",        # medium green
    "Unknown": "#A9A9A9"                            # gray for neutrality
}

transparency_color_map = {
    "Parameters": "#178CFF",       # vivid blue
    "Training Data": "#C0E236",    # chartreuse
    "Dataset Size": "#EF3AAB",     # pink
    "Training Compute": "#B13198"  # plum
}

# -------- Transparency Score Across Model Accessibility Box Plot -------


    # Filter the data
df_by_access = df_dashboard[
    (df_dashboard["model_accessibility"] != "Unspecified") &
    (df_dashboard["transparency_score"].notna())
]

    # Build box traces
traces = []
for access_type in df_by_access["model_accessibility"].unique():
    subset = df_by_access[df_by_access["model_accessibility"] == access_type]
    traces.append(go.Box(
        y=subset["transparency_score"],
        name=access_type,
        marker_color=access_color_map.get(access_type, "#CCCCCC"),
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
    title=dict(text="Transparency Score by Model Accessibility Type",
               pad=dict(b=0),
               x=0.5,
               xanchor='center'
               ),
    yaxis_title="Transparency Score",
    plot_bgcolor='white',
    xaxis=dict(showgrid=False,
               tickfont=dict(size=12),  
               title_font=dict(size=14)), 
    yaxis=dict(showgrid=True, gridcolor='lightgrey',
               tickfont=dict(size=12), 
               title_font=dict(size=14)),
    font=dict(family='Helvetica Neue, Helvetica, Arial, sans-serif'),
    title_font_size=15,
    height=600,
    width=700
)


# ----------- Transparency Score Distribution by Organization Type --------------

    # Create histogram with direct color mapping
transparency_hist_fig = px.histogram(
    df_dashboard,
    x='transparency_score',
    color='org_category',
    nbins=6,
    width=700,  
    height=500,
    barmode='stack',
    title='Distribution of Transparency Scores by Developer Organization Type',
    category_orders={"transparency_score": sorted(df_dashboard["transparency_score"].unique())},
    color_discrete_map=org_color_map,  
    hover_data=["org_category"],
    labels={
        "transparency_score": "Transparency Score",
        "org_category": "Developer Organization Type",
        "count": "Number of Models"
    },
    opacity=0.7
)

    # Update layout
transparency_hist_fig.update_layout(
    plot_bgcolor='#f9f9f9',
    paper_bgcolor='white',
    font=dict(size=12, family='Helvetica Neue, Helvetica, Arial, sans-serif'),
    title=dict(
        pad=dict(b=0),
        x=0.5,
        xanchor='center'
        ),
    title_font=dict(size=15),
    xaxis=dict(
        title="Transparency Score (0–4)",
        tickmode='linear',
        dtick=1,
        title_font=dict(size=14)
    ),
    yaxis=dict(
        title="Number of Models",
        title_font=dict(size=14)
    ),
    legend=dict(
        title="Developer Organization Type",
        orientation="v",
        bgcolor="rgba(255,255,255,0)",
        bordercolor="LightGrey",
        borderwidth=1,
        font=dict(size=11)
    )
)

# Map category to trace index manually
for trace in transparency_hist_fig.data:
    category = trace.name  # Each trace's name is the org_category
    trace.marker.line.width = 1.3  
    trace.marker.line.color = org_color_map.get(category, "#000000")  # Opaque border in same color
    trace.hovertemplate = (
        "<b>Transparency Score</b>: %{x}<br>" +
        "<b>Number of Models</b>: %{y}<br>" +
        f"<b>Developer Organization Type</b>: {category}<extra></extra>"
    )

# --------- Variance of Transparency Scores by Organization Type --------

# Filter the data
df_by_org = df_dashboard[
    (df_dashboard["transparency_score"].notna())
]

# Build box traces
org_box_traces = []
for org_type in df_by_org["org_category"].unique():
    subset = df_by_org[df_by_org["org_category"] == org_type]
    org_box_traces.append(go.Box(
        y=subset["transparency_score"],
        name=org_type,
        marker_color=org_color_map.get(org_type, "#CCCCCC"),
        boxpoints="outliers",  # only show outliers
        jitter=0.3,
        pointpos=0,
        customdata=np.stack((subset["model"], subset["organization"]), axis=-1),
        hovertemplate="<b>Model</b>: %{customdata[0]}<br>" +
                      "<b>Developer Organization</b>: %{customdata[1]}<extra></extra>",
        showlegend=False
    ))

# Create figure
transparency_box_fig = go.Figure(data=org_box_traces)

# Update layout
transparency_box_fig.update_layout(
    title=dict(text="Variation in Transparency Scores by Developer Organization Type",
               pad=dict(b=0),
               x=0.5,
               xanchor='center'
               ),
    yaxis_title="Transparency Score",
    plot_bgcolor='white',
    xaxis=dict(showgrid=False,
               tickfont=dict(size=12),
               title_font=dict(size=14)),
    yaxis=dict(showgrid=True, gridcolor='lightgrey',
               tickfont=dict(size=12),
               title_font=dict(size=14),
               range=[-0.5, 4.5]),
    font=dict(family='Helvetica Neue, Helvetica, Arial, sans-serif'),
    title_font_size=15,
    height=550,
    width=700
)

# --------- Confidence Level for Estimates -----------

    # Filter models that provide training compute in the dataset    
df_conf = df_dashboard[df_dashboard["training_compute_disclosed"] == 1].copy()

    # Count of models per confidence level
confidence_counts = df_conf["confidence"].value_counts().reset_index()
confidence_counts.columns = ["confidence", "model_count"]

    # Specify order
confidence_order = ["Confident", "Likely", "Speculative", "Unknown"]
confidence_counts["confidence"] = pd.Categorical(
    confidence_counts["confidence"],
    categories=confidence_order,
    ordered=True
)
confidence_counts = confidence_counts.sort_values("confidence")

    # Create figure
confidence_fig = px.bar(
    confidence_counts,
    x="confidence",
    y="model_count",
    title="Confidence Levels for Training Compute Estimates",
    labels={
        "confidence": "Confidence Level",
        "model_count": "Number of Models"
    },
    color_discrete_sequence=["#2254F4"],
    text="model_count"
)

    # Layout styling
confidence_fig.update_layout(
    xaxis_title="Confidence Level",
    yaxis_title="Number of Models",
    font=dict(size=14, family='Helvetica Neue, Helvetica, Arial, sans-serif'),
    plot_bgcolor="#E2E6E9",
    paper_bgcolor="#E2E6E9",
    margin=dict(l=20, r=20, t=60, b=20),
    height=350,
    bargap=0.3,
    title={'xanchor': 'center', 'x': 0.5}
)

confidence_fig.update_traces(marker_line_width=0.5, marker_line_color="LightGrey")

# ----------------------
# Define constants and mappings
# ----------------------

# Clean year options
year_options = sorted(
    [int(float(y)) for y in df_dashboard["publication_year"].dropna().unique()],
    reverse=True
)

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
        html.H1("AI data transparency landscape: a deep-dive into Epoch AI's Notable Models dataset"), 
        html.H4("Overview"),
        dcc.Markdown("""
                    This page offers a snapshot of transparency across high-impact AI models based on the best available metadata from EpochAI.

                    EpochAI's dataset provides a structured lens into the otherwise opaque development of foundation models. While it is not exhaustive, and many values are estimated or inferred, it offers a rare opportunity to analyze trends in openness across organizations, regions, and time.

                    This dashboard presents a snapshot in time, summarizing the most visible metadata available as of mid-2025. It does not represent a complete inventory of disclosures, nor does it capture every nuance of transparency. Instead, it serves as a starting point to:

                    - Surface patterns in how developers approach transparency  
                    - Compare organizational practices in disclosure  
                    - Track changes over time in what gets shared — and what doesn’t

                    The aim is not to rank or evaluate, but to illuminate current practices and help inform the path toward more open and responsible AI development.
                    """)
    ], className="full-width-text"),

   # ------- Section 1: Tables with Results Summary on the left ---------
    # html.Div([
    #     html.H2("What the EpochAI Dataset Reveals About Transparency"),
    #     ], className="section-title"),
    
    html.Div([
        # Sidebar narrative:
        html.Div([
            html.H2("Data transparency at a glance"),
            dcc.Markdown("""
            **Key insights**:
            """)
        ], className="sidebar"),

        # Main content (tables)
        html.Div([
            html.H4("Models trained from scratch", className="table-title"),
            summary_table1,
            html.Br(),
            html.H4("Finetuned models", className="table-title"),
            summary_table2
        ], className="visual")  
    ], className="section"),

    # ------- Section 2: Histogram with side narrative --------
    html.Div([
        # Transparency Score Distribution Description
        html.Div([
            html.H2("Where do Notable AI models stand? A landscape of transparency scores"),
            dcc.Markdown("""
            Placeholder Text
         """)
        ], className="sidebar"),

        # Transparency Score Distribution and Variance Visuals
        html.Div([
            dcc.Graph(figure=transparency_hist_fig),
            html.Br(),
            dcc.Graph(figure=transparency_box_fig)
        ], className='visual')
    ], className='section'),

    # ------ Section 3: Key Insight Box Visual 1 --------
    html.Div([
        html.H4("Key insight"),
        html.P("Placeholder text.")
    ], className='insight-box'),

    # ------ Section 4: Transparency Across Model Accessibility -----

    html.Div([
        html.Div([
            html.H2("Does openness equal transparency?"),
            dcc.Markdown("""
            Text Placeholder
         """)
        ], className="sidebar"),
    
        html.Div([
            dcc.Graph(figure=accessibility_box_plot)
        ], className='visual')
    ], className='section'),

    # ------ Section 5: Key Insight Box Visual 2 --------
    html.Div([
        html.H4("Key insight"),
        html.P("Placeholder text.")
    ], className='insight-box'),

    # ------ Section 6: Component-Level Transparency Heatmap -----

    html.Div([
        html.Div([
            html.H2("Behind the scores - a transparency breakdown"),
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

    # ------ Section 7: Key Insight Box Visual 2 --------
    html.Div([
        html.H4("Key insight"),
        html.P("Most open models tend to have higher transparency scores.")
    ], className='insight-box'),

    # ------- Section 8: Global Transparency Overview ------

    html.Div([
        html.Div([
            html.H2("Mapping data transparency around the world"),
            dcc.Markdown("""
            Text Placeholder
         """)
        ], className="sidebar"),
    
        html.Div([
            html.Label("Select start year:"),
            dcc.Dropdown(
                id="start-year-dropdown",
                options=[{"label": "All Years", "value": "all"}] +
                        [{"label": str(year), "value": str(year)} for year in year_options],
                value="all",
                clearable=False,
                placeholder="Start year",
            ),
            html.Label("Select end year:", style={"marginTop": "20px"}),
            dcc.Dropdown(
                id="end-year-dropdown",
                options=[{"label": str(year), "value": str(year)} for year in year_options],
                value=None,
                clearable=True,
                placeholder="End year",
            ),
            dcc.Graph(id="geomap")
        ], className='visual')
    ], className='section'),


    # ------ Section 9: Key Insight Box Visual 4 --------
    html.Div([
        html.H4("Key insight"),
        html.P("Placeholder text.")
    ], className='insight-box'),


    # ------ Section 10: Time Series Chart --------

    html.Div([
        html.Div([
            html.H2("Is transparency improving - or regressing?"),
            dcc.Markdown("""
            Text Placeholder
         """)
        ], className="sidebar"),
    
        html.Div([
            html.Label("Select developer organization type:", style={"marginTop": "20px"}),
            dcc.Dropdown(
                id="org-category-dropdown",
                options=[
                    {"label": "All", "value": "All"},
                    {"label": "Academia", "value": "Academia"},
                    {"label": "Industry", "value": "Industry"},
                    {"label": "Industry-Academia Collaboration", "value": "Industry-Academia Collaboration"},
                    {"label": "Research Collective", "value": "Research Collective"},
                    {"label": "Government / Public Sector", "value": "Government"},
                    {"label": "Cross-sector Collaboration"},
                    ],
                value="All",
                clearable=False
            ),
            dcc.Graph(id="time-series-chart"),
            dcc.Graph(id="component-transparency-chart")
        ], className='visual')
    ], className='section'),


    # ------ Section 11: Key Insight Box Visual 5 --------
    html.Div([
        html.H4("Key insight"),
        html.P("Placeholder text.")
    ], className='insight-box'),

    # ------ Data Transparency Considerations -------
    html.Div([
        html.Div([
            html.H2("Data transparency considerations"),
            html.H4("Notes on data confidence"),
            dcc.Markdown("""
                Confidence levels indicate how certain EpochAI is in its estimate of compute values. Most estimates are tagged as 
                'Confident' or 'Likely', though some are 'Speculative', especially for models with limited public disclosures.
                        """
                    ),
                ], className='notes-sidebar'),
 
        html.Div([
            dcc.Graph(figure=confidence_fig)
            ], className='notes-visual')
    ], className='notes-section'),
])

# ------------ Define Callbacks ----------------


# Callback heatmap
@app.callback(
    Output("heatmap-graph", "figure"),
    Input("comparison-dropdown", "value")
)
def update_heatmap(comparison_column):
    # Filter out 'Unknown' and 'Unspecified'
    df_filtered = df_dashboard[
        df_dashboard[comparison_column].notna() &
        df_dashboard["transparency_score"].notna() &
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
        title=dict(
            text=f"Component-Level Transparency by {column_title_map[comparison_column]}",
            x=0.5,
            xanchor='center',
            pad=dict(b=5)
            ),
        yaxis_title="Transparency Component",
        xaxis=dict(tickangle=45, tickfont=dict(size=12), title_font=dict(size=14)),
        yaxis=dict(tickfont=dict(size=12), title_font=dict(size=14)),
        font=dict(family='Helvetica Neue, Helvetica, Arial, sans-serif'),
        title_font_size=15,
        height=600
    )
    return transparency_heatmap_fig

# Callback geomap

@app.callback(
    Output("geomap", "figure"),
    Input("start-year-dropdown", "value"),
    Input("end-year-dropdown", "value")
)
def update_geomap(start_year, end_year):
    df_map = df_dashboard.copy()
    df_map = df_map[(df_map["org_country"].notna()) &
                    (df_map["transparency_score"].notna()) &
                    (df_map["org_country"] != "Unknown")]

    # Parse year inputs
    if start_year != "all":
        try:
            start_year = int(float(start_year))
        except ValueError:
            start_year = None
    else:
        start_year = None

    if end_year is not None:
        try:
            end_year = int(float(end_year))
        except ValueError:
            end_year = None

     # Filter by year range if not "all"   
    if start_year is not None:
        if end_year is not None:
            df_map = df_map[
                (df_map["publication_year"] >= start_year) &
                (df_map["publication_year"] <= end_year)
            ]
        else:
            df_map = df_map[df_map["publication_year"] >= start_year]


    # Clean country info
    df_map["org_country"] = df_map["org_country"].str.split(",")
    df_map = df_map.explode("org_country")
    df_map["org_country"] = df_map["org_country"].str.strip()

    # Group by country
    country_avg = (
        df_map.groupby("org_country")
        .agg(
            avg_score=("transparency_score", "mean"),
            model_count=("model", "count")
        )
        .reset_index()
    )

    # Build map
    geomap = px.choropleth(
        country_avg,
        locations="org_country",
        locationmode="country names",
        color="avg_score",
        hover_name="org_country",
        hover_data={
            "avg_score": ':.2f',
            "model_count": True
        },
        color_continuous_scale=["#EBF7FF", "#D6F0FF","#08DEF9", "#00B6FF", "#178CFF", "#2254F4"],
        labels={
            "avg_score": "Average Transparency Score",
            "org_country": "Developer Country",
            "model_count": "Number of Models"
        },
        title="Average Transparency Score by Developer Country",
    )

    geomap.update_layout(
        geo=dict(showframe=False, showcoastlines=False, projection_type='natural earth'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500, 
        margin=dict(l=10, r=10, t=80, b=10),
        font=dict(size=12, family='Helvetica Neue, Helvetica, Arial, sans-serif'),
        title={
            'x': 0.5,
            'xanchor': 'center',
            'pad':dict(b=5)
        },
        coloraxis_colorbar=dict(
        title="Transparency Score")
    )

    return geomap

# Callback time-series chart

@app.callback(
    Output("time-series-chart", "figure"),
    Input("org-category-dropdown", "value")
)
def update_time_series_chart(selected_category):
    # Filter to recent years and remove 'Unknown' category
    recent_df = df_dashboard[
        (df_dashboard["publication_year"] >= 2015) &
        (df_dashboard["transparency_score"].notna()) &
        (df_dashboard["org_category"] != "Unknown")
    ]

    # Apply category filter
    if selected_category != "All":
        category_df = recent_df[recent_df["org_category"] == selected_category]
    else:
        category_df = recent_df.copy()

    # Group and calculate average score
    grouped_df = (
        category_df.groupby(["publication_year", "org_region"])
        .agg(avg_score=("transparency_score", "mean"), model_count=("model", "count"))
        .reset_index()
    )

    # Apply 3-year rolling average per region
    grouped_df["avg_score_smoothed"] = (
        grouped_df.sort_values(by=["org_region", "publication_year"])
        .groupby("org_region")["avg_score"]
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )

    # Build line chart
    time_series_fig = px.line(
        grouped_df,
        x="publication_year",
        y="avg_score_smoothed",
        color="org_region",
        color_discrete_sequence=region_colors,
        markers=True,
        title=f"Average Transparency Score Over Time<br><sup>{selected_category} Developer Organizations (3-Year Rolling Average)</sup>",
        hover_data={"publication_year": True, "avg_score_smoothed": ':.2f', "model_count": True},
        labels={
            "publication_year": "Publication Year",
            "avg_score_smoothed": "Average Transparency Score",
            "org_region": "Region",
            "model_count": "Number of Models"
        }
    )

    time_series_fig.update_layout(
        plot_bgcolor='white',         
        paper_bgcolor='white',        
        font=dict(size=12, family='Helvetica Neue, Helvetica, Arial, sans-serif'),
        legend=dict(
            orientation="h",       
            yanchor="top",
            y=-0.2,                
            xanchor="center",
            x=0.5,                 
            font=dict(size=12)
        ),
        title={'xanchor':'center', 'x': 0.5,},
        width=700,
        height=500,
        yaxis=dict(range=[-0.1, 4.1]),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    # Lighten gridlines
    time_series_fig.update_xaxes(
        showgrid=False,
        tickmode='array',
        tickvals=list(range(2015, 2026)),  # ensure all years from 2015 to 2025 inclusive
        ticktext=[str(year) for year in range(2015, 2026)]
    )

    time_series_fig.update_yaxes(
        showgrid=True, 
        gridcolor='lightgray'
    )

    # Highlight incomplete year 
    time_series_fig.add_vrect(
        x0=2024.5, x1=2025.5,
        fillcolor="lightgray", opacity=0.3,
        line_width=0,
        annotation=dict(
            text="Partial Year",
            font=dict(size=11),     
            align="left"
        ),
        annotation_position="top left"  
    )   

    return time_series_fig

# Callback transparency components chart

@app.callback(
    Output("component-transparency-chart", "figure"),
    Input("org-category-dropdown", "value")
)
def update_component_transparency_chart(selected_category):
    # Filter to recent years and remove 'Unknown' category
    recent_df = df_dashboard[
        (df_dashboard["publication_year"] >= 2015) &
        (df_dashboard["transparency_score"].notna()) &
        (df_dashboard["org_category"] != "Unknown")
    ]

    if selected_category != "All":
        category_df = recent_df[recent_df["org_category"] == selected_category]
    else:
        category_df = recent_df.copy()

    # Group and compute average % disclosed per year
    field_trends = (
        category_df
        .groupby("publication_year")[transparency_fields]
        .mean()
        .reset_index()
    )

    # Calculate model count
    model_counts = (
    category_df
    .groupby("publication_year")["model"]
    .count()
    .reset_index()
    .rename(columns={"model": "model_count"})
    )

    # Merge model count into field_trends
    field_trends = field_trends.merge(model_counts, on="publication_year")

    for field in transparency_fields:
        field_trends[field] *= 100  # Convert to percentage

    # Melt for long format
    field_trends_long = field_trends.melt(
        id_vars=["publication_year", "model_count"], 
        var_name="component",
        value_name="pct_disclosed"
    )

    # Round to 2 decimal places 
    field_trends_long["pct_disclosed"] = field_trends_long["pct_disclosed"].round(2)

    # Map component labels
    field_trends_long["component"] = field_trends_long["component"].map(field_label_map)


    # Build figure
    component_transparency_fig = px.line(
        field_trends_long,
        x="publication_year",
        y="pct_disclosed",
        color="component",
        color_discrete_map=transparency_color_map,
        markers=True,
        hover_data={
            "publication_year": True,   
            "pct_disclosed": True,
            "component": True,          
            "model_count": True
            },
        labels={
            "publication_year": "Publication Year",
            "pct_disclosed": "% Transparency",
            "component": "Transparency Component",
            "model_count": "Number of Models"
        },
        title=f"Component-Level Transparency Over Time<br><sup>{selected_category} Developer Organizations</sup>"
    )

    component_transparency_fig.update_layout(
        plot_bgcolor='white',         
        paper_bgcolor='white',        
        font=dict(size=12, family='Helvetica Neue, Helvetica, Arial, sans-serif'),
        legend=dict(
            orientation="h",       
            yanchor="top",
            y=-0.2,               
            xanchor="center",
            x=0.5,                
            font=dict(size=12)
        ),
        title={'xanchor':'center', 'x': 0.5,},
        width=700,
        height=500,
        yaxis=dict(range=[0, 101]),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    # Update axes
    # x-axis: remove vertical gridlines and ensure only integer years
    component_transparency_fig.update_xaxes(
        showgrid=False, 
        tickmode='array',
        tickvals=list(range(2015, 2026)),  # ensure all years from 2015 to 2025 inclusive
        ticktext=[str(year) for year in range(2015, 2026)]
    )
  
   # y-axis 
    component_transparency_fig.update_yaxes(
        showgrid=True, 
        gridcolor='lightgray'
    )

    # Highlight incomplete year 
    component_transparency_fig.add_vrect(
        x0=2024.5, x1=2025.5,
        fillcolor="lightgray", opacity=0.3,
        line_width=0,
        annotation=dict(
            text="Partial Year",
            font=dict(size=11),    
            align="left"
        ),
        annotation_position="top left"  
    )   

    return component_transparency_fig


if __name__ == '__main__':
    app.run()