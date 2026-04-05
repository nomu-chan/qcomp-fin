import marimo

__generated_with = "0.22.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import altair as alt
    import os
    csv_filename='./data/quantum_grid_search_results.csv'

    # This cell will refresh every time the file size changes if you 
    # wrap it in a refresh logic, or you can just run it manually.

    def show_results():
        if os.path.exists(csv_filename):
            return pd.read_csv(csv_filename).tail(10)
        return "No data yet."

    show_results()
    return alt, csv_filename, mo, pd


@app.cell
def load_data(csv_filename, mo, pd):

    df = pd.DataFrame()
    try:
        # Standardize path
        df = pd.read_csv(csv_filename)
    except Exception:
        pass

    mo.plain(df.head())
    return (df,)


@app.cell
def ui_controls(df, mo):
    mo.stop(df.empty, mo.md("## ❌ Data not found."))

    def create_log_indexed_slider(column, label):
        unique_vals = sorted(df[column].unique())
        slider = mo.ui.slider(
            start=0,
            stop=len(unique_vals) - 1,
            step=1,
            value=len(unique_vals) - 1,
            label=label
        )
        return slider, unique_vals

    # Create the sliders
    risk_idx, risk_vals = create_log_indexed_slider('lambda_risk', "Risk")
    reward_idx, reward_vals = create_log_indexed_slider('lambda_reward', "Reward")
    card_idx, card_vals = create_log_indexed_slider('lambda_cardinality', "Cardinality")

    k_selector = mo.ui.radio(
        options=[str(x) for x in sorted(df['K_target'].unique())],
        value=str(df['K_target'].min()),
        label="K Target Assets"
    )
    return (
        card_idx,
        card_vals,
        k_selector,
        reward_idx,
        reward_vals,
        risk_idx,
        risk_vals,
    )


@app.cell
def layout(
    card_idx,
    card_vals,
    k_selector,
    mo,
    reward_idx,
    risk_idx,
    risk_vals,
):
    actual_card_val = card_vals[int(card_idx.value)]

    # 4. Styling the UI using the .style() API from the docs
    # We wrap the log slider to show its actual value next to it
    card_ui = mo.hstack([
        card_idx, 
        mo.md(f"**{actual_card_val}**").style(color="blue", font_weight="bold")
    ]).style(
        background="#f8f9fa", 
        padding="10px", 
        border_radius="8px", 
        border="1px solid #dee2e6"
    )

    actual_reward_val = card_vals[int(reward_idx.value)]

    # 4. Styling the UI using the .style() API from the docs
    # We wrap the log slider to show its actual value next to it
    reward_ui = mo.hstack([
        reward_idx, 
        mo.md(f"**{actual_reward_val}**").style(color="blue", font_weight="bold")
    ]).style(
        background="#f8f9fa", 
        padding="10px", 
        border_radius="8px", 
        border="1px solid #dee2e6"
    )


    actual_risk_val = risk_vals[int(risk_idx.value)]

    # 4. Styling the UI using the .style() API from the docs
    # We wrap the log slider to show its actual value next to it
    risk_ui = mo.hstack([
        risk_idx, 
        mo.md(f"**{actual_risk_val}**").style(color="blue", font_weight="bold")
    ]).style(
        background="#f8f9fa", 
        padding="10px", 
        border_radius="8px", 
        border="1px solid #dee2e6"
    )


    mo.vstack([
        mo.md("# 🌌 Quantum Portfolio Explorer"),
        mo.md("### Hyperparameter Configuration"),
        mo.hstack([risk_ui, reward_ui, card_ui, k_selector], justify="start", align="center")
    ])
    return


@app.cell
def filtering(
    card_idx,
    card_vals,
    k_selector,
    mo,
    reward_idx,
    reward_vals,
    risk_idx,
    risk_vals,
):
    def style_slider(slider, values):
            """Helper to wrap slider with its actual value and CSS styling."""
            current_val = values[int(slider.value)]
            return mo.hstack([
                slider,
                mo.md(f"**{current_val}**").style(color="blue", font_weight="bold")
            ]).style(
                background="#f8f9fa",
                padding="12px",
                border_radius="10px",
                border="1px solid #dee2e6",
                margin="4px"
            )

    hparam = mo.vstack([
        mo.md("# 🌌 Quantum Portfolio Explorer"),
        mo.md("### Hyperparameter Configuration"),
        mo.hstack([
            style_slider(risk_idx, risk_vals),
            style_slider(reward_idx, reward_vals),
            style_slider(card_idx, card_vals),
            k_selector.style(padding="12px")
        ], justify="start", align="center", wrap=True)
    ])
    hparam
    return (hparam,)


@app.cell
def display_results(
    card_idx,
    card_vals,
    df,
    k_selector,
    reward_idx,
    reward_vals,
    risk_idx,
    risk_vals,
):
    filtered = df[
            (df['lambda_risk'] == risk_vals[int(risk_idx.value)]) &
            (df['lambda_reward'] == reward_vals[int(reward_idx.value)]) &
            (df['lambda_cardinality'] == card_vals[int(card_idx.value)]) &
            (df['K_target'] == int(k_selector.value))
        ]
    filtered
    return (filtered,)


@app.cell
def _(filtered, mo):
    mo.stop(filtered.empty, mo.md("⚠️ No matching results for these parameters."))

    table = mo.ui.table(filtered, selection="single", pagination=True)
    mo.vstack([
        mo.md("--- \n ### Computed Result Set"),
        table
    ])
    return


@app.cell
def visualization(alt, df, k_selector, mo):
    mo.stop(df.empty)
    parameter_subset = df[df['K_target'] == int(k_selector.value)]

    parameter_chart = alt.Chart(parameter_subset).mark_line(point=True).encode(
        x=alt.X('lambda_cardinality:Q', scale=alt.Scale(type='symlog'), title="Cardinality (Log)"),
        y=alt.Y('active_assets:Q', title="Active Assets"),
        color=alt.Color('lambda_reward:N', title="Reward"),
        tooltip=['sharpe_ratio', 'ruggedness']
    ).properties(width=500, height=300).interactive()

    view_parameter = mo.vstack([
        mo.md("### Parameter Phase Transition"),
        mo.ui.altair_chart(parameter_chart)
    ])

    view_parameter
    return


@app.cell
def pareto_view(alt, df, mo, pd):
    mo.stop(df.empty)

    # 1. MATHEMATICAL PARETO CALCULATION
    # Sort by Volatility (ascending) and Return (descending)
    sorted_df = df.sort_values(['annual_vol', 'expected_return'], ascending=[True, False])

    pareto_front = []
    max_return = -float('inf')
    for _, row in sorted_df.iterrows():
        if row['expected_return'] > max_return:
            pareto_front.append(row)
            max_return = row['expected_return']

    pareto_df = pd.DataFrame(pareto_front)

    # 2. DEFINE EXPLICIT SELECTION (The Fix)
    # This gives marimo a stable signal named 'select' to listen to
    chart_selection = alt.selection_point(name="select", on="click", clear="dblclick")

    # 3. BASE CHART ENCODING
    base = alt.Chart(df).encode(
        x=alt.X('annual_vol:Q', title="Risk (Annual Volatility)", scale=alt.Scale(zero=False)),
        y=alt.Y('expected_return:Q', title="Reward (Expected Return)", scale=alt.Scale(zero=False))
    )

    # All Data Points (Search Space)
    all_points = base.mark_circle(size=45).encode(
        color=alt.Color('sharpe_ratio:Q', scale=alt.Scale(scheme='viridis'), title="Sharpe Ratio"),
        # Dim non-selected points
        opacity=alt.condition(chart_selection, alt.value(0.9), alt.value(0.2)),
        tooltip=['lambda_risk', 'lambda_reward', 'lambda_cardinality', 'sharpe_ratio']
    ).add_params(chart_selection) # Explicitly attach selection here

    # Pareto Line
    frontier_line = alt.Chart(pareto_df).mark_line(color='#e74c3c', strokeWidth=3).encode(
        x='annual_vol:Q',
        y='expected_return:Q'
    )

    # Highlighted Pareto Points
    frontier_points = alt.Chart(pareto_df).mark_point(color='#e74c3c', size=100, filled=True).encode(
        x='annual_vol:Q',
        y='expected_return:Q',
        tooltip=['lambda_risk', 'lambda_reward', 'lambda_cardinality', 'active_assets']
    )

    # 4. FINAL COMPOSITION
    # We omit .interactive() to prevent signal conflicts with mo.ui.altair_chart
    final_chart = (all_points + frontier_line + frontier_points).properties(
        width=500, 
        height=400,
        title="Pareto Front: Optimal Risk vs. Reward Configurations"
    )

    parento_view = mo.vstack([
        mo.md("### 📈 Efficient Frontier Analysis"),
        mo.md("The **red line** is your Pareto Front. Click any point to view its specific parameters."),
        mo.ui.altair_chart(final_chart)
    ])

    parento_view
    return (parento_view,)


@app.cell
def heatmap_view(alt, df, mo):

    mo.stop(df.empty, mo.md("## ❌ Data not found."))

    # Aggregate by mean in case there are multiple samples per grid point
    heatmap_data = df.groupby(['lambda_reward', 'lambda_cardinality'])['sharpe_ratio'].mean().reset_index()

    heatmap_chart = alt.Chart(heatmap_data).mark_rect().encode(
    x=alt.X('lambda_cardinality:N', title='Cardinality Penalty ($\lambda_{card}$)', sort='descending'),
    y=alt.Y('lambda_reward:N', title='Reward Penalty ($\lambda_{rew}$)', sort='descending'),
    color=alt.Color('sharpe_ratio:Q', title='Avg Sharpe Ratio', scale=alt.Scale(scheme='magma')),
    tooltip=['lambda_reward', 'lambda_cardinality', 'sharpe_ratio']
    ).properties(
    width=450, 
    height=300, 
    title="Robustness Grid: Finding the Stable Plateau"
    )

    heatmap_view = mo.vstack([
        mo.md("### 🗺️ Parameter Stability Heatmap"),
        mo.md("Find large, uniform blocks of color. These are regions where your portfolio performance is robust, meaning small changes in the $\lambda$ weights won't crash your strategy."),
        mo.ui.altair_chart(heatmap_chart)
    ])

    heatmap_view
    return (heatmap_view,)


@app.cell
def _(alt, df, mo):
    mo.stop(df.empty)

    # 1. Selection defined once
    card_selection = alt.selection_point(name="card_filter", fields=['lambda_cardinality'])

    # 2. Base Encoding
    sq_base = alt.Chart(df).encode(
        x=alt.X('lambda_cardinality:N', title='Sparsity Penalty (λ_card)'),
        y=alt.Y('active_assets:Q', title='Actual Active Assets'),
        color=alt.Color('K_target:N', title='Target K', scale=alt.Scale(scheme='tableau10')),
        xOffset=alt.XOffset('K_target:N')
    )

    # 3. VISUAL LAYERS (No params added here)
    boxplot_layer = sq_base.mark_boxplot(extent='min-max', size=15)
    points_layer = sq_base.mark_point(size=20, opacity=0.5).encode(
        tooltip=['K_target', 'active_assets', 'sharpe_ratio', 'ruggedness']
    )

    # 4. THE GHOST LAYER (The only one with params)
    # We use a transparent mark so it's invisible but captures the click/selection
    ghost_layer = sq_base.mark_point(opacity=0).add_params(card_selection)

    # 5. Combine and resolve
    # We put the ghost layer on top so it catches the interaction
    squeeze_chart = alt.layer(
        boxplot_layer, 
        points_layer,
        ghost_layer
    ).resolve_scale(
        color='shared',
        x='shared'
    ).properties(
        width=500, 
        height=400,
        title="Multi-Target K Convergence"
    )
    squeeze_view = mo.vstack([
            mo.md("### 📉 Multi-K Constraint Convergence").style(color="#27ae60"),
            mo.md("Each color represents a different **Target K**."),
            mo.ui.altair_chart(squeeze_chart)
    ]).style(border_left="5px solid #27ae60", padding_left="15px")
    squeeze_view
    return (squeeze_view,)


@app.cell
def _(alt, df, mo, pd):

    # 1. Pre-process for Log scale stability
    clean_df = df.dropna(subset=["lambda_cardinality"]).copy()
    clean_df["ruggedness_clipped"] = clean_df["ruggedness"]

    # 2. Define an Interval Selection (The "Brush")
    # This allows you to drag a rectangle over points
    brush = alt.selection_interval(name="brush")

    # 3. BACKGROUND LAYERS (Green Box & Lines)
    success_zone = (
        alt.Chart(
            pd.DataFrame(
                {
                    "ruggedness_clipped": [1e-2],
                    "x2": [100],
                    "sharpe_ratio": [1.0],
                    "y2": [2.0],
                }
            )
        )
        .mark_rect(fill="green", opacity=0.15)
        .encode(x="ruggedness_clipped:Q", x2="x2:Q", y="sharpe_ratio:Q", y2="y2:Q")
    )

    # 4. POINTS LAYER
    # We add the brush selection here
    points = (
        alt.Chart(clean_df)
        .mark_circle(size=70)
        .encode(
            x=alt.X(
                "ruggedness_clipped:Q",
                scale=alt.Scale(type="log", domain=[1e-2, 1e7], clamp=False),
                title="Ruggedness (Log Scale)",
            ),
            y=alt.Y(
                "sharpe_ratio:Q",
                scale=alt.Scale(domain=[-0.6, 2.0]),
                title="Sharpe Ratio",
            ),
            color=alt.condition(
                brush,
                alt.Color(
                    "lambda_cardinality:N", scale=alt.Scale(scheme="viridis")
                ),
                alt.value("lightgray"),
            ),
            tooltip=[
                "lambda_risk",
                "lambda_reward",
                "lambda_cardinality",
                "sharpe_ratio",
                "ruggedness",
            ],
            opacity=alt.condition(brush, alt.value(0.8), alt.value(0.2)),
        )
        .add_params(brush)
    )

    # 5. Build and Wrap
    chart = alt.layer(success_zone, points).properties(
        width=500, height=400, title="Interactive Ruggedness Drill-Down"
    ).interactive()

    # Wrap in Marimo UI to catch the selection
    rugged_chart_ui = mo.ui.altair_chart(chart)

    ruggedness_view = mo.vstack([
                mo.md("### 🌑 Ruggedness:"),
                mo.md("Green Region is a well rugged region."),
                rugged_chart_ui # Displays data selected by your brush
            ])

    ruggedness_view
    return (ruggedness_view,)


@app.cell
def _(hparam):
    hparam
    return


@app.cell
def _(heatmap_view, mo, parento_view, ruggedness_view, squeeze_view):
    mo.ui.tabs({
        "🎯 Efficiency": parento_view,
        "📉 Convergence": squeeze_view,
        "🗺️ Stability": heatmap_view,
        "🌑 Ruggedness": ruggedness_view
    })
    return


@app.cell
def _():
    return


@app.cell
def _():
    # def _():
    #     # Ensure your dataframe has a 'solver_type' column (e.g., 'Quantum' vs 'Classical')
    #     # return mo.stop(df.empty)

    #     clean_df = df.dropna(subset=['lambda_cardinality']).copy()
    #     clean_df['ruggedness_clipped'] = clean_df['ruggedness'].clip(lower=1)

    #     # 1. Selection (Linked across both charts)
    #     brush = alt.selection_interval(name="brush", encodings=['x', 'y'])

    #     # 2. Shared Base
    #     base_compa = alt.Chart(clean_df).mark_circle(size=60, opacity=0.6).encode(
    #         x=alt.X('ruggedness_clipped:Q', 
    #                 scale=alt.Scale(type='log', domain=[1, 1e7]), 
    #                 title='Ruggedness (Log)'),
    #         y=alt.Y('sharpe_ratio:Q', 
    #                 scale=alt.Scale(domain=[-0.6, 1.4]), 
    #                 title='Sharpe Ratio'),
    #         color=alt.condition(brush, 
    #                             alt.Color('lambda_cardinality:N', scale=alt.Scale(scheme='viridis')), 
    #                             alt.value('lightgray')),
    #         tooltip=['solver_type', 'lambda_cardinality', 'sharpe_ratio', 'ruggedness']
    #     ).add_params(brush).properties(width=300, height=300)

    #     # 3. Faceting by Solver
    #     # This creates two charts side-by-side sharing the same Y-axis
    #     comparison_chart = alt.layer(
    #         # Add the Green Box to both
    #         alt.Chart(pd.DataFrame({'x1': [1], 'x2': [100], 'y1': [1.0], 'y2': [1.4]}))
    #            .mark_rect(fill='green', opacity=0.1).encode(x='x1:Q', x2='x2:Q', y='y1:Q', y2='y2:Q'),
    #         base
    #     ).facet(
    #         column=alt.Column('solver_type:N', title="Solver Comparison")
    #     ).resolve_scale(x='shared', y='shared')

    #     chart_ui = mo.ui.altair_chart(comparison_chart)
    #     return chart_ui

    # _()
    return


if __name__ == "__main__":
    app.run()
