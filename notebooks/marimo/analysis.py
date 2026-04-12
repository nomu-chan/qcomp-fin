import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    PATH_DATA = Path("./data/")

    PATH_CLASSICAL_BASELINE = PATH_DATA / "gridsearch_classical_portfolio.csv"
    PATH_CLASSICAL_ANNEALING = PATH_DATA / "gridsearch_quantum_annealing_classical.csv" # Has errors for prob success
    PATH_QUANTUM_ANNEALING = PATH_DATA / "gridsearch_quantum_annealing_quantum.csv"
    PATH_GATEBASED_Y_SIMULATEDBIFRUCATION = PATH_DATA / "gridsearch_quantum_gatebased_y_simulatedbifrucation.csv" # Has Error Runs for bit string interpretation in simulated bifrucation
    PATH_CLASSICAL_MILP = PATH_DATA / "gridsearch_classical_milp.csv"


    df_clb = pd.read_csv(PATH_CLASSICAL_BASELINE)
    df_clu = df_clb[df_clb['method'] == 'CLASSICAL_UNCONSTRAINED'].copy() # Ideal
    df_cld = df_clb[df_clb['method'] == 'CLASSICAL_HEURISTIC'].copy() # Discrete
    df_qqa = pd.read_csv(PATH_QUANTUM_ANNEALING)
    df_qca = pd.read_csv(PATH_CLASSICAL_ANNEALING)
    df_qgb = pd.read_csv(PATH_GATEBASED_Y_SIMULATEDBIFRUCATION)
    df_sb = df_qgb[df_qgb['method'] == 'ANALOG_BASED'].copy()
    df_gate = df_qgb[df_qgb['method'] == 'GATE_BASED'].copy()
    df_clmilp = pd.read_csv(PATH_CLASSICAL_MILP)

    def correct_success_probabilities(df_list, target_df_index=2, epsilon=1e-6):
        """
        df_list: List of dataframes [df_clb, df_qqa, df_qca, df_qgb]
        target_df_index: Index of the Classical Annealing df (df_qca)
        """
        # 1. Combine all data to find the 'True' Ground State for each instance
        # We identify instances by the parameters that define the QUBO matrix
        instance_cols = ['n_assets', 'n_bits', 'lambda_risk', 'lambda_reward', 'lambda_cardinality', 'K_target']

        combined_df = pd.concat(df_list, ignore_index=True)

        # 2. Map each unique problem instance to the best energy ever found across ALL solvers
        # This assumes your dataframes have an 'energy' or 'min_energy' column
        ground_truth = combined_df.groupby(instance_cols)['base_energy'].min().reset_index()
        ground_truth.rename(columns={'base_energy': 'true_ground_state_energy'}, inplace=True)

        # 3. Merge this truth back into your Classical Annealing dataframe
        df_qca_corrected = df_list[target_df_index].copy()
        df_qca_corrected = df_qca_corrected.merge(ground_truth, on=instance_cols, how='left')

        # 4. Recalculate Success Probability
        # If your CSV has 'num_occurrences' of the best state and 'num_reads'
        if 'num_occurrences' in df_qca_corrected.columns and 'num_reads' in df_qca_corrected.columns:
            # Success only counts if the energy found matches the global minimum
            df_qca_corrected['is_optimal'] = (df_qca_corrected['base_energy'] <= df_qca_corrected['true_ground_state_energy'] + epsilon)

            df_qca_corrected['prob_success'] = np.where(
                df_qca_corrected['is_optimal'],
                df_qca_corrected['num_occurrences'] / df_qca_corrected['num_reads'],
                0.0
            )
        else:
            # Fallback: Binary success (Did it find the global optimum at all in this row?)
            df_qca_corrected['prob_success'] = (
                df_qca_corrected['base_energy'] <= df_qca_corrected['true_ground_state_energy'] + epsilon
            ).astype(float)

        return df_qca_corrected

    # Usage
    df_qca = correct_success_probabilities([df_clb, df_qqa, df_qca, df_qgb])
    return (
        df_cld,
        df_clmilp,
        df_clu,
        df_gate,
        df_qca,
        df_qqa,
        df_sb,
        mo,
        np,
        pd,
        plt,
        sns,
    )


@app.cell
def _(df_cld, df_clmilp, df_clu, df_gate, df_qca, df_qqa, df_sb, mo, plt, sns):


    # Assuming 'df' is your loaded and cleaned combined dataframe
    def instance_characterization(df, df_name: str):
        # 1. Filter for unique problem instances to isolate formulation effects
        # Naming convention: K_target -> κ, lambda_cardinality -> λ_card
        instances = df.groupby([
            'n_assets', 'n_bits', 'lambda_risk', 'lambda_reward', 
            'lambda_cardinality', 'K_target'
        ]).first().reset_index()

        fig, axes = plt.subplots(3, 2, figsize=(16, 20))
        fig.suptitle(f"Layer 1: Problem Manifold & Formulation Characterization\nTarget: {df_name}", 
                     fontsize=20, fontweight='bold', y=0.98)
        plt.subplots_adjust(hspace=0.45, wspace=0.35)

        # --- ROW 1: Sparsity & Landscape Topology ---

        # Plot A: Ground-State Sparsity
        sns.histplot(instances['active_assets'], bins=10, ax=axes[0, 0], color='skyblue', kde=True)
        axes[0, 0].set_title(r"Ground-State Sparsity Distribution ($K_{observed}$)", fontsize=14)
        axes[0, 0].set_xlabel("Number of Active Assets")

        # Plot B: Autocorrelation by Precision (Bit Depth)
        sns.boxplot(data=instances, x='n_bits', y='ruggedness', ax=axes[0, 1], palette="Set2")
        axes[0, 1].set_title(r"Landscape Autocorrelation ($\hat{\rho}$) by Binary Precision ($b$)", fontsize=14)
        axes[0, 1].set_ylabel(r"Ruggedness Coefficient ($\hat{\rho}$)")

        # --- ROW 2: Multi-parameter Interactions ---

        # Plot C: Parametric Collinearity
        # Assessing if the Ising mapping (Hamiltonian construction) is biased
        corr_subset = instances[['n_assets', 'n_bits', 'lambda_risk', 'lambda_reward', 'ruggedness']].rename(columns={
            'n_assets': r'$N$ (Assets)',
            'n_bits': r'$b$ (Precision)',
            'lambda_risk': r'$\lambda_{\sigma^2}$ (Risk)',
            'lambda_reward': r'$\lambda_{\mu}$ (Return)',
            'ruggedness': r'$\hat{\rho}$ (Ruggedness)'
        })

        corr = corr_subset.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1, 0], cbar=False)
        axes[1, 0].set_title("QUBO Hyperparameter Multicollinearity", fontsize=14)

        # Plot D: Efficient Frontier Regimes (The "Financial" manifold)
        sns.scatterplot(data=instances, x='lambda_risk', y='expected_return', hue='n_assets', 
                        style='n_bits', palette="viridis", ax=axes[1, 1], alpha=0.7)
        axes[1, 1].set_title(r"Reward Sensitivity ($\mu$) across Risk Regimes ($\lambda_{\sigma^2}$)", fontsize=14)
        axes[1, 1].set_ylabel("Expected Return")
        axes[1, 1].set_xlabel(r"Risk Weighting ($\lambda_{\sigma^2}$)")
        axes[1, 1].legend(title="System Size ($N$)", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

        # --- ROW 3: Combinatorial Complexity ---

        # Plot E: Constraint Tightness (Relative Cardinality)
        # The 'Phase Transition' region is usually around κ ≈ 0.5
        instances['k_ratio'] = instances['K_target'] / instances['n_assets']
        sns.kdeplot(data=instances, x='k_ratio', hue='n_assets', fill=True, common_norm=False, 
                    palette="tab10", ax=axes[2, 0])
        axes[2, 0].set_title(r"Constraint Tightness: Relative Cardinality ($\kappa = K/N$)", fontsize=14)
        axes[2, 0].set_xlabel(r"Relative Capacity ($\kappa$)")

        # Plot F: Topological Scaling
        sns.lineplot(data=instances, x='lambda_risk', y='ruggedness', hue='n_assets', 
                     marker='o', linewidth=2, ax=axes[2, 1], palette="tab10")
        axes[2, 1].set_title(r"Topological Scaling: $\hat{\rho}$ vs. $\lambda_{\sigma^2}$", fontsize=14)
        axes[2, 1].set_ylabel(r"Ruggedness ($\hat{\rho}$)")
        axes[2, 1].set_xlabel(r"Risk Penalty Weight ($\lambda_{\sigma^2}$)")

        # --- FINAL FORMATTING ---
        # Legend pinning and alignment
        axes[2, 1].legend(title="System Size ($N$)", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

        for ax in axes.flat:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout(rect=[0, 0, 0.88, 0.96])
        return fig


    layer1_view = mo.vstack([
    mo.md("## Layer 1: Problem-Characterization EDA")
    , mo.as_html(instance_characterization(df_clu, 'Classical Ideal Baseline'))
    , mo.as_html(instance_characterization(df_cld, 'Classical GreedySearch Discrete Baseline'))
    , mo.as_html(instance_characterization(df_clmilp, 'Classical MILP Discrete Baseline'))
    , mo.as_html(instance_characterization(df_sb, 'Simulated Bifrucation'))
    , mo.as_html(instance_characterization(df_qqa, 'Quantum Annealing'))
    , mo.as_html(instance_characterization(df_qca, 'Classical Annealing'))
    , mo.as_html(instance_characterization(df_gate, 'Gatebased QAOA'))

    ])
    layer1_view
    return


@app.cell
def _(df_cld, df_clmilp, df_clu, df_gate, df_qca, df_qqa, df_sb):
    def export_layer1_summary(df, df_name: str, output_path="notebooks/layer1_summaries"):
        import os
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Unique instance filtering logic consistent with your plots
        instances = df.groupby([
            'n_assets', 'n_bits', 'lambda_risk', 'lambda_reward', 
            'lambda_cardinality', 'K_target'
        ]).first().reset_index()
        instances['k_ratio'] = instances['K_target'] / instances['n_assets']

        filename = f"{output_path}/{df_name.lower().replace(' ', '_')}_summary.txt"

        with open(filename, 'w') as f:
            f.write(f"SUMMARY REPORT: Layer 1 Problem Characterization\n")
            f.write(f"Target Solver: {df_name}\n")
            f.write("="*50 + "\n\n")

            # 1. Dataset Scale
            f.write("--- DATASET SCALE ---\n")
            f.write(f"Total Problem Instances: {len(instances)}\n")
            f.write(f"Range of N (Assets): {instances['n_assets'].min()} - {instances['n_assets'].max()}\n")
            f.write(f"Range of Bit Depth: {instances['n_bits'].min()} - {instances['n_bits'].max()}\n\n")

            # 2. Landscape Ruggedness Analysis
            f.write("--- LANDSCAPE RUGGEDNESS ---\n")
            f.write(f"Mean Ruggedness: {instances['ruggedness'].mean():.4f}\n")
            f.write(f"Max Ruggedness: {instances['ruggedness'].max():.4f}\n")
            f.write(f"Ruggedness Std Dev: {instances['ruggedness'].std():.4f}\n")
            # Extract the correlation between Risk Penalty and Ruggedness
            risk_corr = instances[['lambda_risk', 'ruggedness']].corr().iloc[0, 1]
            f.write(f"Correlation (Lambda_Risk vs Ruggedness): {risk_corr:.4f}\n\n")

            # 3. Constraint & Sparsity
            f.write("--- CONSTRAINT DYNAMICS ---\n")
            f.write(f"Mean K/N Ratio (Tightness): {instances['k_ratio'].mean():.4f}\n")
            f.write(f"Mean Active Assets in Ground State: {instances['active_assets'].mean():.2f}\n\n")

            # 4. Multicollinearity Check
            f.write("--- PARAMETER CORRELATIONS ---\n")
            corr_matrix = instances[['n_assets', 'n_bits', 'lambda_risk', 'lambda_reward', 'ruggedness']].corr()
            f.write(corr_matrix.to_string())
            f.write("\n\n" + "="*50 + "\n")
            f.write("End of Report")

        print(f"Summary exported to {filename}")

    # Execute exports
    solvers = [
        (df_clu, 'Classical Ideal Baseline'),
        (df_cld, 'Classical GreedySearch Discrete Baseline'),
        (df_clmilp, 'Classical MILP Discrete Baseline'),
        (df_sb, 'Simulated Bifurcation'),
        (df_qqa, 'Quantum Annealing'),
        (df_qca, 'Classical Annealing'),
        (df_gate, 'Gatebased QAOA')
    ]

    for df, name in solvers:
        export_layer1_summary(df, name)
    return


@app.cell
def _(
    df_cld,
    df_clmilp,
    df_clu,
    df_gate,
    df_qca,
    df_qqa,
    df_sb,
    mo,
    pd,
    plt,
    sns,
):
    from scipy.stats import kruskal, spearmanr

    def solver_performance_analysis(df_list, names):
        # 1. Combine all data for cross-solver comparison
        combined_df = pd.concat([d.assign(method=n) for d, n in zip(df_list, names)])
        unique_methods = combined_df['method'].unique()
        colors = sns.color_palette("husl", len(unique_methods))
        palette_dict = dict(zip(unique_methods, colors))
        # 2. Statistical Test: Kruskal-Wallis H-test for Ruggedness Perception
        # Do different solvers see significantly different landscapes?
        ruggedness_groups = [d['ruggedness'].values for d in df_list]
        h_stat, p_val = kruskal(*ruggedness_groups)

        combined_df['ruggedness_decile'] = pd.qcut(combined_df['ruggedness'], 10, labels=False, duplicates='drop')

        fig, axes = plt.subplots(3, 2, figsize=(20, 26)) # Slightly wider for the legend
        fig.suptitle(r"Layer 2: Solver Fidelity & Statistical Validation", fontsize=24, fontweight='bold', y=.98)
        plt.subplots_adjust(hspace=0.4, wspace=0.3)

        # --- ROW 1: The "Finance" Gap ---

        # Plot A: Sharpe Ratio by Solver
        sns.boxplot(data=combined_df, x='method', y='sharpe_ratio', hue='method',  palette=palette_dict, ax=axes[0, 0])
        axes[0, 0].set_title("Objective Function Distribution: Sharpe Ratio ($S$)", fontsize=14)
        axes[0, 0].set_ylabel("Sharpe Ratio")
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Plot B: Cardinality Violation (Constraint Fidelity)
        combined_df['k_error'] = (combined_df['active_assets'] - combined_df['K_target']).abs()
        sns.barplot(data=combined_df, x='method', y='k_error', hue='method', palette=palette_dict, ax=axes[0, 1])
        axes[0, 1].set_title(r"Mean Constraint Violation ($\epsilon_K = |K_{observed} - K_{target}|)$", fontsize=14)
        axes[0, 1].set_ylabel("Cardinality Error")
        axes[0, 1].tick_params(axis='x', rotation=45)

        # --- ROW 2: The "Energy" Perception (Layer 2 Core) ---

        # Plot C: Perceived Ruggedness (The H-test Visual)
        sns.violinplot(data=combined_df, x='method', y='ruggedness', ax=axes[1, 0], cut=0, hue='method',palette=palette_dict)
        axes[1, 0].set_title(f"Landscape Autocorrelation ($\hat{{\\rho}}$) Perception\n(H-Stat: {h_stat:.2f}, p: {p_val:.4f})", fontsize=14)
        axes[1, 0].set_ylabel(r"Ruggedness Coefficient ($\hat{\rho}$)")
        axes[1, 0].set_yscale('log')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # --- ROW 2, PLOT D: FORMAL PARETO FRONTIER ---
        ax_pareto = axes[1, 1]

        # 2. Plot the scatter cloud using the palette dictionary
        sns.scatterplot(
            data=combined_df, 
            x='time_to_solve', 
            y='sharpe_ratio', 
            hue='method', 
            palette=palette_dict,  # FIXED
            alpha=0.1, 
            edgecolor=None, 
            ax=ax_pareto, 
            legend=False
        )

        # 3. Plot the frontier lines using the same palette dictionary
        for name in names:
            m_data = combined_df[combined_df['method'] == name].copy()
            if not m_data.empty:
                m_data = m_data.sort_values('time_to_solve')
                m_data['frontier_sharpe'] = m_data['sharpe_ratio'].cummax()

                sns.lineplot(
                    data=m_data, 
                    x='time_to_solve', 
                    y='frontier_sharpe', 
                    color=palette_dict[name],  # FIXED: Explicitly pull from the map
                    ax=ax_pareto, 
                    linewidth=3, 
                    drawstyle='steps-post',
                    label=name
                )

        # Final Legend and Axis cleanup
        ax_pareto.set_xscale('log')
        ax_pareto.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        # --- ROW 3: Spearman Validation (Thesis-Grade Fidelity) ---

        # Plot E: Spearman Correlation Heatmap (Energy vs Sharpe)
        # We filter for feasible solutions (k_error == 0) to validate the QUBO
        feasible_df = combined_df[combined_df['k_error'] == 0]
        corrs = []
        for name in names:
            m_data = feasible_df[feasible_df['method'] == name]
            if len(m_data) > 2:
                rho, _ = spearmanr(m_data['base_energy'], m_data['sharpe_ratio'])
                corrs.append({'method': name, 'spearman_rho': rho})

        corr_df = pd.DataFrame(corrs)
        sns.barplot(data=corr_df, x='method', y='spearman_rho', palette=palette_dict, hue='method', ax=axes[2, 0])
        axes[2, 0].set_title(r"QUBO Hamiltonian Fidelity: $r_s(H(\mathbf{x}), S)$", fontsize=14)
        axes[2, 0].set_ylabel("Spearman Rank Correlation")
        axes[2, 0].axhline(0, color='black', linewidth=0.8)
        axes[2, 0].tick_params(axis='x', rotation=45)

        # --- ROW 3, PLOT F: CLEANED RESILIENCE ---
        # We use binned ruggedness on the X-axis for a smoother curve
        resilience_ax = axes[2, 1]
        sns.lineplot(
            data=combined_df, 
            x='ruggedness_decile', # Use the binned deciles instead of raw floats
            y='sharpe_ratio', 
            hue='method', 
            ax=resilience_ax,
            estimator='mean',      
            errorbar=('ci', 95),   # Confidence bands will now be smooth
            linewidth=3.5,
            marker='o',            # Adds points to show the trend clearly
            palette=palette_dict
        )

        resilience_ax.set_title(r"Solver Resilience to Landscape Complexity Cohorts", fontsize=16, pad=15)
        resilience_ax.set_xlabel(r"Landscape Ruggedness Decile ($\hat{\rho}_{10}$)", labelpad=12)
        resilience_ax.set_ylabel("Mean Sharpe Ratio ($S$)")
        resilience_ax.set_xticks(range(10)) # Ensure 0-9 deciles are marked
        resilience_ax.grid(True, axis='y', ls="--", alpha=0.3)

        # --- THE LEGEND FIX: PINNED TO FAR RIGHT ---
        # We move the legend totally outside the plot area
        resilience_ax.legend(
            title=r"$\bf{Solver\ Hierarchy}$", 
            bbox_to_anchor=(1.05, 1), 
            loc='upper left', 
            borderaxespad=0.,
            frameon=True,
            shadow=True,
            fontsize=12
        )

        # --- THE LABEL ALIGNMENT FIX ---
        for ax in axes.flat:
            # Tick alignment: ha='right' + rotation_mode='anchor' removes the gap
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            ax.tick_params(axis='both', which='major', labelsize=11)

        # Increase the right margin (0.80) to ensure the legend isn't cut off
        plt.tight_layout(rect=[0, 0, 0.82, 0.96]) 
        return fig

    CL_IDEAL_BASELINE = 'Efficient Frontier'
    CL_GREEDY_BASELINE = 'Greedy Discrete Heuristic'
    CL_MILP_BASELINE = 'MIQP Discrete Global Optimum'
    CL_SIMBIF = 'Simulated Bifurcation (SB)'
    QM_ANNEALING_QUANTUM = 'Simulated Quantum Annealing (SQA)'
    QM_ANNEALING_CLASSICAL = 'Simulated Annealing (SA)'
    QM_GATEBASED_QAOA = 'QAOA (Transverse-Field Mixer)'

    # Execution block
    dfs = [df_clu, df_cld, df_clmilp, df_sb, df_qqa, df_qca, df_gate]
    names = [CL_IDEAL_BASELINE, CL_GREEDY_BASELINE, CL_MILP_BASELINE, CL_SIMBIF, QM_ANNEALING_QUANTUM, QM_ANNEALING_CLASSICAL, QM_GATEBASED_QAOA]

    layer2_view = mo.vstack([
        mo.md("## Layer 2: Solver Outcome & Statistical Validation"),
        mo.as_html(solver_performance_analysis(dfs, names))
    ])
    layer2_view
    return CL_IDEAL_BASELINE, CL_MILP_BASELINE, dfs, names


@app.cell
def _(dfs, names):
    def export_layer2_summary(df_list, names, output_path="notebooks/summaries"):
        import os
        import pandas as pd
        from scipy.stats import kruskal, spearmanr

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # 1. Internal Combine for Stat Processing
        combined_df = pd.concat([d.assign(method=n) for d, n in zip(df_list, names)])
        filename = f"{output_path}/layer2_solver_fidelity_report.txt"

        # 2. Statistical Test: Do solvers see the same landscape?
        ruggedness_groups = [d['ruggedness'].dropna().values for d in df_list]
        h_stat, p_val = kruskal(*ruggedness_groups)

        with open(filename, 'w') as f:
            f.write("LAYER 2 REPORT: Solver Outcome & Statistical Fidelity\n")
            f.write("="*70 + "\n\n")

            # Section 1: Landscape Perception (The H-Test)
            f.write("--- SECTION 1: TOPOLOGICAL PERCEPTION (Kruskal-Wallis) ---\n")
            f.write(f"H-Statistic: {h_stat:.4f}\n")
            f.write(f"p-value:     {p_val:.4e}\n")
            f.write(f"Finding:     {'STATISTICALLY SIGNIFICANT DIVERGENCE' if p_val < 0.05 else 'NO SIGNIFICANT DIVERGENCE'}\n")
            f.write("Note: High H-stat confirms that solver paradigms perceive unique ruggedness\n")
            f.write("profiles for identical problem instances.\n\n")

            # Section 2: Performance & Constraint Fidelity
            f.write("--- SECTION 2: SOLVER PERFORMANCE & CONSTRAINT FIDELITY ---\n")
            f.write(f"{'Method':<25} | {'Med Sharpe':<10} | {'Mean K-Err':<10} | {'Spearman Rho'} \n")
            f.write("-" * 70 + "\n")

            for name in names:
                m_df = combined_df[combined_df['method'] == name]

                # Median Sharpe
                med_sharpe = m_df['sharpe_ratio'].median()

                # Cardinality Error (|Active - K|)
                k_err = (m_df['active_assets'] - m_df['K_target']).abs().mean()

                # Spearman Rho (Fidelity: Energy vs Sharpe)
                # We only look at 'Valid' solutions (k_error < 0.5) to test the QUBO logic
                feasible = m_df[(m_df['active_assets'] - m_df['K_target']).abs() < 0.5]
                if len(feasible) > 5:
                    rho, _ = spearmanr(feasible['base_energy'], feasible['sharpe_ratio'])
                    rho_str = f"{rho:.4f}"
                else:
                    rho_str = "N/A (Infeasible)"

                f.write(f"{name:<25} | {med_sharpe:<10.4f} | {k_err:<10.4f} | {rho_str}\n")

            # Section 3: Efficiency Frontier (Time)
            f.write("\n--- SECTION 3: COMPUTATIONAL EFFICIENCY ---\n")
            f.write(f"{'Method':<25} | {'Avg Time (s)':<12} | {'Max Time (s)'}\n")
            f.write("-" * 60 + "\n")
            for name in names:
                m_df = combined_df[combined_df['method'] == name]
                avg_t = m_df['time_to_solve'].mean()
                max_t = m_df['time_to_solve'].max()
                f.write(f"{name:<25} | {avg_t:<12.5f} | {max_t:.5f}\n")

            f.write("\n" + "="*70 + "\n")
            f.write("End of Layer 2 Report")

        print(f"Layer 2 Summary successfully exported to: {filename}")

    # Execution
    export_layer2_summary(dfs, names)
    return


@app.cell
def _(dfs, names, pd):

    # Combine using a list comprehension to inject the 'method' column on the fly
    combined_df = pd.concat(
        [d.assign(method=name) for d, name in zip(dfs, names)], 
        ignore_index=True
    )

    # Now you can easily filter or group
    print(combined_df['method'].value_counts())
    return (combined_df,)


@app.cell
def _(
    CL_IDEAL_BASELINE,
    CL_MILP_BASELINE,
    combined_df,
    mo,
    names,
    pd,
    plt,
    sns,
):
    from scipy import stats
    def crossover_gap_analysis(df, baseline_col=CL_MILP_BASELINE):
        df = df.copy()
        # Handle deciles safely
        df['rug_decile'] = pd.qcut(df['ruggedness'], 10, labels=False, duplicates='drop')

        # FIX: Identify unique solvers that aren't the baseline or the ideal unconstrained
        # This ensures that when the subset only has [Solver, Baseline], 'solvers' becomes [Solver]
        solvers = [c for c in df['method'].unique() if c not in [CL_IDEAL_BASELINE, baseline_col]]
        num_solvers = len(solvers)
        results = []
        for decile in sorted(df['rug_decile'].unique()):
            for n in sorted(df['n_assets'].unique()):
                subset = df[(df['rug_decile'] == decile) & (df['n_assets'] == n)]
                baseline_median = subset[subset['method'] == baseline_col]['sharpe_ratio'].median()

                for solver in solvers:
                    solver_data = subset[subset['method'] == solver]['sharpe_ratio']
                    if len(solver_data) > 3:
                        # LCB - Baseline Median
                        lcb = solver_data.mean() - (1.96 * stats.sem(solver_data))
                        gap = lcb - baseline_median

                        results.append({
                            'rug_decile': decile,
                            'n_assets': n,
                            'method': solver,
                            'advantage_floor': gap,
                            'advantage': 1 if gap > 0 else 0  # Re-added for logic compatibility
                        })

        gap_df = pd.DataFrame(results)

        # Visualization
        num_solvers = len(solvers)
        fig, axes = plt.subplots(1, num_solvers, figsize=(7 * num_solvers, 6))
        if num_solvers == 1: axes = [axes]

        for i, solver in enumerate(solvers):
            # 1. Pivot the summarized results, NOT the raw dataframe
            pivot = gap_df[gap_df['method'] == solver].pivot_table(
                index='n_assets', 
                columns='rug_decile', 
                values='advantage_floor'
            )

            # REMOVE OR COMMENT OUT THIS LINE:
            # pivot = df.pivot(index='lambda_risk', columns='n_assets', values='sharpe_ratio')

            if pivot.empty or pivot.isna().all().all():
                print(f"⚠️ Warning: No crossover data for {solver} against {baseline_col}.")
                continue

            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0, ax=axes[i])
            axes[i].set_title(f"{solver} vs. MIQP\nAdvantage Floor ($\Delta_{{LCB}}$)")
            axes[i].set_xlabel(r"Landscape Ruggedness Decile ($\hat{\rho}$)")
            axes[i].set_ylabel(r"Asset Universe Size ($N$)")

        return fig, gap_df

    # To find the exact threshold using Segmented Regression logic:
    def estimate_threshold(boundary_df, solver_name):
        # Filter for the specific solver
        s_df = boundary_df[boundary_df['method'] == solver_name].groupby('rug_decile')['advantage'].mean()
        # Identify the first decile where win probability > 0.05
        transition_decile = s_df[s_df > 0.05].index.min()
        return transition_decile
    def generate_method_specific_crossovers(df, names):
        """
        Generates method-by-method crossover heatmaps against the MILP baseline.
        Satisfies Point 8 (Benchmark Quality) and Point 12 (Spacing).
        """
        baseline_col = CL_MILP_BASELINE 

        # Identify solvers to compare against the MILP baseline
        # Excludes the baseline itself and the Ideal (Unconstrained) baseline
        solvers = [n for n in names if n not in [CL_IDEAL_BASELINE, baseline_col]]
        method_views = []

        for solver in solvers:
            # 1. Isolate the comparison pair
            subset_df = df[df['method'].isin([solver, baseline_col])]

            # Guard clause: Check if baseline data exists for these assets/bits
            if subset_df[subset_df['method'] == baseline_col].empty:
                continue

            # 2. Perform the Crossover Gap Analysis
            # Returns (Matplotlib Figure, DataFrame with 'advantage' and 'advantage_floor')
            fig, boundary_data = crossover_gap_analysis(subset_df, baseline_col=baseline_col)

            # 3. Analyze the results for narrative text
            if boundary_data is not None and not boundary_data.empty:
                # We look for the 'advantage' flag (LCB > Median)
                transition_row = boundary_data[boundary_data['advantage'] == 1]

                if not transition_row.empty:
                    # Find the first decile where quantum/heuristic starts winning
                    transition_decile = transition_row['rug_decile'].min()
                    max_gap = transition_row['advantage_floor'].max()
                    transition_text = (
                        f"✅ **Statistical Crossover Found:** Significance threshold reached at "
                        f"Ruggedness Decile {transition_decile}. (Max LCB Floor: +{max_gap:.4f})"
                    )
                else:
                    # Calculate how close it got to the baseline
                    closest = boundary_data['advantage_floor'].max()
                    transition_text = (
                        f"❌ **No Crossover:** Classical dominance persists across all deciles. "
                        f"(Best case gap: {closest:.4f})"
                    )
            else:
                transition_text = "⚠️ **Insufficient Data:** Metadata suggests zero overlapping parameter points."

            # 4. Append to Marimo view with Layout Fixes (Point 12)
            method_views.append(mo.md(f"### {solver} vs. {baseline_col}"))
            method_views.append(mo.md(transition_text))

            if fig is not None:
                # mo.as_html converts the plt figure to a static element
                method_views.append(mo.as_html(fig))
                # CRITICAL: Close the figure to free memory and prevent extra vertical space
                plt.close(fig) 

        # Vertical stack for clean document flow
        return mo.vstack(method_views)

    # Final Execution
    layer3_view = mo.vstack([
        mo.md("## Layer 3: Regime & Crossover Analysis"),
        generate_method_specific_crossovers(combined_df, names)
    ])

    layer3_view
    return


@app.cell
def _(CL_IDEAL_BASELINE, CL_MILP_BASELINE, combined_df, names):
    def export_layer3_operational_report(df, names, output_path="notebooks/summaries"):
        import os
        import pandas as pd
        from scipy import stats

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        baseline_col = CL_MILP_BASELINE
        solvers = [n for n in names if n not in [CL_IDEAL_BASELINE, baseline_col]]
        filename = f"{output_path}/layer3_operational_crossover.txt"

        df = df.copy()
        df['rug_decile'] = pd.qcut(df['ruggedness'], 10, labels=False, duplicates='drop')

        with open(filename, 'w') as f:
            f.write("LAYER 3: REGIME & OPERATIONAL CROSSOVER REPORT\n")
            f.write("Metric: Advantage Floor (95% LCB - Baseline Median)\n")
            f.write("="*75 + "\n\n")

            for solver in solvers:
                f.write(f"--- SOLVER: {solver} ---\n")

                # Filter and calculate metrics
                subset_df = df[df['method'].isin([solver, baseline_col])]
                results = []

                for decile in sorted(df['rug_decile'].unique()):
                    for n in sorted(df['n_assets'].unique()):
                        sub = subset_df[(subset_df['rug_decile'] == decile) & (subset_df['n_assets'] == n)]
                        b_med = sub[sub['method'] == baseline_col]['sharpe_ratio'].median()
                        s_data = sub[sub['method'] == solver]['sharpe_ratio']

                        if len(s_data) > 3:
                            lcb = s_data.mean() - (1.96 * stats.sem(s_data))
                            gap = lcb - b_med
                            results.append({'decile': decile, 'n': n, 'gap': gap})

                res_df = pd.DataFrame(results)

                # Reporting Logic
                transition = res_df[res_df['gap'] > 0]
                if not transition.empty:
                    first_t = transition.iloc[0]
                    f.write(f"STATUS: Operational Crossover REACHED\n")
                    f.write(f"EARLIEST TRANSITION: Decile {first_t['decile']} at N={first_t['n']}\n")
                    f.write(f"MAXIMUM ADVANTAGE: +{res_df['gap'].max():.4f}\n")
                else:
                    f.write(f"STATUS: No Operational Crossover Found\n")
                    f.write(f"CLOSEST APPROACH: {res_df['gap'].max():.4f}\n")

                # Small Data Table
                f.write("\nDetailed Advantage Floor (LCB - Baseline Median):\n")
                pivot = res_df.pivot(index='n', columns='decile', values='gap')
                f.write(pivot.to_string())
                f.write("\n" + "-"*75 + "\n\n")

            f.write("End of Layer 3 Report")

        print(f"Operational report exported to {filename}")

    # Execute the export
    export_layer3_operational_report(combined_df, names)
    return


@app.cell
def _(combined_df, mo, np, pd, plt, sns):
    import statsmodels.formula.api as smf
    from scipy.stats import friedmanchisquare, wilcoxon
    import scikit_posthocs as sp # For Nemenyi test
    from lifelines import KaplanMeierFitter
    import statsmodels.formula.api as smf

    def layer4_scaling_and_stats(df):
        df = df.copy()

        # 1. Create instance_id if it doesn't exist
        # This groups rows that represent the same problem instance
        if 'instance_id' not in df.columns:
            # Create a unique string based on the problem parameters
            df['instance_id'] = df.apply(lambda row: f"{row['n_assets']}_{row['n_bits']}_{row['ruggedness']:.6f}", axis=1)

        # 2. Regression with Interactions
        df['log_ruggedness'] = np.log1p(df['ruggedness'])
        # Clean method names for patsy formula (removes spaces/dashes)
        df['method_clean'] = df['method'].str.replace('[^a-zA-Z0-9]', '_', regex=True)

        formula = "sharpe_ratio ~ method_clean * log_ruggedness + method_clean * n_assets"
        model = smf.ols(formula, data=df).fit()

        # 3. Scaling & Stats Visualization
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # Log-Log Runtime Scaling
        solvers = df['method'].unique()
        scaling_results = []
        for solver in solvers:
            s_df = df[(df['method'] == solver) & (df['time_to_solve'] > 0)].copy()
            if len(s_df) > 5:
                log_n = np.log2(s_df['n_assets'])
                log_t = np.log2(s_df['time_to_solve'])
                slope, intercept = np.polyfit(log_n, log_t, 1)
                scaling_results.append({'method': solver, 'exponent': slope})
                sns.regplot(x=log_n, y=log_t, label=f"{solver} (b={slope:.2f})", ax=axes[0], scatter_kws={'alpha':0.3})

        axes[0].set_title("Runtime scaling: $T \propto N^b$", fontsize=14)
        axes[0].set_xlabel("$\log_2(N)$ Assets")
        axes[0].set_ylabel("$\log_2(Time)$")
        axes[0].legend()

        # 4. Friedman & Nemenyi (Matched Instance Test)
        # Pivot: rows are instances, columns are solvers
        pivot_df = df.pivot_table(index='instance_id', columns='method', values='sharpe_ratio').dropna()

        if not pivot_df.empty and pivot_df.shape[1] > 1:
            # Friedman test checks if the solvers are different globally
            stat, p_friedman = friedmanchisquare(*[pivot_df[col] for col in pivot_df.columns])

            # Nemenyi post-hoc shows which specific pairs differ
            nemenyi_matrix = sp.posthoc_nemenyi_friedman(pivot_df)

            sns.heatmap(nemenyi_matrix, annot=True, cmap='RdYlGn', center=0.05, ax=axes[1])
            axes[1].set_title(f"Nemenyi Pairwise p-values\n(Friedman p={p_friedman:.2e})", fontsize=14)
        else:
            axes[1].text(0.5, 0.5, "Insufficient Matched Data for Friedman Test", ha='center')

        return model, pd.DataFrame(scaling_results), fig

    # Execute
    ols_model, scaling_exponents, scaling_fig = layer4_scaling_and_stats(combined_df)

    # 1. Capture the OLS table as a dataframe for clean display
    summary_df = pd.DataFrame(ols_model.summary().tables[1].data[1:], 
                             columns=ols_model.summary().tables[1].data[0])

    # 2. Create the View
    layer4_view = mo.vstack([
        mo.md("## Layer 4: Rigorous Scaling & Interaction Analysis"),

        mo.md("### A. Algorithmic Scaling and Statistical Significance"),
        mo.as_html(scaling_fig),

        mo.md("### B. Regression Analysis (Interaction Effects)"),
        mo.md("""
        This model evaluates **Sharpe Ratio** as a function of **Method**, **Ruggedness**, and **N Assets**. 
        * Significant $P > |t|$ values (< 0.05) in the interaction terms indicate where a solver's performance scales differently than the baseline.
        """),
        mo.ui.table(summary_df),

        mo.md("### C. Scaling Exponents ($T \propto N^b$)"),
        mo.ui.table(scaling_exponents.sort_values(by='exponent')),

    ])

    layer4_view
    return (
        KaplanMeierFitter,
        ols_model,
        scaling_exponents,
        summary_df,
        wilcoxon,
    )


@app.cell
def _(
    CL_MILP_BASELINE,
    KaplanMeierFitter,
    combined_df,
    mo,
    np,
    pd,
    plt,
    wilcoxon,
):

    def layer4_complete_battery(df, baseline_name=CL_MILP_BASELINE):
        df = df.copy()

        # 1. Synthesize instance_id if missing (CRITICAL for pivoting)
        if 'instance_id' not in df.columns:
            df['instance_id'] = df.apply(lambda row: f"{row['n_assets']}_{row['n_bits']}_{row['ruggedness']:.6f}", axis=1)

        # 2. Bootstrap Confidence Intervals (Headline Metrics)
        def get_bootstrap_ci(data):
            data = data.dropna()
            if len(data) < 5: return (np.nan, np.nan)
            samples = [np.mean(np.random.choice(data, len(data))) for _ in range(1000)]
            return np.percentile(samples, [2.5, 97.5])

        ci_data = []
        for solver in df['method'].unique():
            sub = df[df['method'] == solver]
            sharpe_ci = get_bootstrap_ci(sub['sharpe_ratio'])
            ci_data.append({
                'method': solver, 
                'Sharpe_Mean_LCB': sharpe_ci[0], 
                'Sharpe_Mean_UCB': sharpe_ci[1]
            })

        # 3. Paired Wilcoxon Tests & Effect Sizes (vs Baseline)
        paired_stats = []
        for solver in [s for s in df['method'].unique() if s != baseline_name]:
            # Pivot specifically for the pair to maximize data usage
            pivot = df[df['method'].isin([solver, baseline_name])].pivot_table(
                index='instance_id', columns='method', values='sharpe_ratio').dropna()

            if len(pivot) > 5:
                # Check if there is actual variance in the difference
                diff = pivot[solver] - pivot[baseline_name]
                if not (diff == 0).all():
                    stat, p = wilcoxon(pivot[solver], pivot[baseline_name])
                    n = len(pivot)
                    # Effect size r = Z / sqrt(N)
                    # Simple Z-score approximation
                    z_stat = (stat - (n*(n+1)/4)) / np.sqrt(n*(n+1)*(2*n+1)/24)
                    effect_size = abs(z_stat) / np.sqrt(n)

                    paired_stats.append({
                        'solver': solver, 
                        'p_value': p, 
                        'effect_size_r': effect_size,
                        'n_pairs': n
                    })

        # 4. Survival Analysis (Time-to-Solution)
        # Define "Success" as reaching the 75th percentile of baseline performance
        threshold = df[df['method'] == baseline_name]['sharpe_ratio'].quantile(0.75)
        kmf_fig, ax = plt.subplots(figsize=(10, 6))

        for solver in df['method'].unique():
            s_df = df[df['method'] == solver].copy()
            # Event = 1 if solver reached baseline quality, 0 if it didn't (censored)
            s_df['event'] = (s_df['sharpe_ratio'] >= threshold).astype(int)

            kmf = KaplanMeierFitter()
            # Use time_to_solve; if missing, use a max penalty time
            times = s_df['time_to_solve'].fillna(s_df['time_to_solve'].max() * 2)

            kmf.fit(times, event_observed=s_df['event'], label=solver)
            kmf.plot_survival_function(ax=ax)

        ax.set_title(f"TTS Survival: Time to match {baseline_name} 75th percentile")
        ax.set_ylabel("Probability of NOT yet reaching target quality")
        ax.set_xlabel("Runtime (s)")
        plt.grid(True, which="both", ls="-", alpha=0.5)

        return pd.DataFrame(ci_data), pd.DataFrame(paired_stats), kmf_fig

    # Execute corrected battery
    boot_ci, wilcoxon_results, survival_plot = layer4_complete_battery(combined_df)

    # Assemble final view
    layer4_1_view = mo.vstack([
        mo.md("## Layer 4: Rigorous Statistical Validation"),

        mo.md("### A. Survival Analysis (Time-to-Solution)"),
        mo.md("This plot shows the probability that a solver has *not* reached the target quality over time. A faster decay (steeper curve) indicates a more efficient solver."),
        mo.as_html(survival_plot),

        mo.md("### B. Paired Non-Parametric Tests (vs. Baseline)"),
        mo.md("Wilcoxon signed-rank tests with **Effect Size (r)**. r > 0.5 is large, 0.3 is medium, 0.1 is small."),
        mo.ui.table(wilcoxon_results),

        mo.md("### C. Bootstrap 95% Confidence Intervals"),
        mo.ui.table(boot_ci),


    ])

    layer4_1_view
    return boot_ci, layer4_1_view, wilcoxon_results


@app.cell
def _(
    KaplanMeierFitter,
    boot_ci,
    combined_df,
    layer4_1_view,
    mo,
    ols_model,
    plt,
    scaling_exponents,
    summary_df,
    wilcoxon_results,
):
    def layer4_survival_loglog(df, baseline_name='Discrete Baseline'):
        df = df.copy()
        threshold = df[df['method'] == baseline_name]['sharpe_ratio'].quantile(0.75)

        fig, ax = plt.subplots(figsize=(10, 6))

        for solver in df['method'].unique():
            s_df = df[df['method'] == solver].copy()
            s_df['event'] = (s_df['sharpe_ratio'] >= threshold).astype(int)

            kmf = KaplanMeierFitter()
            # Add epsilon to avoid log(0) and handle NaNs with a penalty
            max_time = df['time_to_solve'].max()
            times = s_df['time_to_solve'].fillna(max_time * 2) + 1e-9

            kmf.fit(times, event_observed=s_df['event'], label=solver)

            # Plotting the survival function (Probability of NOT matching quality)
            ax.step(kmf.survival_function_.index, kmf.survival_function_.iloc[:, 0], label=solver, where='post')

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_title(f"Log-Log TTS Survival: Quality Match Scaling")
        ax.set_ylabel("log P(Not yet reaching target quality)")
        ax.set_xlabel("log Runtime (s)")
        ax.grid(True, which="both", ls="-", alpha=0.3)
        ax.legend()

        return fig

    # Generate the log-log plot
    survival_loglog_plot = layer4_survival_loglog(combined_df)

    # Final Consolidated View
    layer4_final_view = mo.vstack([
        layer4_1_view, # Includes your existing sections A, B, and C

        mo.md("### D. Log-Log Survival Scaling"),
        mo.md("""
        This log-log transformation of the survival curve is used to identify **Power-Law scaling**. 
        * A horizontal line at $10^0$ (1.0) indicates the solver **never** reached the target quality within the observed window.
        * Stiff vertical drops or steep slopes identify the exact runtime where the solver becomes operationally competitive.
        """),
        mo.as_html(survival_loglog_plot),

        mo.md("### E. Interaction Regression Table"),
        mo.ui.table(summary_df)
    ])

    # Define the output filename
    export_filename = "notebooks/summaries/layer4_statistical_report.txt"

    with open(export_filename, "w") as f:
        f.write("====================================================\n")
        f.write("      LAYER 4: RIGOROUS STATISTICAL REPORT          \n")
        f.write("====================================================\n\n")

        # 1. OLS Regression Summary
        f.write("--- 1. INTERACTION REGRESSION SUMMARY ---\n")
        f.write(str(ols_model.summary()))
        f.write("\n\n")

        # 2. Paired Wilcoxon Results
        f.write("--- 2. PAIRED WILCOXON TESTS (vs. Baseline) ---\n")
        if not wilcoxon_results.empty:
            f.write(wilcoxon_results.to_string(index=False))
        else:
            f.write("No paired data available for testing.")
        f.write("\n\n")

        # 3. Bootstrap Confidence Intervals
        f.write("--- 3. BOOTSTRAP 95% CONFIDENCE INTERVALS ---\n")
        f.write(boot_ci.to_string(index=False))
        f.write("\n\n")

        # 4. Scaling Exponents
        f.write("--- 4. ALGORITHMIC SCALING EXPONENTS ---\n")
        f.write(scaling_exponents.sort_values(by='exponent').to_string(index=False))
        f.write("\n\n")

        f.write("--- END OF REPORT ---")

    print(f"Successfully exported report to: {export_filename}")

    layer4_final_view
    return


@app.cell
def _():
    import pathlib

    def concat_all_reports(output_path="notebooks/final_benchmark_appendix.txt"):
        # Define the directories based on your ls -R output
        l1_dir = pathlib.Path("notebooks/layer1_summaries")
        l_rest_dir = pathlib.Path("notebooks/summaries")
        print(l1_dir, l_rest_dir)
        # Order the files logically for the report
        report_structure = [
            ("LAYER 1: RAW PERFORMANCE SUMMARIES", list(l1_dir.glob("*.txt"))),
            ("LAYER 2: SOLVER FIDELITY & RELIABILITY", [l_rest_dir / "layer2_solver_fidelity_report.txt"]),
            ("LAYER 3: REGIME & CROSSOVER ANALYSIS", [l_rest_dir / "layer3_operational_crossover.txt"]),
            ("LAYER 4: RIGOROUS STATISTICAL VALIDATION", [l_rest_dir / "layer4_statistical_report.txt"]),
        ]

        with open(output_path, "w", encoding="utf-8") as outfile:
            outfile.write("="*80 + "\n")
            outfile.write("   QUANTUM VS CLASSICAL PORTFOLIO OPTIMIZATION: MASTER BENCHMARK\n")
            outfile.write("="*80 + "\n\n")

            for section_title, file_list in report_structure:
                outfile.write(f"\n\n{'#'*len(section_title)}\n")
                outfile.write(f"{section_title}\n")
                outfile.write(f"{'#'*len(section_title)}\n\n")

                for file_path in file_list:
                    if file_path.exists():
                        outfile.write(f"--- SOURCE: {file_path.name} ---\n")
                        with open(file_path, "r") as infile:
                            outfile.write(infile.read())
                        outfile.write("\n" + "-"*40 + "\n\n")
                    else:
                        outfile.write(f"!!! MISSING FILE: {file_path.name} !!!\n\n")

        return f"Full report compiled to {output_path}"

    # Run the concatenation
    result_msg = concat_all_reports()
    print(result_msg)
    return


if __name__ == "__main__":
    app.run()
