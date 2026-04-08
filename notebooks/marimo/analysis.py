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

    
    df_clb = pd.read_csv(PATH_CLASSICAL_BASELINE)

    df_qqa = pd.read_csv(PATH_QUANTUM_ANNEALING)
    df_qca = pd.read_csv(PATH_CLASSICAL_ANNEALING)
    df_qgb = pd.read_csv(PATH_GATEBASED_Y_SIMULATEDBIFRUCATION)
    df_sb = df_qgb[df_qgb['method'] == 'ANALOG_BASED'].copy()
    df_gate = df_qgb[df_qgb['method'] == 'GATE_BASED'].copy()

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

    return df_clb, df_gate, df_qca, df_qqa, df_sb, mo, pd, plt, sns


@app.cell
def _(df_clb, df_gate, df_qca, df_qqa, df_sb, mo, plt, sns):


    # Assuming 'df' is your loaded and cleaned combined dataframe
    def instance_characterization(df, df_name: str):
        # 1. Filter for unique instances to prevent solver-run weighting
        # We include K_target and lambda_cardinality as they define the QUBO constraints
        instances = df.groupby([
            'n_assets', 'n_bits', 'lambda_risk', 'lambda_reward', 
            'lambda_cardinality', 'K_target'
        ]).first().reset_index()
    
        fig, axes = plt.subplots(3, 2, figsize=(14, 18))
        fig.suptitle(f"Layer 1 Analysis: {df_name}", fontsize=18, fontweight='bold')
        plt.subplots_adjust(hspace=0.4, wspace=0.3)

        # --- ROW 1: Problem Structure ---
    
        # Plot A: Distribution of Active Assets (Resulting Sparsity)
        sns.histplot(instances['active_assets'], bins=10, ax=axes[0, 0], color='skyblue', kde=True)
        axes[0, 0].set_title("Distribution of Active Assets", fontsize=12)
    
        # Plot B: Landscape Ruggedness by Bit Depth
        # High ruggedness usually correlates with harder optimization for QA
        sns.boxplot(data=instances, x='n_bits', y='ruggedness', ax=axes[0, 1], palette="Set2")
        axes[0, 1].set_title("Landscape Ruggedness by Bit Depth", fontsize=12)

        # --- ROW 2: Parameter Relationships ---

        # Plot C: Correlation Heatmap
        # Identifies if parameters like n_bits are accidentally driving ruggedness
        corr = instances[['n_assets', 'n_bits', 'lambda_risk', 'lambda_reward', 'ruggedness']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1, 0], cbar=False)
        axes[1, 0].set_title("Input Parameter Multi-collinearity", fontsize=12)

        # Plot D: Financial Diversity (Return vs Risk Regimes)
        sns.scatterplot(data=instances, x='lambda_risk', y='expected_return', hue='n_assets', 
                        style='n_bits', palette="viridis", ax=axes[1, 1])
        axes[1, 1].set_title("Expected Return across Risk Regimes", fontsize=12)
        axes[1, 1].legend(title="N Assets", bbox_to_anchor=(1.05, 1), loc='upper left')

        # --- ROW 3: Optimization Difficulty Metrics ---

        # Plot E: Constraint Tightness (K/N ratio)
        # Ratios near 0.5 are often the most combinatorially difficult
        instances['k_ratio'] = instances['K_target'] / instances['n_assets']
        sns.kdeplot(data=instances, x='k_ratio', hue='n_assets', fill=True, common_norm=False, 
                    palette="tab10", ax=axes[2, 0])
        axes[2, 0].set_title("Constraint Tightness (K/N Ratio)", fontsize=12)
        axes[2, 0].set_xlabel("K_target / N_assets")

        # Plot F: The Ruggedness vs. Risk Scaling
        # Shows how the problem "difficulty" scales as the risk penalty increases
        sns.lineplot(data=instances, x='lambda_risk', y='ruggedness', hue='n_assets', 
                     marker='o', ax=axes[2, 1])
        axes[2, 1].set_title("Ruggedness Scaling by Risk Penalty", fontsize=12)

        return fig


    layer1_view = mo.vstack([
    mo.md("## Layer 1: Problem-Characterization EDA")
    , mo.as_html(instance_characterization(df_clb, 'Classical Baseline'))
    , mo.as_html(instance_characterization(df_sb, 'Simulated Bifrucation'))
    , mo.as_html(instance_characterization(df_qqa, 'Quantum Annealing'))
    , mo.as_html(instance_characterization(df_qca, 'Classical Annealing'))
    , mo.as_html(instance_characterization(df_gate, 'Gatebased QAOA'))
    ])
    layer1_view
    return


@app.cell
def _(df_clb, df_gate, df_qca, df_qqa, df_sb):
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
        (df_clb, 'Classical Baseline'),
        (df_sb, 'Simulated Bifurcation'),
        (df_qqa, 'Quantum Annealing'),
        (df_qca, 'Classical Annealing'),
        (df_gate, 'Gatebased QAOA')
    ]

    for df, name in solvers:
        export_layer1_summary(df, name)
    return


@app.cell
def _(df_clb, df_gate, df_qca, df_qqa, df_sb, mo, pd, plt, sns):
    from scipy.stats import kruskal, spearmanr

    def solver_performance_analysis(df_list, names):
        # 1. Combine all data for cross-solver comparison
        combined_df = pd.concat([d.assign(method=n) for d, n in zip(df_list, names)])
    
        # 2. Statistical Test: Kruskal-Wallis H-test for Ruggedness Perception
        # Do different solvers see significantly different landscapes?
        ruggedness_groups = [d['ruggedness'].values for d in df_list]
        h_stat, p_val = kruskal(*ruggedness_groups)
    
        fig, axes = plt.subplots(3, 2, figsize=(16, 20))
        fig.suptitle(f"Layer 2 Analysis: Solver Comparison & Fidelity", fontsize=20, fontweight='bold')
        plt.subplots_adjust(hspace=0.4, wspace=0.3)

        # --- ROW 1: The "Finance" Gap ---

        # Plot A: Sharpe Ratio by Solver
        sns.boxplot(data=combined_df, x='method', y='sharpe_ratio', palette="Set3", ax=axes[0, 0])
        axes[0, 0].set_title("Sharpe Ratio Distribution by Solver", fontsize=14)
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Plot B: Cardinality Violation (Constraint Fidelity)
        combined_df['k_error'] = (combined_df['active_assets'] - combined_df['K_target']).abs()
        sns.barplot(data=combined_df, x='method', y='k_error', palette="Reds", ax=axes[0, 1])
        axes[0, 1].set_title("Mean Cardinality Error (|Active - K|)", fontsize=14)
        axes[0, 1].tick_params(axis='x', rotation=45)

        # --- ROW 2: The "Energy" Perception (Layer 2 Core) ---

        # Plot C: Perceived Ruggedness (The H-test Visual)
        sns.violinplot(data=combined_df, x='method', y='ruggedness', ax=axes[1, 0], cut=0)
        axes[1, 0].set_title(f"Perceived Ruggedness (H-Stat: {h_stat:.2f}, p: {p_val:.4f})", fontsize=14)
        axes[1, 0].set_yscale('log')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Plot D: Time-to-Solve vs Quality
        sns.scatterplot(data=combined_df, x='time_to_solve', y='sharpe_ratio', hue='method', alpha=0.5, ax=axes[1, 1])
        axes[1, 1].set_xscale('log')
        axes[1, 1].set_title("Efficiency Frontier: Time vs Sharpe", fontsize=14)

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
        sns.barplot(data=corr_df, x='method', y='spearman_rho', palette="coolwarm", ax=axes[2, 0])
        axes[2, 0].set_title("QUBO Fidelity: Spearman(Energy, Sharpe)", fontsize=14)
        axes[2, 0].axhline(0, color='black', linewidth=0.8)
        axes[2, 0].tick_params(axis='x', rotation=45)

        # Plot F: Sharpe Decay vs Ruggedness
        sns.lineplot(data=combined_df, x='ruggedness', y='sharpe_ratio', hue='method', ax=axes[2, 1])
        axes[2, 1].set_title("Solver Resilience to Landscape Ruggedness", fontsize=14)
        axes[2, 1].set_xscale('log')

        return fig

    # Execution block
    dfs = [df_clb, df_sb, df_qqa, df_qca, df_gate]
    names = ['Baseline', 'Sim. Bifurcation', 'Quantum Annealing', 'Classical Annealing', 'Gate-based QAOA']

    layer2_view = mo.vstack([
        mo.md("## Layer 2: Solver Outcome & Statistical Validation"),
        mo.as_html(solver_performance_analysis(dfs, names))
    ])
    layer2_view
    return


if __name__ == "__main__":
    app.run()
