import os
import json
import glob
import matplotlib.pyplot as plt


def get_latest_results():
    """Get the most recent results file from the results directory."""
    results_dir = "results"
    result_files = glob.glob(os.path.join(results_dir, "results_*.json"))
    if not result_files:
        raise FileNotFoundError("No results files found")
    
    latest_file = max(result_files, key=os.path.getctime)
    with open(latest_file, 'r') as f:
        return json.load(f)


def create_markdown_tables(results):
    """Create markdown tables for regression and classification results."""
    regression_results = []
    classification_results = []
    
    for result in results['results']:
        if result['status'] != 'completed':
            continue
            
        row = {
            'Dataset': result['dataset'],
            'Synthetic': str(result['is_synthetic']),
            'Samples': result['n_samples'],
            result['final_metrics']['metric']['name'] + '@KappaML': round(
                result['final_metrics']['metric']['value'], 4
            ),
            result['final_metrics']['metric']['name'] + '@Local': round(
                result['local_final_metrics'], 4
            )
        }
        
        if result['task'] == 'regression':
            regression_results.append(row)
        else:
            classification_results.append(row)
    
    def create_table(results, title):
        if not results:
            return f"### {title}\nNo results available.\n\n"
            
        headers = list(results[0].keys())
        header_row = " | ".join(headers)
        separator = "|".join(["-" * len(h) for h in headers])
        
        rows = []
        for result in results:
            row = " | ".join(str(result[h]) for h in headers)
            rows.append(row)
            
        table = f"### {title}\n| {header_row} |\n| {separator} |\n"
        table += "\n".join(f"| {row} |" for row in rows) + "\n\n"
        return table
    
    reg_table = create_table(regression_results, "Regression Results")
    class_table = create_table(classification_results, "Classification Results")
    
    return reg_table, class_table


def plot_task_results(results, task):
    """Plot all models for a specific task type."""
    task_results = [
        r for r in results['results']
        if r['task'] == task and r['status'] == 'completed'
    ]
    
    if not task_results:
        return
        
    plt.figure(figsize=(12, 6))
    for result in task_results:
        samples = [m['samples'] for m in result['metrics']]
        scores = [m['metrics']['metric']['value'] for m in result['metrics']]
        label = f"{result['dataset']} "
        label += f"({'synthetic' if result['is_synthetic'] else 'real'})"
        plt.plot(samples, scores, label=label)
    
    plt.xlabel('Samples')
    plt.ylabel(result['final_metrics']['metric']['name'])
    plt.title(f'{task.capitalize()} Models Performance')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'figures/{task}_results.png')
    plt.close()


def plot_individual_models(results):
    """Create composite figures with separate plots for real and synthetic datasets in regression
    and classification tasks."""
    # Separate results by task and dataset type
    regression_real = [
        r for r in results['results'] 
        if r['task'] == 'regression' and not r['is_synthetic'] and r['status'] == 'completed'
    ]
    regression_synthetic = [
        r for r in results['results'] 
        if r['task'] == 'regression' and r['is_synthetic'] and r['status'] == 'completed'
    ]
    classification_real = [
        r for r in results['results'] 
        if r['task'] == 'classification' and not r['is_synthetic'] and r['status'] == 'completed'
    ]
    classification_synthetic = [
        r for r in results['results'] 
        if r['task'] == 'classification' and r['is_synthetic'] and r['status'] == 'completed'
    ]
    
    def create_task_figure(task_results, task_name, dataset_type):
        if not task_results:
            return
            
        # Calculate grid dimensions
        n_plots = len(task_results)
        n_cols = min(3, n_plots)  # Max 3 plots per row
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create figure
        fig = plt.figure(figsize=(15, 5 * n_rows))
        title = f'{task_name} Models Performance - {dataset_type} Datasets'
        fig.suptitle(title, fontsize=16, y=1.02)
        
        for idx, result in enumerate(task_results, 1):
            ax = fig.add_subplot(n_rows, n_cols, idx)
            
            # Plot KappaML performance
            samples = [m['samples'] for m in result['metrics']]
            scores = [m['metrics']['metric']['value'] for m in result['metrics']]
            ax.plot(samples, scores, label='KappaML', linewidth=2)
            
            # Plot baseline models
            for baseline_name, baseline_metrics in result['baseline_metrics'].items():
                baseline_samples = [m['samples'] for m in baseline_metrics]
                baseline_scores = [m['metrics']['metric']['value'] for m in baseline_metrics]
                ax.plot(baseline_samples, baseline_scores, '--', label=f'Baseline: {baseline_name}')
            
            ax.set_xlabel('Samples')
            ax.set_ylabel(result['final_metrics']['metric']['name'])
            ax.set_title(f"{result['dataset']}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize='small')
            
        plt.tight_layout()
        
        # Save plot
        filename = f"figures/{task_name.lower()}_{dataset_type.lower()}_composite.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
    
    # Create composite figures for each task and dataset type
    create_task_figure(regression_real, 'Regression', 'Real')
    create_task_figure(regression_synthetic, 'Regression', 'Synthetic')
    create_task_figure(classification_real, 'Classification', 'Real')
    create_task_figure(classification_synthetic, 'Classification', 'Synthetic')


def plot_individual_model(result):
    """Create individual plot for a single model."""
    if result['status'] != 'completed':
        return
        
    plt.figure(figsize=(10, 6))
    
    # Plot KappaML performance
    samples = [m['samples'] for m in result['metrics']]
    scores = [m['metrics']['metric']['value'] for m in result['metrics']]
    plt.plot(samples, scores, label='KappaML', linewidth=2)
    
    # Plot baseline models
    for baseline_name, baseline_metrics in result['baseline_metrics'].items():
        baseline_samples = [m['samples'] for m in baseline_metrics]
        baseline_scores = [m['metrics']['metric']['value'] for m in baseline_metrics]
        plt.plot(baseline_samples, baseline_scores, '--', label=f'Baseline: {baseline_name}')
    
    plt.xlabel('Samples')
    plt.ylabel(result['final_metrics']['metric']['name'])
    plt.title(f"{result['dataset']} ({'synthetic' if result['is_synthetic'] else 'real'})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save individual plot
    filename = f"figures/model_{result['dataset'].lower()}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


def main():
    # Load results
    results = get_latest_results()
    
    # Create markdown tables
    reg_table, class_table = create_markdown_tables(results)
    
    # Update results Markdown file
    with open('results/results.md', 'w') as f:
        f.write(reg_table + class_table)
    
    # Create plots
    plot_task_results(results, 'regression')
    plot_task_results(results, 'classification')
    plot_individual_models(results)
    
    # Create individual model plots
    for result in results['results']:
        plot_individual_model(result)


if __name__ == "__main__":
    main()
