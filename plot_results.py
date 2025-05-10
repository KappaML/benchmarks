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
    """Create individual plots for each model."""
    for result in results['results']:
        if result['status'] != 'completed':
            continue
            
        plt.figure(figsize=(10, 5))
        samples = [m['samples'] for m in result['metrics']]
        scores = [m['metrics']['metric']['value'] for m in result['metrics']]
        
        plt.plot(samples, scores)
        plt.xlabel('Samples')
        plt.ylabel(result['final_metrics']['metric']['name'])
        title = f"{result['dataset']} "
        title += f"({'synthetic' if result['is_synthetic'] else 'real'})"
        plt.title(title)
        plt.tight_layout()
        
        # Save plot
        filename = f"figures/model_{result['dataset'].lower()}.png"
        plt.savefig(filename)
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


if __name__ == "__main__":
    main()
