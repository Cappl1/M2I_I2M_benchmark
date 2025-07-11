#!/usr/bin/env python3
"""
Diagnostic script to check experiment configurations and identify issues.
"""

import json
from pathlib import Path
import sys


def diagnose_experiments(logs_dir: str):
    """Check all experiment configurations."""
    logs_path = Path(logs_dir)
    
    print(f"Scanning directory: {logs_path}")
    print("=" * 60)
    
    stats = {
        'total': 0,
        'with_results': 0,
        'task_incremental': 0,
        'class_incremental': 0,
        'unknown': 0,
        'phase1_only': 0,
        'phase2_only': 0,
        'both_phases': 0,
        'no_phases': 0
    }
    
    experiments = []
    
    # Find all BinaryPairs experiments
    for exp_dir in sorted(logs_path.glob('*BinaryPairsExperiment*')):
        stats['total'] += 1
        
        exp_info = {
            'dir': exp_dir.name,
            'has_results': False,
            'scenario_type': 'unknown',
            'dataset_a': 'unknown',
            'dataset_b': 'unknown',
            'phase1': False,
            'phase2': False,
            'config_source': None
        }
        
        # Check for results.json
        results_file = exp_dir / 'results.json'
        if results_file.exists():
            stats['with_results'] += 1
            exp_info['has_results'] = True
            
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    
                config = results.get('config', {})
                exp_info['scenario_type'] = config.get('scenario_type', 'unknown')
                exp_info['dataset_a'] = config.get('dataset_a', 'unknown')
                exp_info['dataset_b'] = config.get('dataset_b', 'unknown')
                exp_info['config_source'] = 'results.json'
                
                # Count scenario types
                if exp_info['scenario_type'] == 'task_incremental':
                    stats['task_incremental'] += 1
                elif exp_info['scenario_type'] == 'class_incremental':
                    stats['class_incremental'] += 1
                else:
                    stats['unknown'] += 1
                    
            except Exception as e:
                print(f"Error reading {results_file}: {e}")
        
        # Check for config files if no results.json
        if not exp_info['has_results']:
            # Try config.yaml or config.yml
            for config_name in ['config.yaml', 'config.yml', 'config_*.yaml', 'config_*.yml']:
                config_files = list(exp_dir.glob(config_name))
                if config_files:
                    config_file = config_files[0]
                    try:
                        with open(config_file, 'r') as f:
                            content = f.read()
                            if 'scenario_type:' in content:
                                for line in content.split('\n'):
                                    if 'scenario_type:' in line:
                                        scenario = line.split(':')[1].strip()
                                        exp_info['scenario_type'] = scenario
                                        exp_info['config_source'] = config_file.name
                                        
                                        if scenario == 'task_incremental':
                                            stats['task_incremental'] += 1
                                        elif scenario == 'class_incremental':
                                            stats['class_incremental'] += 1
                                        break
                    except Exception as e:
                        print(f"Error reading {config_file}: {e}")
        
        # Check for phase directories
        phase1_dir = exp_dir / 'layer_analysis' / 'phase1'
        phase2_dir = exp_dir / 'layer_analysis' / 'phase2'
        
        exp_info['phase1'] = phase1_dir.exists()
        exp_info['phase2'] = phase2_dir.exists()
        
        # Count phase statistics
        if exp_info['phase1'] and exp_info['phase2']:
            stats['both_phases'] += 1
        elif exp_info['phase1']:
            stats['phase1_only'] += 1
        elif exp_info['phase2']:
            stats['phase2_only'] += 1
        else:
            stats['no_phases'] += 1
        
        experiments.append(exp_info)
    
    # Print summary
    print("\nSUMMARY STATISTICS:")
    print(f"Total experiments found: {stats['total']}")
    print(f"With results.json: {stats['with_results']}")
    print(f"Task incremental: {stats['task_incremental']}")
    print(f"Class incremental: {stats['class_incremental']}")
    print(f"Unknown scenario: {stats['unknown']}")
    print()
    print("Phase completion:")
    print(f"Both phases: {stats['both_phases']}")
    print(f"Phase 1 only: {stats['phase1_only']}")
    print(f"Phase 2 only: {stats['phase2_only']}")
    print(f"No phases: {stats['no_phases']}")
    
    # Detailed listing
    print("\n" + "=" * 60)
    print("DETAILED EXPERIMENT LISTING:")
    print("=" * 60)
    
    # Group by scenario type
    for scenario in ['class_incremental', 'task_incremental', 'unknown']:
        scenario_exps = [e for e in experiments if e['scenario_type'] == scenario]
        if scenario_exps:
            print(f"\n{scenario.upper()} ({len(scenario_exps)} experiments):")
            print("-" * 40)
            
            for exp in scenario_exps:
                phase_status = ""
                if exp['phase1'] and exp['phase2']:
                    phase_status = "✓✓"
                elif exp['phase1']:
                    phase_status = "✓-"
                elif exp['phase2']:
                    phase_status = "-✓"
                else:
                    phase_status = "--"
                
                print(f"{exp['dir'][:40]:<40} | {exp['dataset_a']:>10} → {exp['dataset_b']:<10} | Phases: {phase_status} | Config: {exp['config_source'] or 'none'}")
    
    # Check for potential issues
    print("\n" + "=" * 60)
    print("POTENTIAL ISSUES:")
    print("=" * 60)
    
    # Experiments without results.json
    no_results = [e for e in experiments if not e['has_results']]
    if no_results:
        print(f"\n{len(no_results)} experiments without results.json:")
        for exp in no_results[:5]:  # Show first 5
            print(f"  - {exp['dir']}")
        if len(no_results) > 5:
            print(f"  ... and {len(no_results) - 5} more")
    
    # Incomplete experiments
    incomplete = [e for e in experiments if e['phase1'] and not e['phase2']]
    if incomplete:
        print(f"\n{len(incomplete)} incomplete experiments (phase 1 only):")
        for exp in incomplete[:5]:
            print(f"  - {exp['dir']} ({exp['dataset_a']} → {exp['dataset_b']})")
        if len(incomplete) > 5:
            print(f"  ... and {len(incomplete) - 5} more")
    
    # Check if class_incremental might be mislabeled
    if stats['class_incremental'] == 0:
        print("\n⚠️  NO CLASS_INCREMENTAL EXPERIMENTS FOUND!")
        print("This might indicate:")
        print("1. All experiments were run with task_incremental setting")
        print("2. Config files might have incorrect scenario_type")
        print("3. The experiment script might be overriding the config")
        
        # Check experiment names for hints
        print("\nChecking experiment names for 'class' keyword:")
        class_hints = [e for e in experiments if 'class' in e['dir'].lower()]
        if class_hints:
            print(f"Found {len(class_hints)} experiments with 'class' in name:")
            for exp in class_hints[:3]:
                print(f"  - {exp['dir']} (marked as: {exp['scenario_type']})")
    
    return experiments, stats


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnostic_script.py <logs_directory>")
        sys.exit(1)
    
    logs_dir = sys.argv[1]
    experiments, stats = diagnose_experiments(logs_dir)
    
    # Save diagnostic report
    report = {
        'statistics': stats,
        'experiments': experiments
    }
    
    with open('diagnostic_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDiagnostic report saved to: diagnostic_report.json")


if __name__ == "__main__":
    main()
    