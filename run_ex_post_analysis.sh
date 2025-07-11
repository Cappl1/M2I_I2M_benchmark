#!/bin/bash
# Comprehensive analysis pipeline for Binary Pairs experiments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
LOGS_DIR="${1:-logs}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="comprehensive_analysis_${TIMESTAMP}"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Binary Pairs Comprehensive Analysis Pipeline${NC}"
echo -e "${BLUE}================================================${NC}"
echo -e "Logs directory: ${GREEN}$LOGS_DIR${NC}"
echo -e "Output directory: ${GREEN}$OUTPUT_BASE${NC}"
echo ""

# Create output structure
mkdir -p "$OUTPUT_BASE"/{diagnostics,individual,batch,incomplete,fixes}

# Step 1: Run diagnostics
echo -e "\n${YELLOW}Step 1: Running diagnostics...${NC}"
echo "======================================"
python diagnostic_script.py "$LOGS_DIR" > "$OUTPUT_BASE/diagnostics/diagnostic_output.txt" 2>&1
cp diagnostic_report.json "$OUTPUT_BASE/diagnostics/"

# Show key diagnostic info
echo -e "${GREEN}Diagnostic summary:${NC}"
python -c "
import json
with open('$OUTPUT_BASE/diagnostics/diagnostic_report.json', 'r') as f:
    report = json.load(f)
    stats = report['statistics']
    print(f'  Total experiments: {stats[\"total\"]}')
    print(f'  Task incremental: {stats[\"task_incremental\"]}')
    print(f'  Class incremental: {stats[\"class_incremental\"]}')
    print(f'  Complete (both phases): {stats[\"both_phases\"]}')
    print(f'  Incomplete (phase 1 only): {stats[\"phase1_only\"]}')
"

# Step 2: Individual experiment analysis
echo -e "\n${YELLOW}Step 2: Analyzing individual experiments...${NC}"
echo "==========================================="
python ex_post_binary_analyzer.py "$LOGS_DIR" --output "$OUTPUT_BASE/individual" 2>&1 | tee "$OUTPUT_BASE/individual_analysis.log"

# Count successful analyses
INDIVIDUAL_SUCCESS=$(find "$OUTPUT_BASE/individual" -name "forgetting_dynamics.png" | wc -l)
echo -e "${GREEN}Successfully analyzed: $INDIVIDUAL_SUCCESS experiments${NC}"

# Step 3: Batch comparative analysis
echo -e "\n${YELLOW}Step 3: Running batch comparative analysis...${NC}"
echo "=============================================="
python batch_binary_analyzer.py "$LOGS_DIR" --output "$OUTPUT_BASE/batch" 2>&1 | tee "$OUTPUT_BASE/batch_analysis.log"

# Step 4: Analyze incomplete experiments
echo -e "\n${YELLOW}Step 4: Analyzing incomplete experiments...${NC}"
echo "==========================================="
python single_phase_analyzer.py "$LOGS_DIR" --output "$OUTPUT_BASE/incomplete" 2>&1 | tee "$OUTPUT_BASE/incomplete_analysis.log"

# Step 5: Generate Phase 2 dual task fixes for a few examples
echo -e "\n${YELLOW}Step 5: Generating example Phase 2 dual task plots...${NC}"
echo "====================================================="

# Find a few experiments to use as examples
EXAMPLE_COUNT=0
MAX_EXAMPLES=3

for exp_dir in "$LOGS_DIR"/*BinaryPairsExperiment*; do
    if [ -f "$exp_dir/results.json" ] && [ -d "$exp_dir/layer_analysis/phase2" ]; then
        exp_name=$(basename "$exp_dir")
        echo -e "  Generating dual task plot for: ${BLUE}$exp_name${NC}"
        
        python quick_phase2_fix.py "$exp_dir" \
            --output "$OUTPUT_BASE/fixes/${exp_name}_phase2_dual_task.png" 2>/dev/null || true
        
        ((EXAMPLE_COUNT++))
        if [ $EXAMPLE_COUNT -ge $MAX_EXAMPLES ]; then
            break
        fi
    fi
done

# Step 6: Create master summary
echo -e "\n${YELLOW}Step 6: Creating master summary...${NC}"
echo "===================================="

cat > "$OUTPUT_BASE/MASTER_README.md" << EOF
# Binary Pairs Analysis Master Summary

Generated on: $(date)

## Directory Structure

### 📊 \`diagnostics/\`
- \`diagnostic_report.json\` - Complete experiment inventory
- \`diagnostic_output.txt\` - Diagnostic run log

### 🔍 \`individual/\`
- Detailed analysis for each experiment
- Subdirectories named: \`{dataset_a}_{dataset_b}_{scenario_type}/\`
- Each contains forgetting dynamics, phase analyses, and summaries

### 📈 \`batch/\`
- \`forgetting_heatmap_*.png\` - Forgetting rate matrices
- \`layer_forgetting_comparison.png\` - Layer-wise forgetting patterns
- \`scenario_comparison.png\` - Class vs Task incremental comparison
- \`experiment_summary.csv\` - Tabular summary of all experiments
- \`analysis_report.md\` - Detailed findings

### ⚠️  \`incomplete/\`
- Analysis of experiments with only Phase 1 data
- \`incomplete_experiments_summary.md\` - List and potential causes

### 🔧 \`fixes/\`
- Example Phase 2 dual task plots showing both learning and forgetting

## Key Statistics

EOF

# Add statistics to summary
python -c "
import json
with open('$OUTPUT_BASE/diagnostics/diagnostic_report.json', 'r') as f:
    report = json.load(f)
    stats = report['statistics']
    
    print(f'- **Total experiments found:** {stats[\"total\"]}')
    print(f'- **Task incremental:** {stats[\"task_incremental\"]}')
    print(f'- **Class incremental:** {stats[\"class_incremental\"]}')
    print(f'- **Successfully completed:** {stats[\"both_phases\"]}')
    print(f'- **Incomplete (Phase 1 only):** {stats[\"phase1_only\"]}')
    print(f'- **Individual analyses completed:** $INDIVIDUAL_SUCCESS')
" >> "$OUTPUT_BASE/MASTER_README.md"

cat >> "$OUTPUT_BASE/MASTER_README.md" << EOF

## Quick Start Guide

1. **Check experiment status**: See \`diagnostics/diagnostic_report.json\`
2. **View forgetting patterns**: Open \`batch/forgetting_heatmap_*.png\`
3. **Compare scenarios**: See \`batch/scenario_comparison.png\`
4. **Examine specific pairs**: Browse \`individual/\` subdirectories
5. **Understand Phase 2 dynamics**: Check \`fixes/\` for dual task examples

## Common Issues Found

EOF

# Add issues summary
if [ -f "$OUTPUT_BASE/diagnostics/diagnostic_report.json" ]; then
    python -c "
import json
with open('$OUTPUT_BASE/diagnostics/diagnostic_report.json', 'r') as f:
    report = json.load(f)
    stats = report['statistics']
    
    if stats['class_incremental'] == 0:
        print('- ⚠️  **No class_incremental experiments found** - all experiments appear to be task_incremental')
    
    if stats['phase1_only'] > 0:
        print(f'- ⚠️  **{stats[\"phase1_only\"]} incomplete experiments** - Phase 2 failed or was interrupted')
    
    if stats['unknown'] > 0:
        print(f'- ⚠️  **{stats[\"unknown\"]} experiments with unknown scenario type**')
" >> "$OUTPUT_BASE/MASTER_README.md"
fi

echo "" >> "$OUTPUT_BASE/MASTER_README.md"
echo "## Log Files" >> "$OUTPUT_BASE/MASTER_README.md"
echo "" >> "$OUTPUT_BASE/MASTER_README.md"
echo "- \`individual_analysis.log\` - Individual analysis output" >> "$OUTPUT_BASE/MASTER_README.md"
echo "- \`batch_analysis.log\` - Batch analysis output" >> "$OUTPUT_BASE/MASTER_README.md"
echo "- \`incomplete_analysis.log\` - Incomplete experiment analysis" >> "$OUTPUT_BASE/MASTER_README.md"

# Final summary
echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}Analysis Pipeline Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo -e "Results saved to: ${BLUE}$OUTPUT_BASE${NC}"
echo ""
echo "Key outputs:"
echo -e "  📊 Diagnostics: ${BLUE}$OUTPUT_BASE/diagnostics/${NC}"
echo -e "  🔍 Individual: ${BLUE}$OUTPUT_BASE/individual/${NC}"
echo -e "  📈 Batch: ${BLUE}$OUTPUT_BASE/batch/${NC}"
echo -e "  ⚠️  Incomplete: ${BLUE}$OUTPUT_BASE/incomplete/${NC}"
echo -e "  🔧 Fixes: ${BLUE}$OUTPUT_BASE/fixes/${NC}"
echo ""
echo -e "Master summary: ${GREEN}$OUTPUT_BASE/MASTER_README.md${NC}"

# Check for critical issues
if grep -q "NO CLASS_INCREMENTAL EXPERIMENTS FOUND" "$OUTPUT_BASE/diagnostics/diagnostic_output.txt"; then
    echo ""
    echo -e "${RED}⚠️  WARNING: No class_incremental experiments detected!${NC}"
    echo "This suggests all experiments may have been run with task_incremental setting only."
fi

# Cleanup temporary files
rm -f diagnostic_report.json

# Open results if on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo ""
    read -p "Open results directory? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open "$OUTPUT_BASE"
    fi
fi