"""
Orchestrator script to run train -> run -> eval pipeline for all modes and tables.

Usage:
    python orchestrator.py                    # Run all (no_prompt, with_cot, without_cot) x (1,2,3,4)
"""

from model_scripts.train_model import main as train_main
from model_scripts.run_model import main as run_main
from model_scripts.eval_model import main as eval_main

# ============================================================================
# CONFIGURATION - Modifier ici pour changer ce qui s'exécute
# ============================================================================
MODES = ["no_prompt", "with_cot", "without_cot"]  # Modes à exécuter
TABLES = [4]                              # Tables à exécuter
PIPELINE_STEPS = ["train", "run", "eval"]         # Train → Run → Eval
# ============================================================================

def main():
    """Run the complete pipeline."""
    
    total = len(MODES) * len(TABLES)
    count = 0
    
    print(f"\n{'='*80}")
    print(f"ORCHESTRATOR - Running {total} configurations")
    print(f"Modes: {', '.join(MODES)}")
    print(f"Tables: {', '.join(map(str, TABLES))}")
    print(f"{'='*80}\n")
    
    for mode in MODES:
        for table in TABLES:
            count += 1
            print(f"\n[{count}/{total}] {mode.upper()} - Table {table}")
            print("-" * 80)
            
            try:
                for step in PIPELINE_STEPS:
                    if step == "train":
                        print("  → Training...")
                        train_main(table_number=table, mode=mode)
                    elif step == "run":
                        print("  → Running inference...")
                        run_main(table_number=table, mode=mode, eval_set="test")
                    elif step == "eval":
                        print("  → Evaluating...")
                        eval_main(table_number=table, mode=mode)
                
                print(f"  ✓ Success\n")
                
            except Exception as e:
                print(f"  ✗ Error: {e}\n")
    
    print(f"\n{'='*80}")
    print("✓ ALL DONE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
