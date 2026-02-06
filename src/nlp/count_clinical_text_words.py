"""
Quick statistics script to count total words in clinical notes datasets.
For curiosity/statistical analysis before Week 8 NLP work.
"""

import pandas as pd
from pathlib import Path
import re

def count_words(text):
    """Count words in text string."""
    if pd.isna(text):
        return 0
    return len(str(text).split())

def analyze_dataset(filepath, text_columns):
    """Analyze word counts in a dataset."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {filepath.name}")
    print('='*70)
    
    df = pd.read_csv(filepath)
    print(f"Total rows: {len(df):,}")
    
    total_words = 0
    stats = {}
    
    for col in text_columns:
        if col in df.columns:
            word_counts = df[col].apply(count_words)
            col_total = word_counts.sum()
            col_mean = word_counts.mean()
            col_median = word_counts.median()
            col_max = word_counts.max()
            
            total_words += col_total
            stats[col] = {
                'total_words': col_total,
                'mean_words': col_mean,
                'median_words': col_median,
                'max_words': col_max
            }
            
            print(f"\n  Column: '{col}'")
            print(f"    Total words: {col_total:,}")
            print(f"    Mean words/entry: {col_mean:.1f}")
            print(f"    Median words/entry: {col_median:.1f}")
            print(f"    Max words/entry: {col_max:,}")
        else:
            print(f"\n  Column '{col}' not found in dataset")
    
    return total_words, stats


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "raw"
    
    print("\n" + "="*70)
    print("CLINICAL NOTES WORD COUNT ANALYSIS")
    print("="*70)
    
    grand_total_words = 0
    all_stats = {}
    
    # -------------------------------------------------------------------------
    # 1. ChatDoctor Dataset
    # -------------------------------------------------------------------------
    chatdoctor_dir = DATA_DIR / "ChatDoctor"
    
    # chatdoctor5k.json - Medical Q&A pairs
    print("\n\n[1] ChatDoctor - chatdoctor5k.json")
    chatdoctor5k_path = chatdoctor_dir / "chatdoctor5k.json"
    if chatdoctor5k_path.exists():
        df = pd.read_json(chatdoctor5k_path)
        print(f"    Format: JSON with {len(df):,} entries")
        
        # Check columns
        print(f"    Columns: {list(df.columns)}")
        
        total_words = 0
        for col in df.columns:
            if df[col].dtype == 'object':  # Text columns
                word_count = df[col].apply(count_words).sum()
                total_words += word_count
                print(f"      {col}: {word_count:,} words")
        
        grand_total_words += total_words
        all_stats['chatdoctor5k'] = total_words
    else:
        print("    FILE NOT FOUND")
    
    # alpaca_data.json - Instruction dataset
    print("\n[2] ChatDoctor - alpaca_data.json")
    alpaca_path = chatdoctor_dir / "alpaca_data.json"
    if alpaca_path.exists():
        df = pd.read_json(alpaca_path)
        print(f"    Format: JSON with {len(df):,} entries")
        print(f"    Columns: {list(df.columns)}")
        
        total_words = 0
        for col in df.columns:
            if df[col].dtype == 'object':
                word_count = df[col].apply(count_words).sum()
                total_words += word_count
                print(f"      {col}: {word_count:,} words")
        
        grand_total_words += total_words
        all_stats['alpaca_data'] = total_words
    else:
        print("    FILE NOT FOUND")
    
    # format_dataset.csv
    print("\n[3] ChatDoctor - format_dataset.csv")
    format_csv_path = chatdoctor_dir / "format_dataset.csv"
    if format_csv_path.exists():
        df = pd.read_csv(format_csv_path)
        print(f"    Format: CSV with {len(df):,} rows")
        print(f"    Columns: {list(df.columns)}")
        
        total_words = 0
        for col in df.columns:
            if df[col].dtype == 'object':
                word_count = df[col].apply(count_words).sum()
                total_words += word_count
                print(f"      {col}: {word_count:,} words")
        
        grand_total_words += total_words
        all_stats['format_dataset'] = total_words
    else:
        print("    FILE NOT FOUND")
    
    # -------------------------------------------------------------------------
    # 2. MTS-Dialog Dataset
    # -------------------------------------------------------------------------
    mts_dir = DATA_DIR / "MTS-Dialog" / "Main-Dataset"
    
    mts_files = [
        ("MTS-Dialog-TrainingSet.csv", ['dialogue', 'note', 'section_text']),
        ("MTS-Dialog-ValidationSet.csv", ['dialogue', 'note', 'section_text']),
        ("MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv", ['dialogue', 'note', 'section_text']),
        ("MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv", ['dialogue', 'note', 'section_text'])
    ]
    
    print("\n\n[4] MTS-Dialog Dataset")
    for filename, text_cols in mts_files:
        filepath = mts_dir / filename
        if filepath.exists():
            words, _ = analyze_dataset(filepath, text_cols)
            grand_total_words += words
            all_stats[filename] = words
        else:
            print(f"\n  {filename}: FILE NOT FOUND")
    
    # -------------------------------------------------------------------------
    # 3. MTS-Dialog Augmented (if needed)
    # -------------------------------------------------------------------------
    mts_aug_dir = DATA_DIR / "MTS-Dialog" / "Augmented-Data"
    
    aug_files = [
        "MTS-Dialog-Augmented-TrainingSet-1-En-FR-EN-2402-Pairs.csv",
        "MTS-Dialog-Augmented-TrainingSet-2-EN-ES-EN-2402-Pairs.csv",
        "MTS-Dialog-Augmented-TrainingSet-3-FR-and-ES-3603-Pairs-final.csv"
    ]
    
    print("\n\n[5] MTS-Dialog Augmented Data (Optional)")
    for filename in aug_files:
        filepath = mts_aug_dir / filename
        if filepath.exists():
            # These are translations, likely have similar columns
            words, _ = analyze_dataset(filepath, ['dialogue', 'note', 'section_text'])
            grand_total_words += words
            all_stats[filename] = words
        else:
            print(f"\n  {filename}: FILE NOT FOUND")
    
    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("GRAND TOTAL SUMMARY")
    print("="*70)
    print(f"\nTotal words across all clinical notes datasets: {grand_total_words:,}")
    print(f"\nEstimated pages (250 words/page): {grand_total_words/250:,.0f}")
    print(f"Estimated pages (500 words/page): {grand_total_words/500:,.0f}")
    print(f"Estimated tokens (1.3 words/token): {grand_total_words/1.3:,.0f}")
    
    print("\n\nBreakdown by dataset:")
    print("-" * 70)
    for dataset, words in sorted(all_stats.items(), key=lambda x: x[1], reverse=True):
        pct = (words / grand_total_words * 100) if grand_total_words > 0 else 0
        print(f"  {dataset:50s} {words:12,} words ({pct:5.1f}%)")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
