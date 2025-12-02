
#!/usr/bin/env python3
"""
Script to find the common prefix between two files.
Useful for checking what portion of prompts can be cached.
"""

import sys
import os


def find_common_prefix(text1: str, text2: str) -> tuple[str, int]:
    """
    Find the common prefix between two strings.
    Returns (common_prefix, length)
    """
    min_len = min(len(text1), len(text2))
    common_len = 0
    
    for i in range(min_len):
        if text1[i] == text2[i]:
            common_len += 1
        else:
            break
    
    return text1[:common_len], common_len


def find_common_prefix_until_marker(text1: str, text2: str, marker: str) -> tuple[str, int]:
    """
    Find the common prefix up to a specific marker.
    Useful for finding common prefix up to user message in prompts.
    """
    # Find marker position in both texts
    marker_pos1 = text1.find(marker)
    marker_pos2 = text2.find(marker)
    
    if marker_pos1 == -1 or marker_pos2 == -1:
        # Marker not found in one or both files
        return find_common_prefix(text1, text2)
    
    # Get prefix up to marker
    prefix1 = text1[:marker_pos1]
    prefix2 = text2[:marker_pos2]
    
    # Find common prefix of the prefixes
    return find_common_prefix(prefix1, prefix2)


def format_size(size: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def main():
    if len(sys.argv) < 3:
        print("Usage: python t.py <file1> <file2> [--until-marker MARKER]")
        print("\nExample:")
        print("  python t.py prompt_021_processed/prompt01.txt prompt_021_processed/prompt02.txt")
        print("  python t.py prompt_021_processed/prompt01.txt prompt_021_processed/prompt02.txt --until-marker '<|im_start|>user'")
        sys.exit(1)
    
    file1_path = sys.argv[1]
    file2_path = sys.argv[2]
    
    # Check for optional marker argument
    until_marker = None
    if len(sys.argv) >= 5 and sys.argv[3] == '--until-marker':
        until_marker = sys.argv[4]
    
    # Check if files exist
    if not os.path.exists(file1_path):
        print(f"Error: File '{file1_path}' not found")
        sys.exit(1)
    
    if not os.path.exists(file2_path):
        print(f"Error: File '{file2_path}' not found")
        sys.exit(1)
    
    # Read files
    print(f"Reading {file1_path}...")
    with open(file1_path, 'r', encoding='utf-8') as f:
        content1 = f.read()
    
    print(f"Reading {file2_path}...")
    with open(file2_path, 'r', encoding='utf-8') as f:
        content2 = f.read()
    
    file1_size = len(content1)
    file2_size = len(content2)
    
    print(f"\n{'='*80}")
    print(f"File 1: {file1_path}")
    print(f"  Size: {file1_size:,} characters ({format_size(file1_size)})")
    print(f"\nFile 2: {file2_path}")
    print(f"  Size: {file2_size:,} characters ({format_size(file2_size)})")
    print(f"{'='*80}\n")
    
    # Find common prefix
    if until_marker:
        print(f"Finding common prefix until marker: '{until_marker}'")
        common_prefix, common_len = find_common_prefix_until_marker(content1, content2, until_marker)
    else:
        print("Finding common prefix (character by character)...")
        common_prefix, common_len = find_common_prefix(content1, content2)
    
    # Calculate percentages
    pct1 = (common_len / file1_size * 100) if file1_size > 0 else 0
    pct2 = (common_len / file2_size * 100) if file2_size > 0 else 0
    
    # Print results
    print(f"\n{'='*80}")
    print("RESULTS:")
    print(f"{'='*80}")
    print(f"Common prefix length: {common_len:,} characters ({format_size(common_len)})")
    print(f"  - {pct1:.2f}% of file 1")
    print(f"  - {pct2:.2f}% of file 2")
    print(f"\nRemaining in file 1: {file1_size - common_len:,} characters")
    print(f"Remaining in file 2: {file2_size - common_len:,} characters")
    
    # Show preview of common prefix
    print(f"\n{'='*80}")
    print("COMMON PREFIX PREVIEW (first 500 characters):")
    print(f"{'='*80}")
    preview = common_prefix[:500]
    print(preview)
    if len(common_prefix) > 500:
        print(f"\n... ({len(common_prefix) - 500:,} more characters)")
    
    # Show where the difference starts
    if common_len < min(file1_size, file2_size):
        print(f"\n{'='*80}")
        print("DIFFERENCE STARTS AT:")
        print(f"{'='*80}")
        print(f"Position: {common_len:,}")
        
        # Show context around the difference
        context_size = 100
        start = max(0, common_len - context_size)
        end1 = min(file1_size, common_len + context_size)
        end2 = min(file2_size, common_len + context_size)
        
        print(f"\nFile 1 (around position {common_len:,}):")
        print("-" * 80)
        print(repr(content1[start:end1]))
        
        print(f"\nFile 2 (around position {common_len:,}):")
        print("-" * 80)
        print(repr(content2[start:end2]))
    
    # Additional analysis: check for system message prefix
    if "<|im_start|>system" in content1 and "<|im_start|>system" in content2:
        system_end1 = content1.find("<|im_start|>user")
        system_end2 = content2.find("<|im_start|>user")
        
        if system_end1 != -1 and system_end2 != -1:
            system_prefix1 = content1[:system_end1]
            system_prefix2 = content2[:system_end2]
            
            if system_prefix1 == system_prefix2:
                print(f"\n{'='*80}")
                print("SYSTEM MESSAGE ANALYSIS:")
                print(f"{'='*80}")
                print(f"✅ System messages are IDENTICAL")
                print(f"System message length: {len(system_prefix1):,} characters")
                print(f"This is the cacheable prefix for vLLM prefix caching!")
            else:
                sys_common, sys_common_len = find_common_prefix(system_prefix1, system_prefix2)
                print(f"\n{'='*80}")
                print("SYSTEM MESSAGE ANALYSIS:")
                print(f"{'='*80}")
                print(f"⚠️  System messages are DIFFERENT")
                print(f"Common system prefix: {sys_common_len:,} characters")
                print(f"File 1 system message: {len(system_prefix1):,} characters")
                print(f"File 2 system message: {len(system_prefix2):,} characters")


if __name__ == "__main__":
    main()

