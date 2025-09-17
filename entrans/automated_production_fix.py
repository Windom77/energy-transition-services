#!/usr/bin/env python3
"""
Quick fix to replace print statements with production-aware debug_print
"""

import re


def fix_production_logging(file_path):
    """Replace print statements with production-aware logging"""

    print(f"🔧 Fixing production logging in: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Add production mode detection at the top (after imports)
    production_header = '''
import os
import warnings

# Production mode detection
PRODUCTION_MODE = os.getenv('FLASK_ENV', 'development') == 'production'

# Suppress warnings in production
if PRODUCTION_MODE:
    warnings.filterwarnings('ignore')

# Debug print function
def debug_print(*args, **kwargs):
    """Print debug messages only when NOT in production"""
    if not PRODUCTION_MODE:
        print(*args, **kwargs)

'''

    # Find the first import statement and add our code after all imports
    import_pattern = r'((?:from .+? import .+?\n|import .+?\n)+)'
    match = re.search(import_pattern, content, re.MULTILINE)

    if match:
        imports_end = match.end()
        content = content[:imports_end] + production_header + content[imports_end:]
    else:
        # If no imports found, add at the top
        content = production_header + content

    # Replace print statements with debug_print for debug messages
    debug_patterns = [
        r'print\(f"\[FCAS DEBUG\]([^"]*?)"\)',
        r'print\(f"\[FCAS\]([^"]*?)"\)',
        r'print\(f"\[TARIFF\]([^"]*?)"\)',
        r'print\(f"\[BATTERY DEBUG\]([^"]*?)"\)',
        r'print\(f"\[BATTERY\]([^"]*?)"\)',
        r'print\(f"\[OPTIMIZED\]([^"]*?)"\)',
        r'print\(f"\[COST\]([^"]*?)"\)',
        r'print\(f"\[SIZING\]([^"]*?)"\)',
        r'print\(f"DEBUG:([^"]*?)"\)',
    ]

    replacements_made = 0
    for pattern in debug_patterns:
        matches = re.findall(pattern, content)
        if matches:
            replacements_made += len(matches)
            content = re.sub(pattern, r'debug_print(f"[\1]', content)

    # Write the modified content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"   ✅ Made {replacements_made} replacements")
    return replacements_made > 0


def main():
    files_to_fix = [
        'pysam_form_integration2.py',
        'dashboard_module.py',
        'PySAM_main_v2.py'
    ]

    print("🔧 Applying production logging fixes...")

    for file_path in files_to_fix:
        try:
            if fix_production_logging(file_path):
                print(f"   ✅ Fixed: {file_path}")
            else:
                print(f"   ℹ️  No changes needed: {file_path}")
        except Exception as e:
            print(f"   ❌ Error fixing {file_path}: {e}")

    print("\n🎉 Production logging fixes applied!")
    print("Test with: FLASK_ENV=production python app.py")


if __name__ == "__main__":
    main()