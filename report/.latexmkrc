# Keep all generated files out of the source directory.
# This affects manual `latexmk` runs from within the report/ folder.
$out_dir = 'build';
$aux_dir = 'build';

# Use user-space biber install (needed for biblatex).
$biber = '/home/docampor/bin/biber';
