#!/bin/bash

# Use the Java-based PDFBox tool to extract text from pdf files to html format.
# Extract from N files at a time in parallel.
#
# Example usage on four threads:
#   bash ./parse_pdf_box.sh /path/to/pdfbox.jar ../../data/pdf/ ../../data/extracted/html 4


helpFunction()
{
    echo ""
    echo "Usage: bash $0 pdfbox_jar source_dir target_dir num_procs"
    exit 1
}

parsePdf()
{
    file=$1
    filename=$2
    java -jar $pdfbox_jar ExtractText $file $target_dir$filename.html -html -encoding UTF-8
    echo "Extracted $file to $target_dir$filename.html"
}

# Check command line parameters
exit_code=0
pdfbox_jar=$1
source_dir=$2
target_dir=$3
if [ -z "$pdfbox_jar" ] || [ -z "$source_dir" ] || [ -z "$target_dir" ] ||
[ -z "$4" ] || [ "$5" ]
then
    echo "Error: Missing or extraneous arguments";
    helpFunction
fi
MAX_PROCS=$(nproc --all)
num_procs=$(($4< $MAX_PROCS? $4 : $MAX_PROCS))

if [ ! -d "$source_dir" ]; then
    echo "Source directory does not exist: $source_dir";
    helpFunction
fi


# End directory parameters with a forward slash
[[ "${source_dir}" != */ ]] && source_dir="${source_dir}/"
[[ "${target_dir}" != */ ]] && target_dir="${target_dir}/"


# Create target directories if they do not exist
directories=`find $source_dir -type d -printf "%P\n"`
for dir in $directories; do
    if [ ! -d "$target_dir$dir" ]; then
        `mkdir -p $target_dir$dir`;
        echo "Created target directory: $target_dir$dir";
    fi
done


# Process all pdf files in the directory tree in parallel
files=`find $source_dir -name '*.pdf' -printf "%P\n"`
(
for file in $files; do
    filename="${file%.*}"
    ((i=i%$num_procs)); ((i++==0)) && wait
    parsePdf $source_dir$file $filename &
done
wait
)

echo "Exiting with exit code: $exit_code"
exit $exit_code
