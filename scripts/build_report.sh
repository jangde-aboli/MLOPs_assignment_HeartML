#!/usr/bin/env bash
# Convert reports/REPORT.md to reports/REPORT.docx (needs pandoc).
# Open the .docx in Word and "Save as PDF" if a PDF is required.
set -euo pipefail
cd "$(dirname "$0")/../reports"
pandoc REPORT.md -o REPORT.docx --toc --resource-path=.:..
echo "wrote reports/REPORT.docx"
