# Build the paper. Run from anywhere:
#   powershell -File paper\isai-nlp-2026\build.ps1
#
# MiKTeX was installed per-user (winget install --id MiKTeX.MiKTeX -e --scope user),
# so its bin directory may not be on PATH in a shell that started before the
# install. This script calls the binaries by full path and does not care.

$ErrorActionPreference = 'Stop'
$B = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64"
if (-not (Test-Path "$B\pdflatex.exe")) {
    throw "pdflatex not found at $B -- is MiKTeX installed for this user?"
}
Set-Location $PSScriptRoot

# pdflatex -> bibtex -> pdflatex x2 is the standard cycle: the first pass
# writes main.aux, bibtex turns it into main.bbl, and two more passes settle
# the citation labels and the cross-references.
& "$B\pdflatex.exe" -interaction=nonstopmode main.tex | Out-Null
& "$B\bibtex.exe" main | Out-Null
& "$B\pdflatex.exe" -interaction=nonstopmode main.tex | Out-Null
& "$B\pdflatex.exe" -interaction=nonstopmode main.tex | Out-Null

# Report the things that are wrong but do not stop the compile. An overfull
# hbox is text sticking into the margin; a missing citation prints as [?].
# pdflatex exits 0 on all of them, which is why they are checked here.
$over = @(Select-String -Path main.log -Pattern 'Overfull \\hbox')
$undef = @(Select-String -Path main.log -Pattern 'Citation .* undefined|Reference .* undefined')
$pages = (Select-String -Path main.log -Pattern 'Output written on').Line

Write-Host ""
Write-Host "  $pages"
Write-Host "  overfull hboxes:      $($over.Count)"
Write-Host "  undefined refs/cites: $($undef.Count)"
foreach ($o in $over) { Write-Host "    $($o.Line)" }
foreach ($u in $undef) { Write-Host "    $($u.Line)" }
if ($over.Count -or $undef.Count) { Write-Host "  -> fix these before submitting" }

# The figures must still trace to the reports. A paper is written once and the
# reports keep moving, so this is checked on every build rather than by memory.
Write-Host ""
& "..\..\.venv\Scripts\python.exe" check_paper_figures.py
