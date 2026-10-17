param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$destination = Join-Path $projectRoot 'output\moss_json_merger'
$buildPath = Join-Path $projectRoot 'tmp\moss_json_merger_build'
New-Item -ItemType Directory -Force -Path $destination,$buildPath | Out-Null
& $Python -m PyInstaller --noconfirm --windowed --onefile --name MOSS_JSON_Merger --distpath $destination --workpath $buildPath --specpath $buildPath (Join-Path $PSScriptRoot 'app.py')
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed.' }
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'README.md') -Destination $destination -Force
$examplesPath = Join-Path $PSScriptRoot 'examples'
foreach ($sourceFile in Get-ChildItem -LiteralPath $examplesPath -File -Recurse) {
    $relativePath = $sourceFile.FullName.Substring($examplesPath.Length).TrimStart('\')
    $targetFile = Join-Path (Join-Path $destination 'examples') $relativePath
    New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($targetFile)) | Out-Null
    Copy-Item -LiteralPath $sourceFile.FullName -Destination $targetFile -Force
}
Write-Output (Join-Path $destination 'MOSS_JSON_Merger.exe')
