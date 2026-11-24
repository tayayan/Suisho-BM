[CmdletBinding()]
param(
    [string]$Msys2Root = 'C:\Tools\msys2',
    [ValidateSet('SSE2', 'SSE41', 'SSE42', 'AVX2', 'AVX512', 'ZEN2', 'ZEN3')]
    [string]$TargetCpu = 'AVX2',
    [ValidateRange(1, 256)]
    [int]$Jobs = [Environment]::ProcessorCount,
    # Recompile every object (after changing the compiler or TargetCpu).
    [switch]$Rebuild,
    # Profile-guided optimization: build, solve the positions in pgo.sfen
    # (1 second each), then build again with the profile (a few minutes).
    [switch]$Pgo
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$bash = Join-Path $Msys2Root 'usr\bin\bash.exe'
if (-not (Test-Path -LiteralPath $bash -PathType Leaf)) {
    throw "MSYS2 bash.exe was not found: $bash"
}

$root = [IO.Path]::GetFullPath($PSScriptRoot)
$unixRoot = '/' + $root.Substring(0, 1).ToLowerInvariant() + $root.Substring(2).Replace('\', '/')
$clean = if ($Rebuild -or $Pgo) { 'make clean && ' } else { '' }
$make = "make -j$Jobs TARGET_CPU=$TargetCpu"
$command = "cd '$unixRoot/source' && $clean$make"
if ($Pgo) {
    $train = "../Suisho-BM.exe bench hash 512 threads 1 limit 1000 type mate file ../pgo.sfen , quit > /dev/null"
    $command = "cd '$unixRoot/source' && $clean$make PGO_GEN=1 && $train && $make -B PGO_USE=1"
}

$oldMsystem = $env:MSYSTEM
$oldPreference = $ErrorActionPreference
try {
    $env:MSYSTEM = 'MINGW64'
    # The compiler writes warnings to stderr; do not treat them as errors.
    $ErrorActionPreference = 'Continue'
    & $bash -lc $command 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ -match 'error' }
    $exitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $oldPreference
    $env:MSYSTEM = $oldMsystem
}
if ($exitCode -ne 0) { throw "Build failed (exit code $exitCode)." }
Write-Output "Built: $(Join-Path $root 'Suisho-BM.exe')"
